import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import torch

MODEL_UTILS = os.path.join(os.getcwd(), "utils", "model")
if MODEL_UTILS not in sys.path:
    sys.path.insert(1, MODEL_UTILS)

from utils.data.load_data import create_data_loaders
from utils.data.transforms import DataTransform


class ScoreAlignedDataTransformTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.ones(9, dtype=np.float32)
        self.kspace = np.ones((2, 9), dtype=np.complex64)
        self.target = np.ones((9, 9), dtype=np.float32)
        self.attrs = {"max": np.float32(1.0)}

    def test_default_training_tuple_remains_six_items(self):
        sample = DataTransform(False, "max")(
            self.mask, self.kspace, self.target, self.attrs, "case.h5", 0
        )

        self.assertEqual(len(sample), 6)

    def test_forward_tuple_excludes_score_metadata_even_when_requested(self):
        sample = DataTransform(True, -1, score_aligned=True)(
            self.mask, self.kspace, -1, -1, "case.h5", 0,
            score_metadata={"must_not_leak": torch.tensor(1)},
        )

        self.assertEqual(len(sample), 6)

    def test_score_training_collates_boolean_mask_and_preserves_float32_target(self):
        context = {
            "acceleration": 4,
            "boxes": [(0, 0, 9, 9)],
            "max_boxes": 2,
            "full_weight": 0.5,
            "box_weight": 0.75,
        }
        sample = DataTransform(False, "max", score_aligned=True)(
            self.mask,
            self.kspace,
            self.target,
            self.attrs,
            "case_acc4.h5",
            0,
            score_metadata=context,
        )

        self.assertEqual(len(sample), 7)
        metadata = sample[6]
        self.assertEqual(metadata["foreground_mask"].dtype, torch.bool)
        self.assertEqual(metadata["foreground_mask"].element_size(), 1)
        self.assertEqual(sample[2].dtype, torch.float32)
        torch.testing.assert_close(sample[2], torch.from_numpy(self.target))


class ScoreAlignedLoaderTests(unittest.TestCase):
    def _write_volume(self, root, name, annotations, num_slices=2):
        image_dir = root / "image"
        kspace_dir = root / "kspace"
        image_dir.mkdir(parents=True, exist_ok=True)
        kspace_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(image_dir / name, "w") as hf:
            hf.create_dataset(
                "image_label",
                data=np.ones((num_slices, 9, 9), dtype=np.float32),
            )
            hf.attrs["max"] = np.float32(1.0)
            hf.attrs["annotations"] = annotations
        with h5py.File(kspace_dir / name, "w") as hf:
            hf.create_dataset(
                "kspace",
                data=np.ones((num_slices, 2, 9), dtype=np.complex64),
            )
            hf.create_dataset("mask", data=np.ones(9, dtype=np.float32))

    def _args(self, batch_size=3):
        return SimpleNamespace(
            max_key="max",
            target_key="image_label",
            input_key="kspace",
            batch_size=batch_size,
        )

    def test_default_slice_data_preserves_legacy_transform_call(self):
        from utils.data.load_data import SliceData

        with tempfile.TemporaryDirectory(prefix="legacy-loader-") as tmp:
            root = Path(tmp)
            self._write_volume(root, "case.h5", json.dumps({}), num_slices=1)
            calls = []

            def legacy_transform(mask, input_data, target, attrs, fname, dataslice):
                calls.append((fname, dataslice))
                return mask, input_data, target, attrs, fname, dataslice

            dataset = SliceData(
                root,
                legacy_transform,
                input_key="kspace",
                target_key="image_label",
            )
            sample = dataset[0]

            self.assertEqual(calls, [("case.h5", 0)])
            self.assertEqual(len(sample), 6)

    def test_loader_derives_counts_weights_and_collates_mixed_box_counts(self):
        with tempfile.TemporaryDirectory(prefix="score-loader-") as tmp:
            root = Path(tmp)
            self._write_volume(
                root,
                "case_acc4_a.h5",
                json.dumps({
                    "0": [],
                    "1": [{"x": 0, "y": 0, "width": 9, "height": 9}],
                }),
            )
            self._write_volume(
                root,
                "case_acc8_a.h5",
                json.dumps({
                    "0": [
                        {"x": 0, "y": 0, "width": 9, "height": 9},
                        {"x": 1, "y": 1, "width": 8, "height": 8},
                    ],
                    "1": [],
                }),
            )

            loader = create_data_loaders(
                root, self._args(), score_aligned=True
            )
            self.assertEqual(
                loader.dataset.score_counts,
                {"slices": {4: 2, 8: 2}, "boxes": {4: 1, 8: 2}},
            )
            self.assertEqual(
                loader.dataset.score_weights,
                {"full": {4: 0.5, 8: 0.5}, "box": {4: 1.0, 8: 0.5}},
            )
            total_slices = 4
            for kind, counts in loader.dataset.score_counts.items():
                weight_kind = "full" if kind == "slices" else "box"
                for acceleration in (4, 8):
                    mass = (
                        counts[acceleration]
                        * loader.dataset.score_weights[weight_kind][acceleration]
                        / total_slices
                    )
                    self.assertEqual(mass, 0.25)

            first = next(iter(loader))
            metadata = first[6]
            self.assertEqual(metadata["box_count"].tolist(), [0, 1, 2])
            self.assertEqual(tuple(metadata["boxes"].shape), (3, 2, 4))
            self.assertEqual(metadata["foreground_mask"].dtype, torch.bool)
            self.assertEqual(metadata["foreground_mask"].element_size(), 1)

    def test_oracle_counts_derive_verified_weights_without_hardcoding(self):
        from utils.data.load_data import derive_score_weights

        weights = derive_score_weights(
            {4: 2336, 8: 2315}, {4: 1417, 8: 1176}
        )

        self.assertAlmostEqual(weights["full"][4], 0.4977525684931507)
        self.assertAlmostEqual(weights["full"][8], 0.5022678185745141)
        self.assertAlmostEqual(weights["box"][4], 0.8205716302046577)
        self.assertAlmostEqual(weights["box"][8], 0.9887329931972789)

    def test_score_loader_rejects_malformed_acceleration_and_annotations(self):
        malformed = [
            ("case_unknown.h5", json.dumps({})),
            ("case_acc4.h5", "not-json"),
            ("case_acc4.h5", json.dumps([])),
            ("case_acc4.h5", json.dumps({"0": [{"x": 0}]})),
            ("case_acc4.h5", json.dumps({"2": []})),
        ]
        for name, annotations in malformed:
            with self.subTest(name=name, annotations=annotations):
                with tempfile.TemporaryDirectory(prefix="score-invalid-") as tmp:
                    root = Path(tmp)
                    self._write_volume(root, name, annotations, num_slices=1)
                    with self.assertRaises(ValueError):
                        create_data_loaders(
                            root, self._args(), score_aligned=True
                        )

    def test_score_loader_rejects_image_kspace_manifest_mismatch(self):
        for missing_side in ("image", "kspace"):
            with self.subTest(missing_side=missing_side):
                with tempfile.TemporaryDirectory(prefix="score-manifest-") as tmp:
                    root = Path(tmp)
                    self._write_volume(
                        root,
                        "case_acc4_a.h5",
                        json.dumps({"0": [
                            {"x": 0, "y": 0, "width": 9, "height": 9}
                        ]}),
                        num_slices=1,
                    )
                    self._write_volume(
                        root,
                        "case_acc8_a.h5",
                        json.dumps({"0": [
                            {"x": 0, "y": 0, "width": 9, "height": 9}
                        ]}),
                        num_slices=1,
                    )
                    (root / missing_side / "case_acc8_a.h5").unlink()

                    with self.assertRaisesRegex(ValueError, "filename sets"):
                        create_data_loaders(
                            root, self._args(), score_aligned=True
                        )

    def test_score_loader_rejects_per_volume_slice_count_mismatch(self):
        with tempfile.TemporaryDirectory(prefix="score-slices-") as tmp:
            root = Path(tmp)
            for acceleration in (4, 8):
                self._write_volume(
                    root,
                    f"case_acc{acceleration}_a.h5",
                    json.dumps({"0": [
                        {"x": 0, "y": 0, "width": 9, "height": 9}
                    ], "1": []}),
                )
            path = root / "kspace" / "case_acc8_a.h5"
            with h5py.File(path, "w") as hf:
                hf.create_dataset(
                    "kspace", data=np.ones((1, 2, 9), dtype=np.complex64)
                )
                hf.create_dataset("mask", data=np.ones(9, dtype=np.float32))

            with self.assertRaisesRegex(ValueError, "slice count"):
                create_data_loaders(root, self._args(), score_aligned=True)

    def test_score_loader_uses_side_specific_dataset_keys_for_slice_parity(self):
        with tempfile.TemporaryDirectory(prefix="score-side-key-") as tmp:
            root = Path(tmp)
            self._write_volume(
                root,
                "case_acc4_a.h5",
                json.dumps({"0": [
                    {"x": 0, "y": 0, "width": 9, "height": 9}
                ]}),
                num_slices=1,
            )
            self._write_volume(
                root,
                "case_acc8_a.h5",
                json.dumps({
                    "0": [{"x": 0, "y": 0, "width": 9, "height": 9}],
                    "1": [],
                }),
                num_slices=2,
            )
            image_path = root / "image" / "case_acc4_a.h5"
            with h5py.File(image_path, "a") as hf:
                hf.create_dataset(
                    "kspace", data=np.ones((2, 2, 9), dtype=np.complex64)
                )
            kspace_path = root / "kspace" / "case_acc4_a.h5"
            with h5py.File(kspace_path, "w") as hf:
                hf.create_dataset(
                    "kspace", data=np.ones((2, 2, 9), dtype=np.complex64)
                )
                hf.create_dataset("mask", data=np.ones(9, dtype=np.float32))

            with self.assertRaisesRegex(ValueError, "slice count"):
                create_data_loaders(root, self._args(), score_aligned=True)

    def test_score_loader_rejects_missing_side_specific_dataset_key(self):
        with tempfile.TemporaryDirectory(prefix="score-missing-key-") as tmp:
            root = Path(tmp)
            for acceleration in (4, 8):
                self._write_volume(
                    root,
                    f"case_acc{acceleration}_a.h5",
                    json.dumps({"0": [
                        {"x": 0, "y": 0, "width": 9, "height": 9}
                    ]}),
                    num_slices=1,
                )
            path = root / "kspace" / "case_acc4_a.h5"
            with h5py.File(path, "w") as hf:
                hf.create_dataset(
                    "image_label", data=np.ones((1, 9, 9), dtype=np.float32)
                )
                hf.create_dataset("mask", data=np.ones(9, dtype=np.float32))

            with self.assertRaisesRegex(ValueError, "kspace"):
                create_data_loaders(root, self._args(), score_aligned=True)

    def test_score_loader_rejects_annotation_integer_outside_int64(self):
        with tempfile.TemporaryDirectory(prefix="score-int64-") as tmp:
            root = Path(tmp)
            extreme = 10 ** 30
            self._write_volume(
                root,
                "case_acc4_a.h5",
                json.dumps({"0": [
                    {"x": -extreme, "y": 0, "width": extreme + 9, "height": 9}
                ]}),
                num_slices=1,
            )
            self._write_volume(
                root,
                "case_acc8_a.h5",
                json.dumps({"0": [
                    {"x": 0, "y": 0, "width": 9, "height": 9}
                ]}),
                num_slices=1,
            )

            with self.assertRaisesRegex(ValueError, "int64"):
                create_data_loaders(root, self._args(), score_aligned=True)

    def test_score_loader_weights_only_evaluator_accepted_boxes(self):
        with tempfile.TemporaryDirectory(prefix="score-accepted-") as tmp:
            root = Path(tmp)
            self._write_volume(
                root,
                "case_acc4_a.h5",
                json.dumps({"0": [
                    {"x": 0, "y": 0, "width": 9, "height": 9},
                    {"x": -5, "y": 0, "width": 11, "height": 9},
                ]}),
                num_slices=1,
            )
            self._write_volume(
                root,
                "case_acc8_a.h5",
                json.dumps({"0": [
                    {"x": 0, "y": 0, "width": 9, "height": 9}
                ]}),
                num_slices=1,
            )

            loader = create_data_loaders(
                root, self._args(), score_aligned=True
            )

            self.assertEqual(loader.dataset.score_counts["boxes"], {4: 1, 8: 1})
            self.assertEqual(loader.dataset.score_weights["box"], {4: 0.5, 8: 0.5})
            metadata = loader.dataset[0][6]
            self.assertEqual(metadata["box_count"].item(), 2)

    def test_score_loader_rejects_subgroup_without_accepted_boxes(self):
        with tempfile.TemporaryDirectory(prefix="score-no-box-cell-") as tmp:
            root = Path(tmp)
            self._write_volume(
                root,
                "case_acc4_a.h5",
                json.dumps({"0": [
                    {"x": -5, "y": 0, "width": 11, "height": 9}
                ]}),
                num_slices=1,
            )
            self._write_volume(
                root,
                "case_acc8_a.h5",
                json.dumps({"0": [
                    {"x": 0, "y": 0, "width": 9, "height": 9}
                ]}),
                num_slices=1,
            )

            with self.assertRaisesRegex(ValueError, "box counts"):
                create_data_loaders(root, self._args(), score_aligned=True)


class ScoreAlignedLossTests(unittest.TestCase):
    def _metadata(self, box_counts, boxes=None, mask=None):
        batch_size = len(box_counts)
        max_boxes = max(max(box_counts), 1)
        padded = torch.zeros(batch_size, max_boxes, 4, dtype=torch.int64)
        if boxes is not None:
            for sample_index, sample_boxes in enumerate(boxes):
                for box_index, box in enumerate(sample_boxes):
                    padded[sample_index, box_index] = torch.tensor(box)
        if mask is None:
            mask = torch.ones(batch_size, 12, 12, dtype=torch.bool)
        return {
            "acceleration": torch.tensor(
                [4 if index % 2 == 0 else 8 for index in range(batch_size)]
            ),
            "boxes": padded,
            "box_count": torch.tensor(box_counts, dtype=torch.int64),
            "foreground_mask": mask,
            "full_weight": torch.full(
                (batch_size,), 0.3, dtype=torch.float64
            ),
            "box_weight": torch.full(
                (batch_size,), 0.7, dtype=torch.float64
            ),
        }

    def test_perfect_reconstruction_is_exact_zero_for_zero_one_and_multiple_boxes(self):
        from utils.common.metrics import ScoreAlignedLoss

        torch.manual_seed(340700)
        target = torch.rand(3, 12, 12, dtype=torch.float32)
        boxes = [
            [],
            [(0, 0, 7, 7)],
            [(0, 0, 8, 8), (4, 4, 8, 8)],
        ]
        metadata = self._metadata([0, 1, 2], boxes=boxes)

        actual = ScoreAlignedLoss()(target, target, torch.ones(3), metadata)

        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual.item(), 0.0)

    def test_loss_uses_weighted_complements_and_sums_separate_boxes(self):
        from utils.common.metrics import (
            SSIM,
            ScoreAlignedLoss,
            ssim_bbox_tensor,
            ssim_full_tensor,
        )

        torch.manual_seed(340701)
        target = torch.rand(1, 12, 12)
        recon = torch.rand(1, 12, 12, requires_grad=True)
        boxes = [[(0, 0, 9, 9), (3, 3, 9, 9)]]
        metadata = self._metadata([2], boxes=boxes)
        ssim = SSIM()
        full = ssim_full_tensor(
            ssim, recon[0], target[0], metadata["foreground_mask"][0].float(), 1.0
        )
        bbox_values = [
            ssim_bbox_tensor(
                ssim,
                recon[0],
                target[0],
                {"x": x, "y": y, "width": width, "height": height},
                1.0,
            )
            for x, y, width, height in boxes[0]
        ]
        expected = 0.3 * (1 - full) + 0.7 * sum(
            1 - value for value in bbox_values
        )
        constant_one_mutant = 1 - (
            0.3 * full + 0.7 * sum(bbox_values)
        )

        actual = ScoreAlignedLoss()(recon, target, torch.ones(1), metadata)

        torch.testing.assert_close(actual, expected)
        self.assertFalse(torch.isclose(actual, constant_one_mutant).item())
        averaged_boxes = 0.3 * (1 - full) + 0.7 * sum(
            1 - value for value in bbox_values
        ) / len(bbox_values)
        self.assertFalse(torch.isclose(actual, averaged_boxes).item())
        actual.backward()
        self.assertTrue(torch.isfinite(recon.grad).all())
        self.assertGreater(torch.count_nonzero(recon.grad).item(), 0)

    def test_loss_skips_only_boxes_smaller_than_ssim_window(self):
        from utils.common.metrics import ScoreAlignedLoss

        torch.manual_seed(340702)
        target = torch.rand(1, 12, 12)
        recon = torch.rand(1, 12, 12)
        one_box = self._metadata([1], boxes=[[(0, 0, 7, 7)]])
        with_small = self._metadata(
            [2], boxes=[[(0, 0, 7, 7), (0, 0, 6, 12)]]
        )

        expected = ScoreAlignedLoss()(recon, target, torch.ones(1), one_box)
        actual = ScoreAlignedLoss()(recon, target, torch.ones(1), with_small)

        torch.testing.assert_close(actual, expected)

    def test_loss_rejects_malformed_batched_metadata(self):
        from utils.common.metrics import ScoreAlignedLoss

        output = torch.ones(1, 12, 12)
        target = torch.ones_like(output)
        valid = self._metadata([1], boxes=[[(0, 0, 7, 7)]])
        malformed = []
        missing = dict(valid)
        missing.pop("acceleration")
        malformed.append(missing)
        for key, bad_value in (
            ("acceleration", torch.tensor([5])),
            ("box_count", torch.tensor([2])),
            ("foreground_mask", torch.ones(1, 12, 12)),
            ("full_weight", torch.tensor([float("nan")], dtype=torch.float64)),
        ):
            candidate = dict(valid)
            candidate[key] = bad_value
            malformed.append(candidate)
        invalid_box = dict(valid)
        invalid_box["boxes"] = torch.tensor([[[0, 0, 0, 7]]])
        malformed.append(invalid_box)
        nonzero_padding = self._metadata([0], boxes=[[]])
        nonzero_padding["boxes"][0, 0] = torch.tensor([0, 0, 7, 7])
        malformed.append(nonzero_padding)

        for metadata in malformed:
            with self.subTest(metadata=metadata):
                with self.assertRaises(ValueError):
                    ScoreAlignedLoss()(output, target, torch.ones(1), metadata)


class ScoreAlignedTrainingIntegrationTests(ScoreAlignedLoaderTests):
    def test_train_epoch_runs_score_loss_and_updates_parameter(self):
        from utils.common.metrics import ScoreAlignedLoss
        from utils.learning.train_part import train_epoch

        class ScalarImageModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.5))

            def forward(self, kspace, mask):
                return self.scale.expand(kspace.shape[0], 12, 12)

        model = ScalarImageModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        metadata = ScoreAlignedLossTests()._metadata(
            [1], boxes=[[(0, 0, 7, 7)]]
        )
        batch = (
            torch.ones(1, 1, 1, 12, 1, dtype=torch.uint8),
            torch.ones(1, 2, 12, 2),
            torch.ones(1, 12, 12),
            torch.ones(1),
            ["case_acc4.h5"],
            torch.tensor([0]),
            metadata,
        )
        args = SimpleNamespace(
            score_aligned_loss=True, report_interval=100, num_epochs=1
        )
        before = model.scale.detach().clone()

        loss, _ = train_epoch(
            args, 0, model, [batch], optimizer, ScoreAlignedLoss()
        )

        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(before, model.scale.detach()))

    def test_train_wires_opt_in_loss_and_training_metadata_only(self):
        import utils.learning.train_part as train_part

        args = SimpleNamespace(
            GPU_NUM=0,
            cascade=1,
            chans=1,
            sens_chans=1,
            lr=3e-4,
            resume_checkpoint=None,
            num_epochs=0,
            data_path_train=Path("train"),
            data_path_val=Path("val"),
            score_aligned_loss=True,
        )
        model = torch.nn.Linear(1, 1)
        score_loss = torch.nn.Identity()
        with patch.object(
            train_part.torch.cuda, "is_available", return_value=False
        ), patch.object(
            train_part, "VarNet", return_value=model
        ), patch.object(
            train_part, "ScoreAlignedLoss", return_value=score_loss
        ) as score_constructor, patch.object(
            train_part, "SSIMLoss"
        ) as default_constructor, patch.object(
            train_part, "create_data_loaders", side_effect=[[], []]
        ) as create_loaders:
            train_part.train(args)

        score_constructor.assert_called_once_with()
        default_constructor.assert_not_called()
        self.assertEqual(create_loaders.call_count, 2)
        self.assertEqual(
            create_loaders.call_args_list[0].kwargs,
            {
                "data_path": args.data_path_train,
                "args": args,
                "shuffle": True,
                "score_aligned": True,
            },
        )
        self.assertEqual(
            create_loaders.call_args_list[1].kwargs,
            {"data_path": args.data_path_val, "args": args},
        )

    def test_score_loader_creation_consumes_no_torch_rng_and_preserves_sampler_order(self):
        with tempfile.TemporaryDirectory(prefix="score-rng-") as tmp:
            root = Path(tmp)
            self._write_volume(
                root,
                "case_acc4_a.h5",
                json.dumps({"0": [
                    {"x": 0, "y": 0, "width": 9, "height": 9}
                ], "1": []}),
            )
            self._write_volume(
                root,
                "case_acc8_a.h5",
                json.dumps({"0": [
                    {"x": 0, "y": 0, "width": 9, "height": 9}
                ], "1": []}),
            )

            torch.manual_seed(340703)
            before_control = torch.get_rng_state().clone()
            control = create_data_loaders(
                root, self._args(batch_size=2), shuffle=True
            )
            after_control = torch.get_rng_state().clone()

            torch.manual_seed(340703)
            before_score = torch.get_rng_state().clone()
            score = create_data_loaders(
                root,
                self._args(batch_size=2),
                shuffle=True,
                score_aligned=True,
            )
            after_score = torch.get_rng_state().clone()

            torch.testing.assert_close(before_control, after_control)
            torch.testing.assert_close(before_score, after_score)
            torch.testing.assert_close(after_control, after_score)

            torch.manual_seed(340704)
            control_order = [
                name for batch in control for name in batch[4]
            ]
            torch.manual_seed(340704)
            score_order = [name for batch in score for name in batch[4]]
            self.assertEqual(control_order, score_order)


if __name__ == "__main__":
    unittest.main()