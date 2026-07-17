import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from utils.promptmr.contracts import (
    PROMPTMR_PLUS_RECIPE,
    adjacent_slice_indices,
    parse_acceleration_filename,
    validate_model_family_args,
)
from utils.promptmr.planner import collect_dataset_stats, estimate_run
from utils.promptmr.runtime import _apply_upstream_compatibility_shims


class PromptMRTrainingContractTests(unittest.TestCase):
    def test_runtime_explicitly_loads_activation_checkpoint_module(self):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse(
            (root / "utils/promptmr/runtime.py").read_text(encoding="utf-8")
        )
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertIn("torch.utils.checkpoint", imported)

    def test_upstream_promptmr_blocks_receive_required_n_buffer_attribute(self):
        model = SimpleNamespace(
            cascades=[
                SimpleNamespace(model=SimpleNamespace(n_buffer=4)),
                SimpleNamespace(model=SimpleNamespace(n_buffer=4)),
            ]
        )

        _apply_upstream_compatibility_shims(model)

        self.assertEqual([cascade.n_buffer for cascade in model.cascades], [4, 4])

    def test_exact_upstream_recipe(self):
        recipe = PROMPTMR_PLUS_RECIPE
        self.assertEqual(recipe["architecture"]["num_cascades"], 12)
        self.assertEqual(recipe["architecture"]["num_adj_slices"], 5)
        self.assertTrue(recipe["architecture"]["adaptive_input"])
        self.assertEqual(recipe["architecture"]["n_buffer"], 4)
        self.assertEqual(recipe["architecture"]["n_history"], 11)
        self.assertTrue(recipe["architecture"]["use_sens_adj"])
        self.assertTrue(recipe["runtime"]["activation_checkpointing"])
        self.assertTrue(recipe["runtime"]["compute_sens_per_coil"])
        self.assertEqual(recipe["optimizer"], {
            "name": "AdamW", "lr": 1e-4, "weight_decay": 1e-2
        })
        self.assertEqual(recipe["scheduler"], {
            "name": "StepLR", "step_size": 35, "gamma": 0.1
        })
        self.assertEqual(recipe["loss"], {
            "name": "SSIMLoss", "win_size": 7, "k1": 0.01, "k2": 0.03
        })
        self.assertEqual(recipe["training"], {
            "max_epochs": 45,
            "gradient_clip_val": 0.01,
            "precision": "32-true",
            "train_crop": [384, 384],
            "validation_crop": None,
        })

    def test_exact_acceleration_tokens(self):
        self.assertEqual(parse_acceleration_filename("brain_acc4.h5"), 4)
        self.assertEqual(parse_acceleration_filename("brain_acc8_extra.h5"), 8)
        for invalid in (
            "brain.h5", "brain_acc40.h5", "brain_xacc4.h5",
            "brain_acc4_acc8.h5", "brain_acc4_acc4.h5",
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    parse_acceleration_filename(invalid)

    def test_five_slice_boundary_replication(self):
        self.assertEqual(adjacent_slice_indices(0, 4, 5), (0, 0, 0, 1, 2))
        self.assertEqual(adjacent_slice_indices(1, 4, 5), (0, 0, 1, 2, 3))
        self.assertEqual(adjacent_slice_indices(3, 4, 5), (1, 2, 3, 3, 3))
        with self.assertRaises(ValueError):
            adjacent_slice_indices(0, 4, 4)

    def test_model_family_rejects_promptmr_incompatible_legacy_options(self):
        legacy = SimpleNamespace(
            model_family="varnet", cascade=12, chans=18, sens_chans=4,
            score_aligned_loss=False,
        )
        self.assertEqual(validate_model_family_args(legacy), "varnet")
        prompt = SimpleNamespace(
            model_family="promptmr_plus", cascade=12, chans=18,
            sens_chans=4, score_aligned_loss=False,
        )
        self.assertEqual(validate_model_family_args(prompt), "promptmr_plus")
        prompt.score_aligned_loss = True
        with self.assertRaisesRegex(ValueError, "score-aligned"):
            validate_model_family_args(prompt)

    def test_recipe_json_matches_python_contract(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "promptmr_plus_training.json"
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), PROMPTMR_PLUS_RECIPE)

    def test_planner_rejects_dataset_missing_training_contract_fields(self):
        import h5py
        import numpy as np

        cases = (
            "mask", "target_file", "target_dataset", "max_attr",
            "kspace_dtype", "mask_shape", "max_value",
        )
        for missing in cases:
            with self.subTest(missing=missing), tempfile.TemporaryDirectory(
                prefix="promptmr-plan-invalid-"
            ) as tmp:
                root = Path(tmp)
                kspace = root / "kspace"
                image = root / "image"
                kspace.mkdir()
                image.mkdir()
                for acceleration in (4, 8):
                    name = f"volume_acc{acceleration}.h5"
                    with h5py.File(kspace / name, "w") as handle:
                        kspace_dtype = (
                            np.float32 if missing == "kspace_dtype" else np.complex64
                        )
                        handle.create_dataset(
                            "kspace", data=np.zeros((1, 1, 2, 2), dtype=kspace_dtype)
                        )
                        if missing != "mask":
                            mask_shape = (2, 2) if missing == "mask_shape" else (2,)
                            handle.create_dataset(
                                "mask", data=np.ones(mask_shape, dtype=np.uint8)
                            )
                    if missing == "target_file":
                        continue
                    with h5py.File(image / name, "w") as handle:
                        if missing != "target_dataset":
                            handle.create_dataset(
                                "image_label", data=np.zeros((1, 2, 2), dtype=np.float32)
                            )
                        if missing != "max_attr":
                            handle.attrs["max"] = (
                                np.nan if missing == "max_value" else 1.0
                            )
                with self.assertRaises((FileNotFoundError, ValueError)):
                    collect_dataset_stats(root)

    def test_planner_accepts_complete_training_contract_metadata(self):
        import h5py
        import numpy as np

        with tempfile.TemporaryDirectory(prefix="promptmr-plan-valid-") as tmp:
            root = Path(tmp)
            kspace = root / "kspace"
            image = root / "image"
            kspace.mkdir()
            image.mkdir()
            for acceleration in (4, 8):
                name = f"volume_acc{acceleration}.h5"
                with h5py.File(kspace / name, "w") as handle:
                    handle.create_dataset(
                        "kspace", data=np.zeros((1, 1, 2, 2), dtype=np.complex64)
                    )
                    handle.create_dataset("mask", data=np.ones(2, dtype=np.uint8))
                with h5py.File(image / name, "w") as handle:
                    handle.create_dataset(
                        "image_label", data=np.zeros((1, 2, 2), dtype=np.float32)
                    )
                    handle.attrs["max"] = 1.0
            stats = collect_dataset_stats(root)

        self.assertEqual(stats["volumes"], {4: 1, 8: 1})
        self.assertEqual(stats["slices"], {4: 1, 8: 1})
        self.assertGreater(stats["max_sample_bytes"], 0)
        self.assertGreater(stats["total_volume_bytes"], 0)

    def test_planner_counts_4x_8x_without_creating_output_directories(self):
        with tempfile.TemporaryDirectory(prefix="promptmr-plan-") as tmp:
            root = Path(tmp)
            kspace = root / "kspace"
            kspace.mkdir()
            for name in ("a_acc4.h5", "b_acc4.h5", "c_acc8.h5"):
                (kspace / name).touch()
            before = sorted(path.relative_to(root) for path in root.rglob("*"))
            stats = collect_dataset_stats(root, metadata_reader=lambda _: {
                "slices": 10, "sample_bytes": 1024, "target_bytes": 256
            })
            estimate = estimate_run(
                stats, batch_size=1, epochs=45, retain_val_epochs=True,
                checkpoint_reserve_gib=8.0,
            )
            after = sorted(path.relative_to(root) for path in root.rglob("*"))

        self.assertEqual(stats["volumes"], {4: 2, 8: 1})
        self.assertEqual(stats["slices"], {4: 20, 8: 10})
        self.assertEqual(estimate["steps_per_epoch"], 30)
        self.assertEqual(estimate["total_steps"], 1350)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
