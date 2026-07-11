import contextlib
import csv
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


def load_evaluate_val_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_val.py"
    spec = importlib.util.spec_from_file_location("evaluate_val_for_test", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EvaluateValAggregationTests(unittest.TestCase):
    def test_leaderboard_row_equal_weights_accelerations_not_observations(self):
        module = load_evaluate_val_module()
        rows = [
            {
                "scope": "acc4",
                "ssim_full_mean": 0.9,
                "ssim_bbox_mean": 0.8,
                "ssim_full_count": 400,
                "ssim_bbox_count": 100,
                "volumes": 15,
                "slices": 400,
                "bbox_annotations": 100,
            },
            {
                "scope": "acc8",
                "ssim_full_mean": 0.7,
                "ssim_bbox_mean": 0.6,
                "ssim_full_count": 40,
                "ssim_bbox_count": 10,
                "volumes": 15,
                "slices": 40,
                "bbox_annotations": 10,
            },
        ]

        row = module.leaderboard_equal_acc_row(rows)

        self.assertEqual(row["scope"], "leaderboard_equal_acc")
        self.assertAlmostEqual(row["ssim_full_mean"], 0.8)
        self.assertAlmostEqual(row["ssim_bbox_mean"], 0.7)
        self.assertAlmostEqual(row["quality_score"], 0.75)
        self.assertEqual(row["aggregation"], "equal mean of acc4 and acc8")

    def test_leaderboard_row_rejects_missing_acceleration_metrics(self):
        module = load_evaluate_val_module()
        rows = [
            {
                "scope": "acc4",
                "ssim_full_mean": 0.9,
                "ssim_bbox_mean": 0.8,
                "ssim_full_count": 400,
                "ssim_bbox_count": 100,
                "volumes": 15,
                "slices": 400,
                "bbox_annotations": 100,
            }
        ]

        with self.assertRaisesRegex(ValueError, "acc4 and acc8"):
            module.leaderboard_equal_acc_row(rows)


class EvaluateValStrictModeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.target_dir = self.root / "targets"
        self.recon_dir = self.root / "recons"
        self.out_dir = self.root / "metrics"
        self.target_dir.mkdir()
        self.recon_dir.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def run_main(self, *extra_args):
        module = load_evaluate_val_module()
        argv = [
            "--target-dir", str(self.target_dir),
            "--recon-dir", str(self.recon_dir),
            "--out-dir", str(self.out_dir),
            *extra_args,
        ]
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                module.main(argv)
        self.assertEqual(raised.exception.code, 1)
        return stderr.getvalue()

    def run_main_successfully(self, *extra_args):
        module = load_evaluate_val_module()
        argv = [
            "--target-dir", str(self.target_dir),
            "--recon-dir", str(self.recon_dir),
            "--out-dir", str(self.out_dir),
            *extra_args,
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(io.StringIO()):
                module.main(argv)

    def strict_args(self, volumes=1, slices=1, boxes=1):
        return (
            "--require-complete",
            "--expected-volumes", str(volumes),
            "--expected-slices", str(slices),
            "--expected-boxes", str(boxes),
        )

    def write_volume(self, directory, name, data=None, *, stored_max=1.0, annotations=None):
        if data is None:
            data = np.ones((1, 16, 16), dtype=np.float32)
        key = "image_label" if directory == self.target_dir else "reconstruction"
        with h5py.File(directory / name, "w") as hf:
            hf.create_dataset(key, data=data)
            if directory == self.target_dir:
                if stored_max is not None:
                    hf.attrs["max"] = stored_max
                if annotations is not None:
                    hf.attrs["annotations"] = annotations

    def write_complete_acc_pair(self):
        annotations = json.dumps(
            {"0": [{"x": 0, "y": 0, "width": 12, "height": 12}]}
        )
        for name in ("brain_acc4.h5", "brain_acc8.h5"):
            self.write_volume(self.target_dir, name, annotations=annotations)
            self.write_volume(self.recon_dir, name)

    def test_require_complete_requires_explicit_coverage_expectations(self):
        message = self.run_main("--require-complete")

        self.assertIn("requires --expected-volumes", message)

    def test_require_complete_rejects_reconstruction_without_target_file(self):
        self.write_volume(self.recon_dir, "brain_acc4.h5")

        message = self.run_main(*self.strict_args())

        self.assertIn("missing target file", message)

    def test_require_complete_rejects_target_without_reconstruction_file(self):
        self.write_volume(self.target_dir, "brain_acc4.h5")

        message = self.run_main(*self.strict_args())

        self.assertIn("missing reconstruction file", message)

    def test_require_complete_rejects_absent_stored_max(self):
        self.write_volume(self.target_dir, "brain_acc4.h5", stored_max=None)
        self.write_volume(self.recon_dir, "brain_acc4.h5")

        message = self.run_main(*self.strict_args())

        self.assertIn("missing stored max", message)

    def test_require_complete_rejects_malformed_stored_max(self):
        self.write_volume(self.target_dir, "brain_acc4.h5", stored_max="not-a-number")
        self.write_volume(self.recon_dir, "brain_acc4.h5")

        message = self.run_main(*self.strict_args())

        self.assertIn("stored max must be a finite positive scalar", message)

    def test_require_complete_rejects_nonfinite_stored_max(self):
        self.write_volume(self.target_dir, "brain_acc4.h5", stored_max=np.nan)
        self.write_volume(self.recon_dir, "brain_acc4.h5")

        message = self.run_main(*self.strict_args())

        self.assertIn("stored max must be a finite positive scalar", message)

    def test_require_complete_rejects_nonpositive_stored_max(self):
        self.write_volume(self.target_dir, "brain_acc4.h5", stored_max=0.0)
        self.write_volume(self.recon_dir, "brain_acc4.h5")

        message = self.run_main(*self.strict_args())

        self.assertIn("stored max must be a finite positive scalar", message)

    def test_infer_acc_name_requires_one_exact_filename_token(self):
        module = load_evaluate_val_module()
        expected = {
            "knee_acc4_1.h5": "acc4",
            "KNEE_ACC8_15.H5": "acc8",
            "brain_acc4.h5": "acc4",
            "brain_acc40.h5": "unknown",
            "brainacc4.h5": "unknown",
            "brain-acc8.h5": "unknown",
            "brain_acc4_acc8.h5": "unknown",
            "brain_acc4_acc4.h5": "unknown",
        }

        for name, acceleration in expected.items():
            with self.subTest(name=name):
                self.assertEqual(module.infer_acc_name(Path(name)), acceleration)

    def test_require_complete_rejects_unknown_acceleration(self):
        self.write_volume(self.target_dir, "brain.h5")
        self.write_volume(self.recon_dir, "brain.h5")

        message = self.run_main(*self.strict_args())

        self.assertIn("unknown acceleration", message)

    def test_require_complete_rejects_acc40_filename(self):
        self.write_volume(self.target_dir, "brain_acc40.h5", annotations="{}")
        self.write_volume(self.recon_dir, "brain_acc40.h5")

        message = self.run_main(*self.strict_args(boxes=0))

        self.assertIn("unknown acceleration", message)

    def test_require_complete_rejects_ambiguous_acceleration_filename(self):
        name = "brain_acc4_acc8.h5"
        self.write_volume(self.target_dir, name, annotations="{}")
        self.write_volume(self.recon_dir, name)

        message = self.run_main(*self.strict_args(boxes=0))

        self.assertIn("unknown acceleration", message)

    def test_require_complete_rejects_missing_annotations_even_when_aggregate_box_count_matches(self):
        valid_annotations = json.dumps(
            {"0": [{"x": 0, "y": 0, "width": 12, "height": 12}]}
        )
        volumes = {
            "brain_acc4_1.h5": None,
            "brain_acc4_2.h5": valid_annotations,
            "brain_acc8_1.h5": valid_annotations,
        }
        for name, annotations in volumes.items():
            self.write_volume(self.target_dir, name, annotations=annotations)
            self.write_volume(self.recon_dir, name)

        message = self.run_main(*self.strict_args(volumes=3, slices=3, boxes=2))

        self.assertIn("missing annotations attribute", message)

    def test_require_complete_rejects_invalid_annotations_top_level_values(self):
        module = load_evaluate_val_module()
        invalid_values = [None, "", "null", "true", "17", '"scalar"', [], [1]]

        for raw in invalid_values:
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ValueError, "annotations must be a JSON object"):
                    module.parse_annotations(raw, 1, require_complete=True)

    def test_require_complete_rejects_invalid_annotation_slice_keys(self):
        module = load_evaluate_val_module()
        invalid_objects = [
            {"slice": []},
            {"-1": []},
            {"01": []},
            {"1": []},
            {0: []},
        ]

        for annotations in invalid_objects:
            with self.subTest(annotations=annotations):
                with self.assertRaisesRegex(ValueError, "invalid annotation slice key"):
                    module.parse_annotations(annotations, 1, require_complete=True)

    def test_require_complete_rejects_invalid_annotation_slice_containers(self):
        module = load_evaluate_val_module()
        box = {"x": 0, "y": 0, "width": 12, "height": 12}
        invalid_containers = [None, "", True, 1, box, {"boxes": [box]}]

        for container in invalid_containers:
            with self.subTest(container=container):
                with self.assertRaisesRegex(ValueError, "must contain a list of boxes"):
                    module.parse_annotations(
                        {"0": container}, 1, require_complete=True
                    )

    def test_require_complete_rejects_adversarial_malformed_annotation_boxes(self):
        module = load_evaluate_val_module()
        valid_box = {"x": 0, "y": 0, "width": 12, "height": 12}
        malformed_boxes = [
            None,
            True,
            7,
            [],
            {"x": 0, "y": 0, "width": 12},
            {**valid_box, "x": "0"},
            {**valid_box, "x": True},
            {**valid_box, "x": 0.0},
            {**valid_box, "width": 0},
            {"bbox": [0, 0, 12, 12]},
        ]

        for box in malformed_boxes:
            with self.subTest(box=box):
                with self.assertRaisesRegex(ValueError, "malformed annotation box"):
                    module.parse_annotations(
                        {"0": [box]}, 1, require_complete=True
                    )

    def test_require_complete_accepts_documented_empty_annotations_object(self):
        module = load_evaluate_val_module()

        boxes, count, skipped = module.parse_annotations(
            "{}", 1, require_complete=True
        )

        self.assertEqual((dict(boxes), count, skipped), ({}, 0, 0))

    def test_require_complete_accepts_empty_object_for_one_zero_box_volume(self):
        valid_annotations = json.dumps(
            {"0": [{"x": 0, "y": 0, "width": 12, "height": 12}]}
        )
        volumes = {
            "brain_acc4_1.h5": "{}",
            "brain_acc4_2.h5": valid_annotations,
            "brain_acc8_1.h5": valid_annotations,
        }
        for name, annotations in volumes.items():
            self.write_volume(self.target_dir, name, annotations=annotations)
            self.write_volume(self.recon_dir, name)

        self.run_main_successfully(
            *self.strict_args(volumes=3, slices=3, boxes=2)
        )

        summary = json.loads(
            (self.out_dir / "metrics.json").read_text(encoding="utf-8")
        )
        overall = next(row for row in summary["rows"] if row["scope"] == "overall")
        self.assertEqual(overall["bbox_annotations"], 2)

    def test_require_complete_rejects_malformed_annotations(self):
        self.write_volume(self.target_dir, "brain_acc4.h5", annotations="{bad json")
        self.write_volume(self.recon_dir, "brain_acc4.h5")

        message = self.run_main(*self.strict_args())

        self.assertIn("malformed annotations", message)

    def test_require_complete_rejects_malformed_annotation_box(self):
        annotations = json.dumps(
            {"0": [{"x": "bad", "y": 0, "width": 8, "height": 8}]}
        )
        self.write_volume(self.target_dir, "brain_acc4.h5", annotations=annotations)
        self.write_volume(self.recon_dir, "brain_acc4.h5")

        message = self.run_main(*self.strict_args())

        self.assertIn("malformed annotation box", message)

    def test_require_complete_rejects_unassigned_annotation_box(self):
        data = np.ones((2, 16, 16), dtype=np.float32)
        annotations = json.dumps([{"x": 0, "y": 0, "width": 12, "height": 12}])
        self.write_volume(self.target_dir, "brain_acc4.h5", data=data, annotations=annotations)
        self.write_volume(self.recon_dir, "brain_acc4.h5", data=data)

        message = self.run_main(*self.strict_args(slices=2))

        self.assertIn("annotation box has no valid slice index", message)

    def test_require_complete_rejects_target_reconstruction_shape_mismatch(self):
        self.write_volume(
            self.target_dir,
            "brain_acc4.h5",
            data=np.ones((1, 16, 16), dtype=np.float32),
        )
        self.write_volume(
            self.recon_dir,
            "brain_acc4.h5",
            data=np.ones((1, 16, 15), dtype=np.float32),
        )

        message = self.run_main(*self.strict_args())

        self.assertIn("shape mismatch target=(1, 16, 16) recon=(1, 16, 15)", message)

    def test_require_complete_rejects_target_reconstruction_slice_mismatch(self):
        self.write_volume(
            self.target_dir,
            "brain_acc4.h5",
            data=np.ones((2, 16, 16), dtype=np.float32),
        )
        self.write_volume(
            self.recon_dir,
            "brain_acc4.h5",
            data=np.ones((1, 16, 16), dtype=np.float32),
        )

        message = self.run_main(*self.strict_args(slices=2))

        self.assertIn("slice-count mismatch target=2 recon=1", message)

    def test_require_complete_rejects_skipped_full_metric(self):
        data = np.zeros((1, 16, 16), dtype=np.float32)
        annotations = json.dumps(
            {"0": [{"x": 0, "y": 0, "width": 12, "height": 12}]}
        )
        self.write_volume(self.target_dir, "brain_acc4.h5", data=data, annotations=annotations)
        self.write_volume(self.recon_dir, "brain_acc4.h5", data=data)

        message = self.run_main(*self.strict_args())

        self.assertIn("full metric skipped for slice 0", message)

    def test_require_complete_rejects_skipped_bbox_metric(self):
        annotations = json.dumps(
            {"0": [{"x": 0, "y": 0, "width": 4, "height": 4}]}
        )
        self.write_volume(self.target_dir, "brain_acc4.h5", annotations=annotations)
        self.write_volume(self.recon_dir, "brain_acc4.h5")

        message = self.run_main(*self.strict_args())

        self.assertIn("bbox metric skipped for slice 0 box 0", message)

    def test_require_complete_rejects_expected_volume_count_mismatch(self):
        self.write_complete_acc_pair()

        message = self.run_main(*self.strict_args(volumes=3, slices=2, boxes=2))

        self.assertIn("volume coverage mismatch: expected=3 actual=2", message)

    def test_require_complete_rejects_expected_slice_count_mismatch(self):
        self.write_complete_acc_pair()

        message = self.run_main(*self.strict_args(volumes=2, slices=3, boxes=2))

        self.assertIn("slice coverage mismatch: expected=3 actual=2", message)

    def test_require_complete_rejects_expected_box_count_mismatch(self):
        self.write_complete_acc_pair()

        message = self.run_main(*self.strict_args(volumes=2, slices=2, boxes=3))

        self.assertIn("box coverage mismatch: expected=3 actual=2", message)

    def test_require_complete_rejects_missing_acceleration_metric_scope(self):
        annotations = json.dumps(
            {"0": [{"x": 0, "y": 0, "width": 12, "height": 12}]}
        )
        self.write_volume(self.target_dir, "brain_acc4.h5", annotations=annotations)
        self.write_volume(self.recon_dir, "brain_acc4.h5")

        message = self.run_main(*self.strict_args())

        self.assertIn("complete metrics require both acc4 and acc8", message)

    def test_require_complete_accepts_valid_reduced_expected_counts_and_writes_csv_rows(self):
        self.write_complete_acc_pair()

        self.run_main_successfully(*self.strict_args(volumes=2, slices=2, boxes=2))

        summary = json.loads((self.out_dir / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["skipped"], [])
        rows_by_scope = {row["scope"]: row for row in summary["rows"]}
        self.assertEqual(
            list(rows_by_scope),
            ["overall", "acc4", "acc8", "leaderboard_equal_acc", "unknown"],
        )
        self.assertEqual(rows_by_scope["overall"]["volumes"], 2)
        self.assertEqual(rows_by_scope["overall"]["slices"], 2)
        self.assertEqual(rows_by_scope["overall"]["bbox_annotations"], 2)
        self.assertEqual(rows_by_scope["overall"]["ssim_full_count"], 2)
        self.assertEqual(rows_by_scope["overall"]["ssim_bbox_count"], 2)
        self.assertAlmostEqual(rows_by_scope["leaderboard_equal_acc"]["quality_score"], 1.0)
        with (self.out_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
            csv_rows = list(csv.DictReader(handle))
        self.assertEqual([row["scope"] for row in csv_rows], list(rows_by_scope))

    def test_non_strict_annotation_parser_preserves_malformed_and_unassigned_skips(self):
        module = load_evaluate_val_module()

        boxes, count, skipped = module.parse_annotations("{bad json", 1)
        self.assertEqual((dict(boxes), count, skipped), ({}, 0, 0))

        raw = json.dumps([{"x": 0, "y": 0, "width": 12, "height": 12}])
        boxes, count, skipped = module.parse_annotations(raw, 2)
        self.assertEqual((dict(boxes), count, skipped), ({}, 0, 1))

    def test_non_strict_annotation_parser_preserves_empty_and_scalar_compatibility(self):
        module = load_evaluate_val_module()

        for raw in (None, "", "null", "true", "17", '"scalar"', [], [1]):
            with self.subTest(raw=raw):
                boxes, count, skipped = module.parse_annotations(raw, 1)
                self.assertEqual((dict(boxes), count, skipped), ({}, 0, 0))

    def test_non_strict_main_preserves_success_outputs_and_csv_scope_rows(self):
        self.write_complete_acc_pair()

        self.run_main_successfully()

        summary = json.loads((self.out_dir / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["skipped"], [])
        expected_scopes = [
            "overall",
            "acc4",
            "acc8",
            "leaderboard_equal_acc",
            "unknown",
        ]
        self.assertEqual([row["scope"] for row in summary["rows"]], expected_scopes)
        with (self.out_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
            self.assertEqual([row["scope"] for row in csv.DictReader(handle)], expected_scopes)


if __name__ == "__main__":
    unittest.main()
