import importlib.util
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
