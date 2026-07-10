import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/phase2_score.py"


class Phase2ScoreCLITests(unittest.TestCase):
    def run_parser(self, log_text: str) -> dict:
        with tempfile.TemporaryDirectory() as temp_name:
            temp_dir = Path(temp_name)
            log_path = temp_dir / "eval.log"
            out_path = temp_dir / "score.json"
            log_path.write_text(log_text, encoding="utf-8")
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--log",
                    str(log_path),
                    "--out-json",
                    str(out_path),
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            return json.loads(out_path.read_text(encoding="utf-8"))

    def test_parses_committed_official_fixture(self) -> None:
        fixture = REPO_ROOT / "tests/fixtures/phase2_score_official.log"
        result = self.run_parser(fixture.read_text(encoding="utf-8"))

        expected = {
            "ssim_full": 0.9178,
            "ssim_bbox": 0.9108,
            "recon_time_s": 383.5,
            "time_ms_per_slice": 173.3,
            "ssim_full_acc4": 0.9408,
            "ssim_full_acc8": 0.8949,
            "ssim_bbox_acc4": 0.9373,
            "ssim_bbox_acc8": 0.8844,
            "recon_time_acc4_s": 198.65,
            "recon_time_acc8_s": 184.85,
            "quality_score": 0.9143,
            "time_score": 0.00095140625,
            "total_score": 0.91525140625,
        }
        for key, value in expected.items():
            self.assertAlmostEqual(result[key], value, msg=key)

    def test_parses_scientific_notation(self) -> None:
        result = self.run_parser(
            "Leaderboard SSIM_full : 9.178e-1\n"
            "Leaderboard SSIM_bbox : 9.108e-1\n"
            "Leaderboard Recon Time : 3.835e2s (1.733e2 ms/slice)\n"
        )

        self.assertAlmostEqual(result["ssim_full"], 0.9178)
        self.assertAlmostEqual(result["ssim_bbox"], 0.9108)
        self.assertAlmostEqual(result["recon_time_s"], 383.5)
        self.assertAlmostEqual(result["time_ms_per_slice"], 173.3)

    def test_accepts_equal_separators_and_spaced_units(self) -> None:
        result = self.run_parser(
            "Leaderboard SSIM_full = 0.9178\n"
            "Leaderboard SSIM_bbox = 0.9108\n"
            "Leaderboard Recon Time = 383.50 s (173.3 ms / slice)\n"
        )

        self.assertAlmostEqual(result["ssim_full"], 0.9178)
        self.assertAlmostEqual(result["ssim_bbox"], 0.9108)
        self.assertAlmostEqual(result["recon_time_s"], 383.5)
        self.assertAlmostEqual(result["time_ms_per_slice"], 173.3)

    def test_rejects_out_of_range_ssim(self) -> None:
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_parser(
                "Leaderboard SSIM_full : 1.2\n"
                "Leaderboard SSIM_bbox : 0.9108\n"
                "Leaderboard Recon Time : 383.50s (173.3 ms/slice)\n"
            )

    def test_rejects_nonpositive_total_reconstruction_time(self) -> None:
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_parser(
                "Leaderboard SSIM_full : 0.9178\n"
                "Leaderboard SSIM_bbox : 0.9108\n"
                "Leaderboard Recon Time : -383.50s (173.3 ms/slice)\n"
            )

    def test_rejects_nonpositive_ms_per_slice(self) -> None:
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_parser(
                "Leaderboard SSIM_full : 0.9178\n"
                "Leaderboard SSIM_bbox : 0.9108\n"
                "Leaderboard Recon Time : 383.50s (-173.3 ms/slice)\n"
            )

    def test_rejects_nonfinite_total_reconstruction_time(self) -> None:
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_parser(
                "Leaderboard SSIM_full : 0.9178\n"
                "Leaderboard SSIM_bbox : 0.9108\n"
                "Leaderboard Recon Time : 1e999s (173.3 ms/slice)\n"
            )

    def test_rejects_nonfinite_ms_per_slice(self) -> None:
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_parser(
                "Leaderboard SSIM_full : 0.9178\n"
                "Leaderboard SSIM_bbox : 0.9108\n"
                "Leaderboard Recon Time : 383.50s (1e999 ms/slice)\n"
            )

    def test_rejects_out_of_range_subgroup_ssim(self) -> None:
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_parser(
                "Leaderboard SSIM_full : 0.9178\n"
                "Leaderboard SSIM_bbox : 0.9108\n"
                "Leaderboard Recon Time : 383.50s (173.3 ms/slice)\n"
                "SSIM_full (acc4): 1.1\n"
            )

    def test_rejects_malformed_numeric_suffix(self) -> None:
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_parser(
                "Leaderboard SSIM_full : 0.9178e+\n"
                "Leaderboard SSIM_bbox : 0.9108\n"
                "Leaderboard Recon Time : 383.50s (173.3 ms/slice)\n"
            )


if __name__ == "__main__":
    unittest.main()
