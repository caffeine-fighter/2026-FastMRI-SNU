import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.plan_exp035_continuations import (
    CANDIDATE_NAME,
    CONTROL_NAME,
    build_plan,
)


class Exp035ContinuationPlanTests(unittest.TestCase):
    def _source(self, root, generation="test-generation"):
        source = root / f".checkpoint-generation-{generation}-model.pt"
        source.write_bytes(b"immutable-exp035-test-checkpoint")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        return source, digest

    def test_builds_two_dry_run_matched_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, digest = self._source(root)
            plan = build_plan(
                source,
                result_root=root / "result",
                expected_generation="test-generation",
                expected_sha256=digest,
            )

        self.assertEqual(plan["mode"], "dry_run_only")
        self.assertFalse(plan["gpu_started"])
        self.assertEqual([arm["net_name"] for arm in plan["arms"]], [CONTROL_NAME, CANDIDATE_NAME])
        self.assertEqual([arm["learning_rate"] for arm in plan["arms"]], [0.001, 0.0003])
        for arm in plan["arms"]:
            self.assertIn("--resume-checkpoint", arm["command"])
            self.assertIn("--retain-val-epochs", arm["command"])
            self.assertIn("35", arm["command"])

    def test_rejects_wrong_checkpoint_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _ = self._source(root)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                build_plan(
                    source,
                    result_root=root / "result",
                    expected_generation="test-generation",
                    expected_sha256="0" * 64,
                )

    def test_rejects_existing_arm_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, digest = self._source(root)
            result_root = root / "result"
            (result_root / CONTROL_NAME).mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "output already exists"):
                build_plan(
                    source,
                    result_root=result_root,
                    expected_generation="test-generation",
                    expected_sha256=digest,
                )

    def test_rejects_symlink_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, digest = self._source(root)
            link = root / ".checkpoint-generation-test-generation-link-model.pt"
            try:
                link.symlink_to(source)
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation is unavailable")
            with self.assertRaisesRegex(ValueError, "not a symlink"):
                build_plan(
                    link,
                    result_root=root / "result",
                    expected_generation="test-generation",
                    expected_sha256=digest,
                )


if __name__ == "__main__":
    unittest.main()
