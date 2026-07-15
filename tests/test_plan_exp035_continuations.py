import hashlib
import os
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
        root.mkdir(parents=True, exist_ok=True)
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
            sha_flag = arm["command"].index("--resume-checkpoint-sha256")
            self.assertEqual(arm["command"][sha_flag + 1], digest)
            gpu_name_flag = arm["command"].index("--require-cuda-device-name")
            self.assertEqual(
                arm["command"][gpu_name_flag + 1], "NVIDIA GeForce GTX 1080"
            )
            self.assertIn("--retain-val-epochs", arm["command"])
            self.assertIn("35", arm["command"])

    def test_builds_matched_recovery_names_without_reusing_collided_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            control_source, digest = self._source(root / "control")
            candidate_source, candidate_digest = self._source(root / "candidate")
            self.assertEqual(candidate_digest, digest)
            plan = build_plan(
                control_source,
                candidate_checkpoint=candidate_source,
                result_root=root / "result",
                expected_generation="test-generation",
                expected_sha256=digest,
                name_suffix="_R1",
            )
            control_inode = (control_source.stat().st_dev, control_source.stat().st_ino)
            candidate_inode = (
                candidate_source.stat().st_dev,
                candidate_source.stat().st_ino,
            )

        self.assertEqual(
            [arm["net_name"] for arm in plan["arms"]],
            [f"{CONTROL_NAME}_R1", f"{CANDIDATE_NAME}_R1"],
        )
        checkpoint_paths = []
        for arm in plan["arms"]:
            flag = arm["command"].index("--resume-checkpoint")
            checkpoint_paths.append(arm["command"][flag + 1])
        self.assertEqual(
            checkpoint_paths,
            [str(control_source.resolve()), str(candidate_source.resolve())],
        )
        self.assertNotEqual(control_inode, candidate_inode)

    def test_recovery_plan_requires_distinct_candidate_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, digest = self._source(root)
            with self.assertRaisesRegex(ValueError, "distinct candidate checkpoint"):
                build_plan(
                    source,
                    result_root=root / "result",
                    expected_generation="test-generation",
                    expected_sha256=digest,
                    name_suffix="_R1",
                )

    def test_recovery_plan_rejects_hardlinked_private_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            control_source, digest = self._source(root / "control")
            candidate_dir = root / "candidate"
            candidate_dir.mkdir()
            candidate_source = candidate_dir / control_source.name
            os.link(control_source, candidate_source)

            with self.assertRaisesRegex(ValueError, "not hardlinks"):
                build_plan(
                    control_source,
                    candidate_checkpoint=candidate_source,
                    result_root=root / "result",
                    expected_generation="test-generation",
                    expected_sha256=digest,
                    name_suffix="_R1",
                )

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
