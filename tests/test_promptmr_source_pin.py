import ast
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "promptmr_plus"


class PromptMRSourcePinTests(unittest.TestCase):
    def test_git_attributes_preserve_exact_vendored_bytes(self):
        attributes = {
            line.strip()
            for line in (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertIn("vendor/promptmr_plus/** -text", attributes)

    def test_manifest_pins_exact_repository_commit_and_license(self):
        manifest = json.loads(
            (VENDOR / "SOURCE_MANIFEST.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["repository"], "https://github.com/hellopipu/PromptMR-plus")
        self.assertEqual(manifest["commit"], "934eeda6d4d18cd39e406fa1eee9e1f70603cb5e")
        self.assertEqual(manifest["license"]["spdx_expression"], "LicenseRef-RU-NCRL")
        self.assertEqual(manifest["license"]["commercial_use"], "restricted")

    def test_every_vendored_upstream_file_matches_pinned_hash(self):
        manifest = json.loads(
            (VENDOR / "SOURCE_MANIFEST.json").read_text(encoding="utf-8")
        )
        for relative, expected in manifest["upstream_files"].items():
            with self.subTest(relative=relative):
                actual = hashlib.sha256((VENDOR / relative).read_bytes()).hexdigest()
                self.assertEqual(actual, expected)

    def test_runtime_model_uses_vendored_promptmr_not_a_local_reimplementation(self):
        factory = (ROOT / "utils" / "promptmr" / "runtime.py").read_text(encoding="utf-8")
        tree = ast.parse(factory)
        imports = [
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ]
        self.assertIn("models.promptmr_v2", imports)
        self.assertNotIn("utils.model.promptmr", imports)

    def test_training_entrypoint_preserves_varnet_default_and_has_one_family_selector(self):
        tree = ast.parse((ROOT / "train.py").read_text(encoding="utf-8"))
        family_calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
                continue
            if any(
                isinstance(arg, ast.Constant) and arg.value == "--model-family"
                for arg in node.args
            ):
                family_calls.append(node)
        self.assertEqual(len(family_calls), 1)
        keywords = {keyword.arg: keyword.value for keyword in family_calls[0].keywords}
        self.assertEqual(ast.literal_eval(keywords["default"]), "varnet")
        self.assertIsInstance(keywords["choices"], ast.Name)
        self.assertEqual(keywords["choices"].id, "MODEL_FAMILIES")

    def test_training_still_uses_atomic_checkpoint_and_retained_publication(self):
        source = (ROOT / "utils" / "learning" / "train_part.py").read_text(encoding="utf-8")
        artifact = source.index("artifact_name = f\".checkpoint-generation-")
        provenance = source.index("-publication.json")
        retained = source.index("_publish_retained_epoch(staged_retained")
        manifest = source.index("_publish_checkpoint_manifest", retained)
        alias = source.index("_publish_stable_alias", manifest)
        self.assertLess(artifact, provenance)
        self.assertLess(provenance, retained)
        self.assertLess(retained, manifest)
        self.assertLess(manifest, alias)
        self.assertNotIn(".retained-provenance.json", source)
        self.assertIn("build_training_state(", source)


if __name__ == "__main__":
    unittest.main()
