#!/usr/bin/env python
"""Fail-closed validation for the minimal pinned PromptMR+ vendor snapshot."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "promptmr_plus"
MANIFEST_PATH = VENDOR / "SOURCE_MANIFEST.json"
LOCAL_PACKAGES = {"data", "models", "mri_utils"}


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def local_module_exists(module: str) -> bool:
    parts = module.split(".")
    path = VENDOR.joinpath(*parts)
    return path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()


def validate(verify_upstream: bool) -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    commit = manifest["commit"]
    upstream_files = manifest["upstream_files"]
    local_files = manifest.get("local_files", {})
    expected_files = {"SOURCE_MANIFEST.json", *upstream_files, *local_files}
    actual_files = {
        path.relative_to(VENDOR).as_posix()
        for path in VENDOR.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        raise RuntimeError(
            f"Vendor allowlist mismatch: missing={sorted(expected_files - actual_files)}, "
            f"unexpected={sorted(actual_files - expected_files)}"
        )

    for relative, metadata in sorted(local_files.items()):
        if sha256((VENDOR / relative).read_bytes()) != metadata["sha256"]:
            raise RuntimeError(f"Local vendor allowlist hash mismatch for {relative}")

    for relative, expected_digest in sorted(upstream_files.items()):
        payload = (VENDOR / relative).read_bytes()
        actual_digest = sha256(payload)
        if actual_digest != expected_digest:
            raise RuntimeError(
                f"Canonical local hash mismatch for {relative}: {actual_digest}"
            )
        if verify_upstream:
            url = (
                "https://raw.githubusercontent.com/hellopipu/PromptMR-plus/"
                f"{commit}/{relative}"
            )
            request = urllib.request.Request(
                url, headers={"User-Agent": "snuaichallenge-source-validator"}
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                canonical = response.read()
            if canonical != payload or sha256(canonical) != expected_digest:
                raise RuntimeError(f"Pinned upstream Git blob mismatch for {relative}")

    license_payload = (VENDOR / manifest["license"]["file"]).read_bytes()
    if sha256(license_payload) != manifest["license"]["sha256"]:
        raise RuntimeError("RU-NCRL license digest mismatch")
    license_text = license_payload.decode("utf-8").lower()
    for required in ("noncommercial", "redistribution", "disclaimer"):
        if required not in license_text:
            raise RuntimeError(f"RU-NCRL notice missing required term: {required}")

    requirements = {
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if "einops" not in requirements:
        raise RuntimeError("Missing PromptMR+ runtime dependency: einops")

    unresolved = []
    for relative in sorted(upstream_files):
        if not relative.endswith(".py"):
            continue
        tree = ast.parse((VENDOR / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else ([node.module] if node.module else [])
            )
            for name in names:
                root = name.split(".", 1)[0]
                if root in LOCAL_PACKAGES and not local_module_exists(name):
                    unresolved.append((relative, name))
    if unresolved:
        raise RuntimeError(f"Unresolved vendored imports: {unresolved}")

    print(
        json.dumps(
            {
                "status": "PASS",
                "commit": commit,
                "canonical_hashes": len(upstream_files),
                "allowlisted_files": len(actual_files),
                "license": manifest["license"]["spdx_expression"],
                "upstream_verified": verify_upstream,
            },
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-upstream", action="store_true")
    return validate(parser.parse_args().verify_upstream)


if __name__ == "__main__":
    raise SystemExit(main())
