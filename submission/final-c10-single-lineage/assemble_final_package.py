#!/usr/bin/env python3
"""Atomically assemble and seal the only R30 final package on VESSL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid


ROOT = Path(__file__).resolve().parent
LEGACY_SERVER_ROOT = "/" + "root"
LEGACY_RESULT_ROOT = LEGACY_SERVER_ROOT + "/result"
LEGACY_DATA_ROOT = LEGACY_SERVER_ROOT + "/Data"
LEGACY_WORKSPACE_ROOT = "/" + "workspace"
LEARNED_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}
SKIP_NAMES = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", "result", "checkpoints"}


def fail(message: str) -> None:
    raise RuntimeError(f"FINAL_PACKAGE_ASSEMBLY_REFUSED: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        fail(f"invalid JSON {path}: {error}")
    if not isinstance(value, dict):
        fail(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(
            (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def stable_copy(source: Path, destination: Path) -> str:
    if not source.is_file() or source.is_symlink():
        fail(f"regular source file required: {source}")
    before = source.stat()
    digest = sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        fail(f"destination already exists: {destination}")
    shutil.copy2(source, destination)
    after = source.stat()
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or sha256(source) != digest
        or sha256(destination) != digest
    ):
        fail(f"source changed while copying: {source}")
    return digest


def replace_copy(source: Path, destination: Path) -> str:
    destination.unlink(missing_ok=True)
    return stable_copy(source, destination)


def copy_source_tree(source: Path, destination: Path) -> dict[str, str]:
    if not source.is_dir() or source.is_symlink():
        fail(f"regular source directory required: {source}")
    copied: dict[str, str] = {}
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part in SKIP_NAMES for part in relative.parts):
            continue
        if path.is_symlink():
            fail(f"source symlink is forbidden: {path}")
        if not path.is_file():
            continue
        if path.suffix.lower() in LEARNED_SUFFIXES:
            fail(f"learned state is forbidden in source tree: {path}")
        copied[relative.as_posix()] = stable_copy(path, destination / relative)
    return copied


def copy_skeleton(destination: Path) -> None:
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if any(part in SKIP_NAMES for part in relative.parts):
            continue
        if relative.parts[0] in {"project", "evidence", "reproduction"}:
            continue
        if relative.as_posix() in {"best_model.pt", "package-manifest.json"}:
            continue
        if path.is_symlink():
            fail(f"skeleton symlink is forbidden: {path}")
        if path.is_file():
            stable_copy(path, destination / relative)


def validate_controller(value: dict, receipt_path: Path) -> None:
    if (
        value.get("schema") != "vessl-g10-architecture-dispatcher-receipt-v1"
        or value.get("state") != "PASS"
        or value.get("winner") != "R2_C10"
        or value.get("final_admission_mode") != "PRIMARY_REQUESTED"
        or value.get("acc4_specialist_status") != "TRAINED_ON_VESSL_AND_ROUTED"
        or value.get("acc8_specialist_status") != "TRAINED_ON_VESSL_AND_ROUTED"
        or value.get("post_refiner_status") != "TRAINED_ON_VESSL_AND_PACKAGED"
        or value.get("fallback_checkpoint") is not None
        or value.get("fallback_checkpoint_sha256") is not None
        or value.get("leaderboard_data_read") is not False
        or value.get("external_learned_state_imported") is not False
        or value.get("all_final_learned_state_vessl_only") is not True
    ):
        fail(f"controller is not the exact single-final R30 run: {receipt_path}")


def controller_file(controller: dict, run_key: str, filename: str) -> Path:
    value = controller.get(run_key)
    if not isinstance(value, str) or not value:
        fail(f"controller is missing {run_key}")
    return Path(value) / filename


def validate_text_evidence(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink() or path.stat().st_size < 256:
        fail(f"non-empty regular {label} is required: {path}")
    sample = path.read_bytes()[:64 * 1024]
    if b"\0" in sample:
        fail(f"{label} is not a text file: {path}")


def portable_text_copy(
    source: Path,
    destination: Path,
    replacements: tuple[tuple[str, str, int], ...] = (),
) -> tuple[str, str]:
    if not source.is_file() or source.is_symlink():
        fail(f"regular portable source file required: {source}")
    original_sha = sha256(source)
    text = source.read_text(encoding="utf-8").replace("\r\n", "\n")
    for old, new, expected_count in replacements:
        observed_count = text.count(old)
        if observed_count != expected_count:
            fail(
                f"portable path rewrite count changed for {source.name}: "
                f"{old!r} observed={observed_count} expected={expected_count}"
            )
        text = text.replace(old, new)
    if re.search(
        r"(?i)['\"]/(?:root|workspace|home|mnt|data|result|output|content)(?:/|['\"])",
        text,
    ):
        fail(f"portable source retains an absolute workspace/data path: {source}")
    if re.search(r"(?i)(?<![A-Za-z0-9_])[A-Z]:[\\/]", text):
        fail(f"portable source retains a Windows absolute path: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        fail(f"portable destination already exists: {destination}")
    destination.write_text(text, encoding="utf-8", newline="\n")
    return original_sha, sha256(destination)


def portable_manifest_value(value: object) -> object:
    if isinstance(value, dict):
        return {key: portable_manifest_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_manifest_value(item) for item in value]
    if isinstance(value, str):
        mappings = (
            (LEGACY_DATA_ROOT, "DATA_ROOT"),
            (LEGACY_RESULT_ROOT, "OUT_DIR/result"),
            (LEGACY_SERVER_ROOT + "/", "SERVER_ROOT/"),
            (LEGACY_WORKSPACE_ROOT + "/", "WORKSPACE_ROOT/"),
        )
        for prefix, replacement in mappings:
            if value.startswith(prefix):
                return replacement + value[len(prefix):]
    return value


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--controller-receipt", type=Path, required=True)
parser.add_argument("--r30-deployment-receipt", type=Path, required=True)
parser.add_argument("--project-source", type=Path, required=True)
parser.add_argument("--handoff-root", type=Path, required=True)
parser.add_argument("--stage", type=Path, required=True)
parser.add_argument("--specialist-stage", type=Path, required=True)
parser.add_argument("--provenance", type=Path, required=True)
parser.add_argument("--generalist-log", type=Path, required=True)
parser.add_argument("--acc4-log", type=Path, required=True)
parser.add_argument("--acc8-log", type=Path, required=True)
parser.add_argument("--naf-s-log", type=Path, required=True)
parser.add_argument("--pip-freeze", type=Path, required=True)
args = parser.parse_args()

for evidence_path, evidence_label in (
    (args.generalist_log, "generalist training log"),
    (args.acc4_log, "ACC4 specialist training log"),
    (args.acc8_log, "ACC8 specialist training log"),
    (args.naf_s_log, "NAF_S training log"),
    (args.pip_freeze, "VESSL pip freeze"),
):
    validate_text_evidence(evidence_path, evidence_label)
freeze_text = args.pip_freeze.read_text(encoding="utf-8", errors="replace").lower()
for package_name in ("torch", "numpy", "h5py"):
    if not any(line.startswith(f"{package_name}==") for line in freeze_text.splitlines()):
        fail(f"VESSL pip freeze is missing {package_name}==...")
normalize_name = lambda value: re.sub(r"[-_.]+", "-", value.strip().lower())
required_versions = {}
for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("--"):
        continue
    if "==" not in stripped:
        fail(f"requirements entry is not exactly pinned: {stripped}")
    name, version = stripped.split("==", 1)
    required_versions[normalize_name(name)] = version.lower()
frozen_versions = {}
for line in freeze_text.splitlines():
    if "==" in line:
        name, version = line.split("==", 1)
        frozen_versions[normalize_name(name)] = version.strip()
for name, version in required_versions.items():
    if frozen_versions.get(name) != version:
        fail(
            f"requirements/pip-freeze mismatch for {name}: "
            f"required={version!r} frozen={frozen_versions.get(name)!r}"
        )

output = args.output.resolve()
if output.exists():
    fail(f"single final output already exists: {output}")
output.parent.mkdir(parents=True, exist_ok=True)
controller = load_json(args.controller_receipt)
validate_controller(controller, args.controller_receipt)

checkpoint_raw = controller.get("final_checkpoint")
checkpoint_sha = controller.get("final_checkpoint_sha256")
if not isinstance(checkpoint_raw, str) or not isinstance(checkpoint_sha, str):
    fail("controller final checkpoint binding is absent")
checkpoint = Path(checkpoint_raw)
if not checkpoint.is_file() or sha256(checkpoint) != checkpoint_sha:
    fail("controller final checkpoint hash mismatch")

snapshot = controller.get("inference_source_snapshot")
if not isinstance(snapshot, dict):
    fail("controller inference source snapshot is absent")
snapshot_directory = Path(str(snapshot.get("directory", "")))
snapshot_manifest = Path(str(snapshot.get("manifest", "")))
snapshot_files = snapshot.get("files")
if (
    not snapshot_directory.is_dir()
    or not snapshot_manifest.is_file()
    or sha256(snapshot_manifest) != snapshot.get("manifest_sha256")
    or not isinstance(snapshot_files, list)
    or not snapshot_files
):
    fail("controller inference source snapshot binding is invalid")
for item in snapshot_files:
    if not isinstance(item, dict) or not isinstance(item.get("path"), str):
        fail("invalid source snapshot file record")
    source = snapshot_directory / item["path"]
    if not source.is_file() or sha256(source) != item.get("sha256"):
        fail(f"source snapshot hash mismatch: {source}")

staging = Path(tempfile.mkdtemp(prefix=".final-c10-r30-assembly-", dir=output.parent))
try:
    copy_skeleton(staging)

    project = staging / "project"
    project_hashes: dict[str, str] = {}
    for name in ("recon_eval.py", "reconstruct.py"):
        project_hashes[name] = stable_copy(args.project_source / name, project / name)
    for name in ("utils", "third_party"):
        for relative, digest in copy_source_tree(args.project_source / name, project / name).items():
            project_hashes[f"{name}/{relative}"] = digest

    snapshot_destinations: dict[str, str] = {}
    for item in snapshot_files:
        relative = Path(item["path"])
        target_relative = Path("utils/learning/test_part.py") if relative == Path("test_part.py") else relative
        destination = project / target_relative
        digest = replace_copy(snapshot_directory / relative, destination)
        if digest != item["sha256"]:
            fail(f"snapshot overlay hash mismatch: {relative}")
        snapshot_destinations[relative.as_posix()] = target_relative.as_posix()

    reproduction = staging / "reproduction"
    raw = staging / "evidence/raw"
    raw_contract_sources = {
        "FINAL_C10_SINGLE_LINEAGE_R30_NEIGHBOR_ZF.json": (
            args.handoff_root / "FINAL_C10_SINGLE_LINEAGE_R30_NEIGHBOR_ZF.json"
        ),
        "final-tactics-c10-r30-neighbor-zf.json": (
            args.handoff_root / "final-tactics-c10-r30-neighbor-zf.json"
        ),
        "FINAL_C10_SINGLE_LINEAGE_R30_INFERENCE.json": (
            args.stage / "FINAL_C10_SINGLE_LINEAGE_R30_INFERENCE.json"
        ),
        "R30_CPU_PREFLIGHT.json": args.stage / "R30_CPU_PREFLIGHT.json",
        "inference-source-snapshot-manifest.json": snapshot_manifest,
    }
    raw_training_sources = {
        "controller.py": args.stage / "controller.py",
        "generalist/train.py": args.stage / "train.py",
        "generalist/promptmr_production.py": args.stage / "promptmr_production.py",
        "specialist/train.py": args.specialist_stage / "train.py",
        "specialist/promptmr_production.py": (
            args.specialist_stage / "promptmr_production.py"
        ),
        "vessl_train_post_refiner.py": args.stage / "vessl_train_post_refiner.py",
        "vessl_build_routed_promptmr_checkpoint.py": (
            args.stage / "vessl_build_routed_promptmr_checkpoint.py"
        ),
        "promptmr_post_refiner.py": args.stage / "promptmr_post_refiner.py",
        "promptmr_router.py": args.stage / "promptmr_router.py",
        "promptmr_mask_router.py": args.stage / "promptmr_mask_router.py",
        "promptmr_legal_mask.py": args.stage / "promptmr_legal_mask.py",
        "test_part.py": args.stage / "test_part.py",
        "preflight_r30.py": args.stage / "preflight_r30.py",
    }
    raw_contract_hashes = {
        name: stable_copy(source, raw / "contracts" / name)
        for name, source in raw_contract_sources.items()
    }
    raw_training_source_hashes = {
        name: stable_copy(source, raw / "training_source" / name)
        for name, source in raw_training_sources.items()
    }
    original_provenance_sha = stable_copy(
        args.provenance,
        raw / "organizer-data-provenance-original.json",
    )

    portable_sources = {
        "generalist/train.py": (args.stage / "train.py", ()),
        "generalist/promptmr_production.py": (
            args.stage / "promptmr_production.py",
            (),
        ),
        "specialist/train.py": (args.specialist_stage / "train.py", ()),
        "specialist/promptmr_production.py": (
            args.specialist_stage / "promptmr_production.py",
            (
                (
                    f'result_root = Path("{LEGACY_RESULT_ROOT}").resolve()',
                    'result_root = Path(os.environ["FASTMRI_RESULT_ROOT"]).resolve()',
                    1,
                ),
                (
                    f"VESSL model-only parent must be under {LEGACY_RESULT_ROOT}",
                    "VESSL model-only parent must be under FASTMRI_RESULT_ROOT",
                    1,
                ),
            ),
        ),
        "vessl_train_post_refiner.py": (
            args.stage / "vessl_train_post_refiner.py",
            (
                (
                    f'VESSL_RESULT_ROOT = Path("{LEGACY_RESULT_ROOT}")',
                    'VESSL_RESULT_ROOT = Path(os.environ["FASTMRI_RESULT_ROOT"]).resolve()',
                    1,
                ),
                (
                    f'and value.get("root") == "{LEGACY_DATA_ROOT}/train"',
                    'and Path(str(value.get("root", ""))).resolve()\n'
                    '        == Path(os.environ["FASTMRI_TRAIN_ROOT"]).resolve()',
                    1,
                ),
                (
                    f"base checkpoint is outside {LEGACY_RESULT_ROOT}",
                    "base checkpoint is outside FASTMRI_RESULT_ROOT",
                    1,
                ),
                (
                    f"routed branch is outside {LEGACY_RESULT_ROOT}",
                    "routed branch is outside FASTMRI_RESULT_ROOT",
                    1,
                ),
            ),
        ),
        "vessl_build_routed_promptmr_checkpoint.py": (
            args.stage / "vessl_build_routed_promptmr_checkpoint.py",
            (
                (
                    f'VESSL_RESULT_ROOT = Path("{LEGACY_RESULT_ROOT}")',
                    'VESSL_RESULT_ROOT = Path(os.environ["FASTMRI_RESULT_ROOT"]).resolve()',
                    1,
                ),
                (
                    f"post-refiner source is outside {LEGACY_RESULT_ROOT}",
                    "post-refiner source is outside FASTMRI_RESULT_ROOT",
                    1,
                ),
            ),
        ),
        "promptmr_post_refiner.py": (args.stage / "promptmr_post_refiner.py", ()),
        "promptmr_router.py": (args.stage / "promptmr_router.py", ()),
        "promptmr_mask_router.py": (args.stage / "promptmr_mask_router.py", ()),
        "promptmr_legal_mask.py": (args.stage / "promptmr_legal_mask.py", ()),
        "test_part.py": (args.stage / "test_part.py", ()),
    }
    portability_files = {}
    for relative, (source, replacements) in portable_sources.items():
        original_sha, portable_sha = portable_text_copy(
            source,
            reproduction / relative,
            replacements,
        )
        portability_files[relative] = {
            "original_sha256": original_sha,
            "portable_sha256": portable_sha,
            "path_only_rewrite_count": sum(item[2] for item in replacements),
        }

    provenance_value = portable_manifest_value(load_json(args.provenance))
    atomic_json(reproduction / "organizer-data-provenance.json", provenance_value)
    atomic_json(
        reproduction / "portability-receipt.json",
        {
            "schema": "fastmri-r30-path-portability-receipt-v1",
            "state": "PASS",
            "numeric_recipe_changed": False,
            "model_architecture_changed": False,
            "data_identity_changed": False,
            "original_provenance_sha256": original_provenance_sha,
            "portable_provenance_sha256": sha256(
                reproduction / "organizer-data-provenance.json"
            ),
            "files": portability_files,
        },
    )
    source_files = sorted(
        path for path in reproduction.rglob("*") if path.is_file() and path.name != "source-sha256sums.txt"
    )
    (reproduction / "source-sha256sums.txt").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(reproduction).as_posix()}\n" for path in source_files),
        encoding="utf-8",
        newline="\n",
    )

    raw_sources = {
        "controller-final-receipt.json": args.controller_receipt,
        "acc4-terminal.json": controller_file(controller, "acc4_specialist_run", "terminal.json"),
        "acc8-terminal.json": controller_file(controller, "acc8_specialist_run", "terminal.json"),
    }
    raw_hashes = {name: stable_copy(source, raw / name) for name, source in raw_sources.items()}
    reproduction_hashes = {
        path.relative_to(reproduction).as_posix(): sha256(path)
        for path in source_files
    }
    generalist_receipt = controller_file(controller, "selected_run", "controller-terminal-checkpoint.json")
    naf_receipt = controller_file(controller, "post_refiner_run", "receipt.json")

    training_log_sources = {
        "generalist.log": args.generalist_log,
        "acc4_specialist.log": args.acc4_log,
        "acc8_specialist.log": args.acc8_log,
        "naf_s.log": args.naf_s_log,
    }
    training_log_hashes = {
        name: stable_copy(source, staging / "evidence/training_logs" / name)
        for name, source in training_log_sources.items()
    }
    environment_hash = stable_copy(
        args.pip_freeze,
        staging / "evidence/environment/pip_freeze.txt",
    )

    stable_copy(checkpoint, staging / "best_model.pt")
    atomic_json(
        staging / "evidence/assembly-receipt.json",
        {
            "schema": "vessl-r30-single-final-package-assembly-v1",
            "state": "PASS",
            "candidate_count": 1,
            "fallback_registered": False,
            "controller_receipt_sha256": sha256(args.controller_receipt),
            "r30_deployment_receipt_sha256": sha256(args.r30_deployment_receipt),
            "final_checkpoint_sha256": checkpoint_sha,
            "inference_source_snapshot_manifest_sha256": sha256(snapshot_manifest),
            "snapshot_destinations": snapshot_destinations,
            "project_source_hashes": project_hashes,
            "reproduction_source_hashes": reproduction_hashes,
            "raw_contract_hashes": raw_contract_hashes,
            "raw_training_source_hashes": raw_training_source_hashes,
            "original_provenance_sha256": original_provenance_sha,
            "raw_receipt_hashes": raw_hashes,
            "training_log_hashes": training_log_hashes,
            "pip_freeze_sha256": environment_hash,
            "created_unix": time.time(),
        },
    )

    subprocess.run(
        [
            sys.executable,
            str(staging / "materialize_r30_evidence.py"),
            "--controller-receipt", str(raw / "controller-final-receipt.json"),
            "--r30-deployment-receipt", str(args.r30_deployment_receipt),
            "--generalist-receipt", str(generalist_receipt),
            "--acc4-terminal", str(raw / "acc4-terminal.json"),
            "--acc8-terminal", str(raw / "acc8-terminal.json"),
            "--naf-receipt", str(naf_receipt),
        ],
        cwd=staging,
        check=True,
    )
    subprocess.run([sys.executable, str(staging / "seal_package.py")], cwd=staging, check=True)
    if output.exists():
        fail(f"single final output appeared during assembly: {output}")
    os.rename(staging, output)
    print(
        json.dumps(
            {
                "state": "FINAL_C10_R30_SINGLE_PACKAGE_ASSEMBLED",
                "output": str(output),
                "candidate_count": 1,
                "best_model_sha256": checkpoint_sha,
                "package_manifest_sha256": sha256(output / "package-manifest.json"),
            },
            sort_keys=True,
        )
    )
except Exception:
    if staging.exists():
        shutil.rmtree(staging)
    raise
