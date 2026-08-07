#!/usr/bin/env python3
"""Fail-closed audit for the organizer-facing final submission package."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import tempfile


HERE = Path(__file__).resolve().parent
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")
LEARNED_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}
RESIDUAL_NAMES = {
    ".DS_Store",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    "Thumbs.db",
    "__MACOSX",
    "__pycache__",
}
ALLOWED_TOP_LEVEL = {
    "README.md",
    "SHA256SUMS",
    "assemble_final_package.py",
    "best_model.pt",
    "build_submission_archive.py",
    "evidence",
    "materialize_r30_evidence.py",
    "package-manifest.json",
    "project",
    "recon_eval.sh",
    "record_official_evaluation.py",
    "reproduce_final.sh",
    "reproduction",
    "requirements.txt",
    "run_official_evaluation_once.sh",
    "seal_package.py",
    "submission_audit.py",
    "verify_package.py",
}
OFFICIAL_ABSOLUTE_PATH_EXCEPTIONS = {
    "assemble_final_package.py",
    "project/recon_eval.py",
    "project/reconstruct.py",
}
CODE_SUFFIXES = {".py", ".sh", ".json", ".yaml", ".yml", ".toml"}
UNIX_ABSOLUTE = re.compile(
    r"(?i)(?<![A-Za-z0-9_])(?:Path\()?['\"]/(?:root|workspace|home|mnt|data|result|output|content)(?:/|['\"])"
)
WINDOWS_ABSOLUTE = re.compile(r"(?i)(?<![A-Za-z0-9_])[A-Z]:[\\/]")


def fail(message: str) -> None:
    raise SystemExit(f"SUBMISSION_AUDIT_FAILED: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_relative(relative: PurePosixPath) -> None:
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        fail(f"unsafe archive path: {relative}")
    for component in relative.parts:
        if component in RESIDUAL_NAMES or SAFE_COMPONENT.fullmatch(component) is None:
            fail(f"unsafe filename component: {relative}")


def audit_tree(root: Path, *, require_submission_ready: bool) -> None:
    if not root.is_dir() or root.is_symlink():
        fail(f"regular package directory required: {root}")
    top_level = {path.name for path in root.iterdir()}
    unexpected = sorted(top_level - ALLOWED_TOP_LEVEL)
    if unexpected:
        fail(f"unrelated top-level files: {unexpected}")

    learned = []
    hardcoded = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        safe_relative(PurePosixPath(relative.as_posix()))
        if path.is_symlink():
            fail(f"symlink is forbidden: {relative.as_posix()}")
        if not path.is_file():
            continue
        if path.suffix.lower() in LEARNED_SUFFIXES:
            learned.append(relative.as_posix())
        if (
            path.suffix.lower() in CODE_SUFFIXES
            and relative.as_posix() not in OFFICIAL_ABSOLUTE_PATH_EXCEPTIONS
            and relative.parts[0] != "evidence"
        ):
            text = path.read_text(encoding="utf-8", errors="replace")
            for line_number, line in enumerate(text.splitlines(), 1):
                if UNIX_ABSOLUTE.search(line) or WINDOWS_ABSOLUTE.search(line):
                    hardcoded.append(f"{relative.as_posix()}:{line_number}")
    if require_submission_ready and learned != ["best_model.pt"]:
        fail(f"exactly best_model.pt must be learned state, observed {learned}")
    if hardcoded:
        fail(f"hardcoded absolute paths in runnable code/config: {hardcoded[:20]}")

    readme_path = root / "README.md"
    requirements_path = root / "requirements.txt"
    if not readme_path.is_file() or not requirements_path.is_file():
        fail("README.md or requirements.txt is missing")
    readme = readme_path.read_text(encoding="utf-8", errors="strict")
    for required_text in (
        "bash recon_eval.sh",
        "bash reproduce_final.sh",
        "Routing table",
        "Directory tree",
        "sha256sum -c SHA256SUMS",
    ):
        if required_text not in readme:
            fail(f"README is missing {required_text!r}")
    affirmative_selection = re.search(
        r"(?i)(?:selected|chose|picked).{0,80}leaderboard|leaderboard.{0,80}(?:selected|chose|picked)",
        readme,
    )
    if affirmative_selection:
        fail("README states leaderboard-based model selection")

    if require_submission_ready:
        required_evidence = (
            "evidence/training_logs/generalist.log",
            "evidence/training_logs/acc4_specialist.log",
            "evidence/training_logs/acc8_specialist.log",
            "evidence/training_logs/naf_s.log",
            "evidence/environment/pip_freeze.txt",
            "evidence/official-evaluation-receipt.json",
            "SHA256SUMS",
            "package-manifest.json",
        )
        for relative in required_evidence:
            path = root / relative
            if not path.is_file() or path.stat().st_size == 0:
                fail(f"required final evidence is missing: {relative}")
        completed = subprocess.run(
            [sys.executable, str(root / "verify_package.py"), "--submission-ready"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            sys.stderr.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            fail("verify_package.py --submission-ready failed")


def audit_archive(archive: Path) -> None:
    if not archive.is_file() or archive.suffixes[-2:] != [".tar", ".gz"]:
        fail(f"regular .tar.gz archive required: {archive}")
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        if not members:
            fail("archive is empty")
        roots = set()
        for member in members:
            if "\\" in member.name:
                fail(f"Windows backslash archive member: {member.name}")
            relative = PurePosixPath(member.name)
            safe_relative(relative)
            roots.add(relative.parts[0])
            if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
                fail(f"unsupported archive member type: {member.name}")
        if len(roots) != 1:
            fail(f"archive must have one top-level directory, observed {sorted(roots)}")
        with tempfile.TemporaryDirectory(prefix="fastmri-archive-audit-") as temporary:
            destination = Path(temporary)
            # Member names and types were checked above; avoid version-specific
            # extraction filters so this also runs under the pinned Python 3.10.
            handle.extractall(destination)
            extracted = destination / next(iter(roots))
            audit_tree(extracted, require_submission_ready=True)


parser = argparse.ArgumentParser()
parser.add_argument("--package-root", type=Path, default=HERE)
parser.add_argument("--archive", type=Path)
parser.add_argument("--require-submission-ready", action="store_true")
args = parser.parse_args()

audit_tree(args.package_root.resolve(), require_submission_ready=args.require_submission_ready)
if args.archive is not None:
    audit_archive(args.archive.resolve())
print("FINAL_SUBMISSION_AUDIT_PASS")
