#!/usr/bin/env python3
"""Build and re-audit a deterministic Linux-safe final tar.gz archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parent
SAFE_SLUG = re.compile(r"^[A-Za-z0-9_-]+$")
SKIP_NAMES = {
    ".DS_Store",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    "Thumbs.db",
    "__MACOSX",
    "__pycache__",
    "result",
    "runtime_output",
}


def fail(message: str) -> None:
    raise SystemExit(f"SUBMISSION_ARCHIVE_REFUSED: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--team-slug", required=True)
parser.add_argument("--output-dir", type=Path, default=ROOT.parent)
args = parser.parse_args()

if SAFE_SLUG.fullmatch(args.team_slug) is None:
    fail("team slug must contain only ASCII letters, digits, underscore, or hyphen")

subprocess.run(
    [sys.executable, str(ROOT / "submission_audit.py"), "--require-submission-ready"],
    cwd=ROOT,
    check=True,
)

base_name = f"2026_FastMRI_{args.team_slug}"
output_dir = args.output_dir.resolve()
output_dir.mkdir(parents=True, exist_ok=True)
archive = output_dir / f"{base_name}.tar.gz"
sidecar = output_dir / f"{base_name}.tar.gz.sha256"
if archive.exists() or sidecar.exists():
    fail("archive or checksum sidecar already exists")

paths = []
for path in sorted(ROOT.rglob("*")):
    relative = path.relative_to(ROOT)
    if any(component in SKIP_NAMES for component in relative.parts):
        continue
    if path.is_symlink():
        fail(f"symlink is forbidden: {relative.as_posix()}")
    paths.append(path)

temporary_fd, temporary_name = tempfile.mkstemp(
    prefix=f".{base_name}.", suffix=".tar.gz.tmp", dir=output_dir
)
os.close(temporary_fd)
temporary = Path(temporary_name)
try:
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as zipped:
            with tarfile.open(mode="w", fileobj=zipped, format=tarfile.PAX_FORMAT) as tar:
                root_info = tarfile.TarInfo(base_name)
                root_info.type = tarfile.DIRTYPE
                root_info.mode = 0o755
                root_info.mtime = 0
                root_info.uid = root_info.gid = 0
                root_info.uname = root_info.gname = "root"
                tar.addfile(root_info)
                for path in paths:
                    relative = path.relative_to(ROOT).as_posix()
                    info = tar.gettarinfo(str(path), arcname=f"{base_name}/{relative}")
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = "root"
                    info.mode = 0o755 if path.is_dir() or path.suffix == ".sh" else 0o644
                    if path.is_file():
                        with path.open("rb") as handle:
                            tar.addfile(info, handle)
                    else:
                        tar.addfile(info)
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary, archive)
finally:
    temporary.unlink(missing_ok=True)

sidecar.write_text(f"{sha256(archive)}  {archive.name}\n", encoding="ascii", newline="\n")
try:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "submission_audit.py"),
            "--require-submission-ready",
            "--archive",
            str(archive),
        ],
        cwd=ROOT,
        check=True,
    )
except Exception:
    archive.unlink(missing_ok=True)
    sidecar.unlink(missing_ok=True)
    raise
print(f"FINAL_SUBMISSION_ARCHIVE={archive}")
print(f"FINAL_SUBMISSION_ARCHIVE_SHA256={sha256(archive)}")
