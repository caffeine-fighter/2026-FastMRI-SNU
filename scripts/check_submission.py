#!/usr/bin/env python3
import subprocess
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "train.py",
    "reconstruct.py",
    ".gitignore",
    ".env.example",
    "scripts/print_run_context.py",
    "scripts/evaluate_val.py",
    "scripts/plot_loss.py",
    "scripts/check_submission.py",
    "experiments/experiment_log.csv",
    "docs/vessl_workflow.md",
    "docs/metric_notes.md",
    "docs/experiment_plan.md",
    "docs/decisions.md",
]

ROOT_FORBIDDEN_DIRS = {
    "Data",
    "data",
    "result",
    "results",
    "runs",
    "checkpoints",
    "wandb",
    "mlruns",
}

DANGEROUS_SUFFIXES = {
    ".h5",
    ".pt",
    ".pth",
    ".ckpt",
    ".pem",
    ".key",
}

ALLOWLIST = {
    ".env.example",
}


def run_git(args):
    safe_root = str(REPO_ROOT).replace("\\", "/")
    cmd = ["git", "-c", f"safe.directory={safe_root}", "-C", str(REPO_ROOT), *args]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception as exc:
        output = getattr(exc, "output", "")
        message = f"git {' '.join(args)} failed"
        if output:
            message += f":\n{output}"
        else:
            message += f": {exc}"
        raise RuntimeError(message) from exc


def normalize(path):
    return path.replace("\\", "/")


def is_forbidden_tracked(path):
    p = normalize(path)
    posix = PurePosixPath(p)
    name = posix.name
    parts = posix.parts

    if p in ALLOWLIST or name in ALLOWLIST:
        return None

    if parts and parts[0] in ROOT_FORBIDDEN_DIRS:
        return f"tracked file under forbidden root directory: {parts[0]}/"

    if name == ".env" or name.startswith(".env.") or name.endswith(".env") or name == "secrets.env":
        return "tracked environment or secret file"

    if name.startswith("id_rsa") or name.startswith("id_ed25519"):
        return "tracked private key candidate"

    for suffix in DANGEROUS_SUFFIXES:
        if name.endswith(suffix):
            return f"tracked forbidden artifact suffix: {suffix}"

    return None


def main():
    errors = []

    for file_path in REQUIRED_FILES:
        if Path(file_path).exists():
            print(f"[OK] {file_path}")
        else:
            errors.append(f"missing required file: {file_path}")

    try:
        tracked = run_git(["ls-files"])
    except RuntimeError as exc:
        errors.append(str(exc))
        tracked = []

    for path in tracked:
        reason = is_forbidden_tracked(path)
        if reason:
            errors.append(f"{reason}: {path}")

    try:
        ignored_tracked = run_git(["ls-files", "-ci", "--exclude-standard"])
    except RuntimeError as exc:
        errors.append(str(exc))
        ignored_tracked = []

    for path in ignored_tracked:
        errors.append(
            "tracked file is ignored by .gitignore; fix .gitignore or untrack it: "
            + path
        )

    if errors:
        print("\nBlocking issues:")
        for item in errors:
            print(f"  - {item}")
    else:
        print("\nNo blocking submission-safety issues found.")

    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
