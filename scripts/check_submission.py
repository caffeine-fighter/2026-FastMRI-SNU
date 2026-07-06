#!/usr/bin/env python3
import fnmatch
import subprocess
from pathlib import Path

REQUIRED_FILES = [
    "README.md",
    "train.py",
    "reconstruct.py",
    ".gitignore",
    ".env.example",
    "scripts/print_run_context.py",
    "scripts/evaluate_val.py",
    "scripts/plot_loss.py",
    "experiments/experiment_log.csv",
    "docs/vessl_workflow.md",
    "docs/metric_notes.md",
    "docs/experiment_plan.md",
    "docs/decisions.md",
]

FORBIDDEN_TRACKED_PATTERNS = [
    "Data/*",
    "data/*",
    "*.h5",
    "result/*",
    "results/*",
    "runs/*",
    "checkpoints/*",
    "*.pt",
    "*.pth",
    "*.ckpt",
    ".env",
    ".env.local",
]

def git_ls_files():
    try:
        out = subprocess.check_output(["git", "ls-files"], text=True, stderr=subprocess.STDOUT)
        return [line.strip() for line in out.splitlines() if line.strip()]
    except Exception as exc:
        print(f"[WARN] git ls-files unavailable: {exc}")
        return []

def main():
    errors = []

    for path in REQUIRED_FILES:
        if Path(path).exists():
            print(f"[OK] {path}")
        else:
            errors.append(f"missing required file: {path}")

    tracked = git_ls_files()
    for path in tracked:
        normalized = path.replace("\\", "/")
        for pattern in FORBIDDEN_TRACKED_PATTERNS:
            if fnmatch.fnmatch(normalized, pattern):
                errors.append(f"forbidden tracked file: {path}")

    if errors:
        print("\nBlocking issues:")
        for item in errors:
            print(f"  - {item}")
    else:
        print("\nNo blocking submission-safety issues found.")

    raise SystemExit(1 if errors else 0)

if __name__ == "__main__":
    main()
