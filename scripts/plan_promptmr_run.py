#!/usr/bin/env python
"""CLI wrapper for the metadata-only PromptMR+ run planner."""

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.promptmr.planner import main


if __name__ == "__main__":
    raise SystemExit(main())
