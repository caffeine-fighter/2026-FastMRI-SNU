"""Pure-Python contracts for the pinned PromptMR+ training integration."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PureWindowsPath


MODEL_FAMILIES = ("varnet", "promptmr_plus")
_WINDOWS_INVALID_NAME_CHARS = frozenset('<>:"/\\|?*')
_WINDOWS_RESERVED_STEMS = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
    | {f"COM{index}" for index in "¹²³"}
    | {f"LPT{index}" for index in "¹²³"}
)
_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "promptmr_plus_training.json"
)
PROMPTMR_PLUS_RECIPE = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
PROMPTMR_PLUS_RECIPE_SHA256 = hashlib.sha256(
    json.dumps(
        PROMPTMR_PLUS_RECIPE, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
).hexdigest()
_ACCELERATION_TOKEN = re.compile(r"acc(4|8)")


def run_name_component(value) -> Path:
    """Return one non-special relative component for run output names."""
    path = Path(value)
    windows_path = PureWindowsPath(value)
    windows_name = windows_path.name
    windows_stem = windows_name.split(".", 1)[0].rstrip(" ").upper()
    if (
        path.is_absolute()
        or len(path.parts) != 1
        or path.name in {".", ".."}
        or windows_path.drive
        or windows_path.root
        or len(windows_path.parts) != 1
        or windows_path.name in {".", ".."}
        or windows_stem in _WINDOWS_RESERVED_STEMS
        or any(
            character in _WINDOWS_INVALID_NAME_CHARS or ord(character) < 32
            for character in windows_name
        )
        or windows_name.endswith((".", " "))
    ):
        raise ValueError("must be one non-special relative path component")
    return path


def parse_acceleration_filename(filename: str) -> int:
    """Return the one exact underscore-delimited acc4/acc8 token."""
    name = Path(filename).name
    matches = [
        token
        for token in Path(name).stem.lower().split("_")
        if _ACCELERATION_TOKEN.fullmatch(token)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Unknown acceleration in {name}: expected exactly one acc4/acc8 token"
        )
    return int(matches[0][3:])


def adjacent_slice_indices(
    center: int, num_slices: int, num_adj_slices: int = 5
) -> tuple[int, ...]:
    """Match upstream fastMRI boundary replication for odd adjacent windows."""
    if isinstance(num_adj_slices, bool) or num_adj_slices <= 0 or num_adj_slices % 2 != 1:
        raise ValueError("Number of adjacent slices must be a positive odd integer")
    if isinstance(num_slices, bool) or num_slices <= 0:
        raise ValueError("Volume slice count must be positive")
    if isinstance(center, bool) or not 0 <= center < num_slices:
        raise ValueError("Center slice is outside the volume")
    radius = num_adj_slices // 2
    return tuple(
        min(max(index, 0), num_slices - 1)
        for index in range(center - radius, center + radius + 1)
    )


def validate_model_family_args(args) -> str:
    family = getattr(args, "model_family", "varnet")
    if family not in MODEL_FAMILIES:
        raise ValueError(f"Unsupported model family: {family!r}")
    if family == "promptmr_plus" and getattr(args, "score_aligned_loss", False):
        raise ValueError(
            "PromptMR+ does not support the legacy score-aligned loss; use the exact SSIM recipe"
        )
    return family


def checkpoint_model_contract(model_family: str) -> dict[str, str]:
    if model_family == "varnet":
        return {"model_family": "varnet"}
    if model_family != "promptmr_plus":
        raise ValueError(f"Unsupported model family: {model_family!r}")
    return {
        "model_family": "promptmr_plus",
        "recipe_id": PROMPTMR_PLUS_RECIPE["recipe_id"],
        "recipe_sha256": PROMPTMR_PLUS_RECIPE_SHA256,
        "source_commit": PROMPTMR_PLUS_RECIPE["source"]["commit"],
    }
