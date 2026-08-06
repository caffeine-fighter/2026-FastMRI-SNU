#!/usr/bin/env python3
"""Deterministic legal Cartesian-mask family used by PromptMR+ training.

This module contains no leaderboard paths, payloads, frequencies, targets, or
learned state.  It implements only the scalar generator family selected during
external research so the final VESSL run can start from a fresh initialization.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np


SCHEMA = "promptmr-legal-cartesian-mask-family-v1"
CENTER_FRACTION = 0.08
SUPPORTED_ACCELERATIONS = (4, 8)


def _binary_vector(raw: np.ndarray) -> np.ndarray:
    value = np.asarray(raw)
    while value.ndim > 1 and 1 in value.shape:
        value = np.squeeze(value)
    if value.ndim != 1 or value.size <= 0:
        raise ValueError("sampling mask must be a non-empty one-dimensional vector")
    if not np.isfinite(value).all():
        raise ValueError("sampling mask contains non-finite values")
    rounded = np.rint(value)
    if not np.array_equal(value, rounded) or not np.isin(rounded, (0, 1)).all():
        raise ValueError("sampling mask must be binary")
    return rounded.astype(np.bool_)


def legal_acs_bounds(width: int) -> tuple[int, int]:
    width = int(width)
    if width <= 0:
        raise ValueError("width must be positive")
    acs_width = int(round(width * CENTER_FRACTION))
    if not 0 < acs_width <= width:
        raise ValueError("invalid ACS width")
    start = (width - acs_width + 1) // 2
    return start, start + acs_width


def legal_cartesian_mask(
    width: int,
    acceleration: int,
    residue: int,
) -> np.ndarray:
    width = int(width)
    acceleration = int(acceleration)
    residue = int(residue)
    if acceleration not in SUPPORTED_ACCELERATIONS:
        raise ValueError("acceleration must be 4 or 8")
    if not 0 <= residue < acceleration:
        raise ValueError("residue is outside its acceleration family")
    start, end = legal_acs_bounds(width)
    mask = np.zeros(width, dtype=np.bool_)
    mask[residue::acceleration] = True
    mask[start:end] = True
    return mask


def parse_legal_cartesian_mask(
    raw: np.ndarray,
    *,
    acceleration: int | None = None,
) -> dict[str, int | float | str] | None:
    """Return exact generator features or ``None`` for generalist fail-safe.

    Exact equality against every legal residue deliberately rejects ambiguous,
    malformed, or out-of-family masks instead of guessing a specialist route.
    """

    try:
        mask = _binary_vector(raw)
    except ValueError:
        return None
    accelerations = (
        (int(acceleration),)
        if acceleration is not None
        else SUPPORTED_ACCELERATIONS
    )
    matches: list[tuple[int, int]] = []
    for candidate_acceleration in accelerations:
        if candidate_acceleration not in SUPPORTED_ACCELERATIONS:
            continue
        for residue in range(candidate_acceleration):
            if np.array_equal(
                mask,
                legal_cartesian_mask(
                    int(mask.size),
                    candidate_acceleration,
                    residue,
                ),
            ):
                matches.append((candidate_acceleration, residue))
    if len(matches) != 1:
        return None
    matched_acceleration, residue = matches[0]
    acs_start, acs_end = legal_acs_bounds(int(mask.size))
    density = float(np.count_nonzero(mask)) / float(mask.size)
    return {
        "schema": SCHEMA,
        "acceleration": matched_acceleration,
        "mask_density": density,
        "period": matched_acceleration,
        "offset": residue,
        "native_width": int(mask.size),
        "acs_width": acs_end - acs_start,
        "acs_start": acs_start,
        "acs_end_exclusive": acs_end,
        "residue": residue,
        "normalized_acs_width": (acs_end - acs_start) / int(mask.size),
        "normalized_native_width": int(mask.size) / 384.0,
        "normalized_residue": residue / matched_acceleration,
    }


def residue_for_sample(
    *,
    seed: int,
    epoch: int,
    filename: str,
    slice_number: int,
    acceleration: int,
) -> int:
    """Cycle every sample through every residue over consecutive epochs."""

    acceleration = int(acceleration)
    if acceleration not in SUPPORTED_ACCELERATIONS:
        raise ValueError("acceleration must be 4 or 8")
    if int(epoch) < 0 or int(slice_number) < 0:
        raise ValueError("epoch and slice number must be non-negative")
    token = (
        f"{SCHEMA}\0{int(seed)}\0{filename}\0{int(slice_number)}"
    ).encode("utf-8")
    base = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
    return (base + int(epoch)) % acceleration


def training_shape(
    *,
    kspace_height: int,
    native_width: int,
    target_height: int,
    target_width: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Preserve native readout width while bounding the phase-encode height."""

    values = tuple(
        map(int, (kspace_height, native_width, target_height, target_width))
    )
    if min(values) <= 0:
        raise ValueError("spatial dimensions must be positive")
    kspace_shape = (min(384, values[0]), values[1])
    output_shape = (
        min(values[2], kspace_shape[0]),
        min(values[3], kspace_shape[1]),
    )
    return kspace_shape, output_shape


def coverage_manifest(
    inventory: Iterable[dict[str, int | str]],
    *,
    seed: int,
) -> dict[str, object]:
    """Describe every legal width×acceleration×residue cell, without scores."""

    records = []
    seen: set[tuple[int, int]] = set()
    for raw in inventory:
        acceleration = int(raw["acceleration"])
        width = int(raw["width"])
        if acceleration not in SUPPORTED_ACCELERATIONS:
            raise ValueError("inventory contains an unsupported acceleration")
        legal_acs_bounds(width)
        records.append(
            {
                "name": str(raw["name"]),
                "acceleration": acceleration,
                "native_width": width,
            }
        )
        seen.add((acceleration, width))
    cells = []
    for acceleration, width in sorted(seen):
        start, end = legal_acs_bounds(width)
        for residue in range(acceleration):
            cells.append(
                {
                    "acceleration": acceleration,
                    "native_width": width,
                    "acs_width": end - start,
                    "acs_start": start,
                    "residue": residue,
                }
            )
    return {
        "schema": "promptmr-legal-mask-coverage-manifest-v1",
        "generator_schema": SCHEMA,
        "seed": int(seed),
        "center_fraction": CENTER_FRACTION,
        "residue_policy": "per_sample_full_cycle_across_epochs",
        "sampling_weight_policy": "uniform_legal_residues_not_public_frequency",
        "source_scope": "organizer_train_full_kspace_only",
        "public_target_or_image_used": False,
        "public_payload_required_on_vessl": False,
        "unknown_mask_route": "generalist",
        "records": sorted(
            records,
            key=lambda item: (
                item["acceleration"],
                item["native_width"],
                item["name"],
            ),
        ),
        "coverage_cells": cells,
    }
