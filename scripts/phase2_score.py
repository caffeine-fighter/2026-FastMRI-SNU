#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


def tiebreaker(ms_per_slice: float) -> float:
    if ms_per_slice <= 80.0:
        return 0.001
    if ms_per_slice >= 2000.0:
        return 0.0
    return 0.001 * (2000.0 - ms_per_slice) / (2000.0 - 80.0)


def find_float(patterns, text, name, required=True):
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return float(m.group(1))
    if required:
        raise SystemExit(f"ERROR: could not parse {name}")
    return None


def main():
    parser = argparse.ArgumentParser(description="Parse Phase 2 recon_eval.sh output and compute total score.")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    text = args.log.read_text(encoding="utf-8", errors="replace")

    ssim_full = find_float(
        [
            r"Leaderboard\s+SSIM[_ ]full\s*:\s*([0-9.]+)",
            r"SSIM[_ ]full\s*:\s*([0-9.]+)",
        ],
        text,
        "Leaderboard SSIM_full",
    )

    ssim_bbox = find_float(
        [
            r"Leaderboard\s+SSIM[_ ]bbox\s*:\s*([0-9.]+)",
            r"SSIM[_ ]bbox\s*:\s*([0-9.]+)",
        ],
        text,
        "Leaderboard SSIM_bbox",
    )

    total_time_s = find_float(
        [
            r"Leaderboard\s+Recon\s+Time\s*:\s*([0-9.]+)\s*s",
            r"Recon\s+Time\s*\(total\)\s*:\s*([0-9.]+)\s*s",
        ],
        text,
        "Leaderboard Recon Time seconds",
        required=False,
    )

    ms_per_slice = find_float(
        [
            r"Leaderboard\s+Recon\s+Time\s*:\s*[0-9.]+\s*s\s*\(\s*([0-9.]+)\s*ms\s*/\s*slice\s*\)",
            r"\(\s*([0-9.]+)\s*ms\s*/\s*slice\s*\)",
        ],
        text,
        "ms/slice",
    )

    details = {
        "ssim_full_acc4": find_float([r"SSIM[_ ]full\s*\(acc4\)\s*:\s*([0-9.]+)"], text, "SSIM_full acc4", required=False),
        "ssim_full_acc8": find_float([r"SSIM[_ ]full\s*\(acc8\)\s*:\s*([0-9.]+)"], text, "SSIM_full acc8", required=False),
        "ssim_bbox_acc4": find_float([r"SSIM[_ ]bbox\s*\(acc4\)\s*:\s*([0-9.]+)"], text, "SSIM_bbox acc4", required=False),
        "ssim_bbox_acc8": find_float([r"SSIM[_ ]bbox\s*\(acc8\)\s*:\s*([0-9.]+)"], text, "SSIM_bbox acc8", required=False),
        "recon_time_acc4_s": find_float([r"Recon\s+Time\s*\(acc4\)\s*:\s*([0-9.]+)\s*s"], text, "Recon Time acc4", required=False),
        "recon_time_acc8_s": find_float([r"Recon\s+Time.*\(acc8\)\s*:\s*([0-9.]+)\s*s"], text, "Recon Time acc8", required=False),
    }

    quality_score = 0.5 * ssim_full + 0.5 * ssim_bbox
    time_score = tiebreaker(ms_per_slice)
    total_score = quality_score + time_score

    result = {
        "tag": args.tag,
        "log": str(args.log),
        "ssim_full": ssim_full,
        "ssim_bbox": ssim_bbox,
        "quality_score": quality_score,
        "recon_time_s": total_time_s,
        "time_ms_per_slice": ms_per_slice,
        "time_score": time_score,
        "total_score": total_score,
        **details,
    }

    print("=== Phase 2 score ===")
    for key, value in result.items():
        print(f"{key}: {value}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"saved json: {args.out_json}")

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        write_header = not args.out_csv.exists()
        with args.out_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(result.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(result)
        print(f"appended csv: {args.out_csv}")


if __name__ == "__main__":
    main()
