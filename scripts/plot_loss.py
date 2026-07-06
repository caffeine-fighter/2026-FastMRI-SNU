#!/usr/bin/env python3
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Plot validation loss from val_loss_log.npy.")
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--loss-log", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()

def main():
    args = parse_args()

    if args.exp_name and args.loss_log is None:
        args.loss_log = Path("../result") / args.exp_name / "val_loss_log.npy"

    if args.exp_name and args.out is None:
        args.out = Path("reports") / "figures" / f"{args.exp_name}_val_loss.png"

    if args.loss_log is None:
        raise SystemExit("ERROR: provide --loss-log or --exp-name")
    if args.out is None:
        raise SystemExit("ERROR: provide --out or --exp-name")
    if not args.loss_log.exists():
        raise SystemExit(f"ERROR: loss log not found: {args.loss_log}")

    import numpy as np
    import matplotlib.pyplot as plt

    log = np.load(args.loss_log)
    if log.ndim != 2 or log.shape[1] < 2:
        raise SystemExit(f"ERROR: expected shape (N, 2), got {log.shape}")

    epochs = log[:, 0]
    losses = log[:, 1]

    best_idx = int(np.argmin(losses))
    print(f"best_epoch: {epochs[best_idx]}")
    print(f"best_val_loss: {losses[best_idx]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(epochs, losses, marker="o")
    plt.xlabel("epoch")
    plt.ylabel("validation loss")
    plt.title(args.exp_name or args.loss_log.stem)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"saved: {args.out}")

if __name__ == "__main__":
    main()
