"""Sleep until shortly before the job's walltime runs out, then print the
best checkpoint's stats. Runs alongside a training script in the background
so results are visible in the log even if the scheduler kills the job before
it reaches its final epoch.
"""
import argparse
import os
import time

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to the last_*_checkpoint.pt file")
    parser.add_argument("--sleep-seconds", type=float, required=True, help="seconds to wait before reporting")
    args = parser.parse_args()

    time.sleep(args.sleep_seconds)

    if not os.path.exists(args.checkpoint):
        print(f"[print_best_stats] no checkpoint found at {args.checkpoint} after waiting", flush=True)
        return

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    print(
        f"[print_best_stats] job nearing walltime | "
        f"best val loss so far {checkpoint['best_val_loss']:.4f} | "
        f"last completed epoch {checkpoint['epoch']} | "
        f"hyperparams {checkpoint.get('hyperparams')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
