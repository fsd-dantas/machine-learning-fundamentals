"""
Run several scripts concurrently, one per GPU.

Kaggle's `GPU T4 x2` gives two accelerators, and the runs in this experiment are
independent: the ablations do not read each other's output, and each writes result JSONs
under a distinct label. Two lanes therefore halve the wall-clock of the expensive stages —
which matters, because a free-tier session can be reclaimed and everything not yet
downloaded dies with it.

    python run_parallel.py "s4_augmentation.py --ablation-policy --seeds 42 7" \
                           "s4_augmentation.py --ablation-optimizer --seeds 42 7"

Commands are handed to GPUs round-robin via `CUDA_VISIBLE_DEVICES`, so each process sees
exactly one device and TensorFlow's memory-growth setting cannot collide.

**The splits are materialised before any lane starts.** Otherwise two processes race to
build `data/cifar10_splits.npz` on first use, and a half-written `.npz` read by the other
lane is the kind of failure that produces a plausible-looking wrong number rather than a
crash.

Logs are streamed to `logs/<lane>.log` and tailed on failure — interleaving two training
runs on one stdout is unreadable.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import common

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LOG_DIR = common.HERE / "logs"


def visible_gpus() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--list-gpus"], capture_output=True,
                             text=True, check=True).stdout
        return len([line for line in out.splitlines() if line.strip()])
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commands", nargs="+",
                        help='each a quoted command, e.g. "s1_cnn_scratch.py --seed 7"')
    parser.add_argument("--gpus", type=int, default=None,
                        help="lanes to use (default: every GPU detected)")
    args = parser.parse_args()

    n_gpus = args.gpus or visible_gpus()
    if n_gpus < 1:
        raise SystemExit("no GPU detected — run the scripts directly instead")

    LOG_DIR.mkdir(exist_ok=True)

    # Build the splits once, here, before any lane can race for them.
    print("materialising splits…")
    print(f"  {common.load_splits().summary()}\n")

    procs = []
    started = time.perf_counter()
    for i, command in enumerate(args.commands):
        gpu = i % n_gpus
        log = LOG_DIR / f"lane{i}_gpu{gpu}.log"
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        handle = log.open("w", encoding="utf-8")
        proc = subprocess.Popen([sys.executable, *command.split()], cwd=common.HERE,
                                env=env, stdout=handle, stderr=subprocess.STDOUT)
        procs.append((command, gpu, proc, handle, log))
        print(f"GPU {gpu} ← python {command}")
        print(f"        log: {log.relative_to(common.HERE)}")

    print(f"\n{len(procs)} lanes running on {n_gpus} GPU(s). Waiting…\n", flush=True)

    failures = []
    for command, gpu, proc, handle, log in procs:
        code = proc.wait()
        handle.close()
        elapsed = (time.perf_counter() - started) / 60
        if code == 0:
            print(f"OK   [GPU {gpu}] {command}  ({elapsed:.1f} min elapsed)", flush=True)
        else:
            failures.append(command)
            print(f"FAIL [GPU {gpu}] {command}  (exit {code})", file=sys.stderr)
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
            print("\n".join(tail), file=sys.stderr)

    total = (time.perf_counter() - started) / 60
    print(f"\ntotal wall-clock: {total:.1f} min · "
          f"{len(list(common.RESULTS_DIR.glob('*.json')))} artefacts in results/")
    if failures:
        raise SystemExit(f"{len(failures)} lane(s) failed")


if __name__ == "__main__":
    main()
