"""
Orchestrator — runs every strategy and every ablation, in dependency order.

Sized for a free-tier Colab T4. The full sweep is roughly 4.5 hours, which does not
fit in one uninterrupted free session; it is therefore split into stages that can be
run independently and resumed. Results accumulate in `results/` as JSON, so a stage
that already ran is never repeated and a disconnected session costs only its current
stage.

Stage order matters exactly once: `s1` (or any TF script) must run before `s5`,
because it is what materialises the cached `.npz` splits that the PyTorch script
reads back.

Usage:
    python run_all.py --stage core        # the 5 strategies, primary seed   (~70 min)
    python run_all.py --stage seeds       # the 5 strategies, 3 seeds        (~2 h)
    python run_all.py --stage ablations   # the 4 open questions             (~2.5 h)
    python run_all.py --stage report      # tables + confusion matrix        (~10 s)
    python run_all.py --all
    python run_all.py --stage core --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

import common

STAGES: dict[str, list[list[str]]] = {
    # The headline comparison: one configuration per strategy, primary seed.
    "core": [
        ["s1_cnn_scratch.py"],
        ["s2_feature_extraction.py", "--backbone", "mobilenetv2", "--classifier", "svm"],
        ["s3_finetuning.py", "--backbone", "mobilenetv2", "--head", "gap"],
        ["s4_augmentation.py", "--policy", "flip_crop"],
        ["s5_vit.py"],
    ],
    # Dispersion for the headline numbers: no ranking claim is made from one run.
    "seeds": [
        cmd + ["--seed", str(seed)]
        for seed in common.SEEDS[1:]
        for cmd in (
            ["s1_cnn_scratch.py"],
            ["s3_finetuning.py", "--backbone", "mobilenetv2", "--head", "gap"],
            ["s4_augmentation.py", "--policy", "flip_crop"],
            ["s5_vit.py"],
        )
    ],
    # The four open questions in the assignment.
    "ablations": [
        ["s2_feature_extraction.py", "--ablation"],          # 2(a) backbone swap
        ["s3_finetuning.py", "--ablation-head"],             # 4(a) Flatten vs GMP
        ["s4_augmentation.py", "--ablation-optimizer"],      # 4(b) optimiser
        ["s4_augmentation.py", "--ablation-policy"],         # 4(c) augmentation policy
    ],
    # Reference run: the lecture notebook's exact setup (VGG16, Flatten, frozen
    # backbone, no phase 2), so the report can quantify what the departures bought.
    "baseline": [
        ["s3_finetuning.py", "--backbone", "vgg16", "--head", "flatten", "--frozen-only"],
    ],
    "report": [
        ["report.py"],
    ],
}

ORDER = ["core", "seeds", "ablations", "baseline", "report"]


def run(cmd: list[str], *, dry_run: bool) -> bool:
    printable = " ".join(["python"] + cmd)
    print(f"\n{'='*78}\n>>> {printable}\n{'='*78}", flush=True)
    if dry_run:
        return True

    started = time.perf_counter()
    result = subprocess.run([sys.executable, *cmd], cwd=common.HERE, check=False)
    elapsed = (time.perf_counter() - started) / 60

    if result.returncode != 0:
        print(f"!!! FAILED after {elapsed:.1f} min: {printable}", file=sys.stderr)
        return False
    print(f"<<< done in {elapsed:.1f} min", flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=ORDER, action="append", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true",
                        help="do not abort the stage when one run fails")
    args = parser.parse_args()

    stages = ORDER if args.all else (args.stage or ["core"])
    started = time.perf_counter()
    failures: list[str] = []

    for stage in stages:
        print(f"\n########## STAGE: {stage} ##########")
        for cmd in STAGES[stage]:
            if not run(cmd, dry_run=args.dry_run):
                failures.append(" ".join(cmd))
                if not args.keep_going:
                    raise SystemExit(1)

    total = (time.perf_counter() - started) / 60
    print(f"\n{'='*78}\ntotal: {total:.1f} min · stages: {', '.join(stages)}")
    if failures:
        print(f"failures: {len(failures)}")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
