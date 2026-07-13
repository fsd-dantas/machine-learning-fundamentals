"""
Orchestrator — runs every strategy and every ablation, in dependency order.

Sized for a free-tier Colab T4, at `common.WORKING_RESOLUTION` (128px) for the
convolutional strategies.

The stages are ordered by **marginal value per GPU-minute**, so that the deliverable is
complete after the first one and only gets stronger afterwards. If time runs out, what
is missing is dispersion and ablations — not results.

    core       every strategy, primary seed          ~35 min   -> report is deliverable
    ablations  the four open questions, 2 seeds      ~70 min   -> the graded differentiator
    seeds      the remaining seeds for the headline  ~50 min   -> dispersion on the main table
    baseline   the lecture notebook's exact setup    ~10 min   -> nice-to-have
    report     tables, confusion matrix, McNemar     ~10 s     -> CPU, run it after each stage

Results accumulate in `results/` as JSON, so a stage that already ran is never repeated
and a dropped Colab session costs only the run in flight. `report.py` degrades honestly:
configurations with one seed report no standard deviation rather than a fake zero.

Stage order matters exactly once: a TensorFlow script must run before `s5`, because it is
what materialises the cached `.npz` splits the PyTorch ViT reads back.

Usage:
    python run_all.py --stage core
    python run_all.py --stage core --stage ablations --stage report
    python run_all.py --all
    python run_all.py --stage core --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

import common

ABLATION_SEEDS = ["42", "7"]
"""Two seeds for the ablation sweeps, not three.

Two is enough to see whether an effect is larger than run-to-run noise, which is the
only question the open questions actually ask ("does it change the result
*significantly*?"). The third seed buys a better variance estimate at a 50% cost
increase — a bad trade under a deadline, and an honest one to declare.
"""

STAGES: dict[str, list[list[str]]] = {
    # The headline comparison: one configuration per strategy, primary seed.
    "core": [
        ["s1_cnn_scratch.py"],
        ["s2_feature_extraction.py", "--backbone", "mobilenetv2", "--classifier", "svm"],
        ["s3_finetuning.py", "--backbone", "mobilenetv2", "--head", "gap"],
        ["s4_augmentation.py", "--policy", "flip_crop"],
        ["s5_vit.py"],
    ],
    # The four open questions. This is where the marks are: everything above is
    # expected of any submission, and only these are a controlled experiment.
    "ablations": [
        ["s2_feature_extraction.py", "--ablation"],                        # 2(a)
        ["s3_finetuning.py", "--ablation-head", "--seeds", *ABLATION_SEEDS],   # 4(a)
        ["s4_augmentation.py", "--ablation-policy", "--seeds", *ABLATION_SEEDS],  # 4(c)
        ["s4_augmentation.py", "--ablation-optimizer", "--seeds", *ABLATION_SEEDS],  # 4(b)
    ],
    # Dispersion for the headline table: no ranking claim rests on a single run.
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
    # Reference run: the lecture notebook's exact setup (VGG16, Flatten, frozen
    # backbone, no phase 2), so the report can quantify what the departures bought.
    "baseline": [
        ["s3_finetuning.py", "--backbone", "vgg16", "--head", "flatten", "--frozen-only"],
    ],
    "report": [
        ["report.py"],
    ],
}

ORDER = ["core", "ablations", "seeds", "baseline", "report"]


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
