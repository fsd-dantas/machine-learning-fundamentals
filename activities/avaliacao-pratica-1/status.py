"""
What has run, what has not, and the exact commands to finish.

Written because `run_all.py` has no skip-if-exists check: re-running a stage retrains
everything in it and spends the GPU minutes again. After a dropped Colab session or an
interrupted cell, the recovery move is to run only what is missing — and that requires
knowing what is missing, which is what this prints.

It compares `results/` against the full expected inventory (every strategy, ablation and
seed the assignment needs) and emits a copy-pasteable command list for the gap.

Usage:
    python status.py            # inventory + the commands to finish
    python status.py --commands # just the commands, one per line
"""

from __future__ import annotations

import argparse
import sys

import common

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BACKBONES = ["mobilenetv2", "resnet50", "inceptionv3"]
CLASSIFIERS = ["svm", "mlp"]
HEADS = ["flatten", "gmp", "gap"]
POLICIES = ["lecture", "flip", "flip_crop", "strong"]
OPTIMIZERS = ["adam", "adamw", "sgd", "rmsprop"]
ABLATION_SEEDS = [42, 7]

VIT = "vit_base_patch16_224_in21k"


def expected() -> dict[str, list[tuple[str, str]]]:
    """(run_id, command) per stage, in the order the runs are produced."""
    plan: dict[str, list[tuple[str, str]]] = {}

    plan["core"] = [
        ("s1_scratch__cnn_scratch__seed42", "python s1_cnn_scratch.py"),
        ("s2_features__mobilenetv2_svm__seed42",
         "python s2_feature_extraction.py --backbone mobilenetv2 --classifier svm"),
        ("s3_finetune__mobilenetv2_gap__seed42",
         "python s3_finetuning.py --backbone mobilenetv2 --head gap"),
        ("s4_augment__mobilenetv2_gap_flip_crop__seed42",
         "python s4_augmentation.py --policy flip_crop"),
        (f"s5_vit__{VIT}__seed42", "python s5_vit.py"),
    ]

    # 2(a) backbone swap — one command produces all six
    plan["ablation 2(a) backbone"] = [
        (f"s2_features__{b}_{c}__seed42", "python s2_feature_extraction.py --ablation")
        for b in BACKBONES for c in CLASSIFIERS
    ]
    # 4(a) Flatten vs GMP vs GAP
    plan["ablation 4(a) head"] = [
        (f"s3_finetune__mobilenetv2_{h}__seed{s}",
         "python s3_finetuning.py --ablation-head --seeds 42 7")
        for h in HEADS for s in ABLATION_SEEDS
    ]
    # 4(c) augmentation policy
    plan["ablation 4(c) policy"] = [
        (f"s4_augment__policy_{p}__seed{s}",
         "python s4_augmentation.py --ablation-policy --seeds 42 7")
        for p in POLICIES for s in ABLATION_SEEDS
    ]
    # 4(b) optimiser
    plan["ablation 4(b) optimizer"] = [
        (f"s4_augment__optimizer_{o}__seed{s}",
         "python s4_augmentation.py --ablation-optimizer --seeds 42 7")
        for o in OPTIMIZERS for s in ABLATION_SEEDS
    ]

    plan["seeds"] = [
        (run_id, cmd)
        for seed in (7, 2024)
        for run_id, cmd in (
            (f"s1_scratch__cnn_scratch__seed{seed}",
             f"python s1_cnn_scratch.py --seed {seed}"),
            (f"s3_finetune__mobilenetv2_gap__seed{seed}",
             f"python s3_finetuning.py --backbone mobilenetv2 --head gap --seed {seed}"),
            (f"s4_augment__mobilenetv2_gap_flip_crop__seed{seed}",
             f"python s4_augmentation.py --policy flip_crop --seed {seed}"),
            (f"s5_vit__{VIT}__seed{seed}", f"python s5_vit.py --seed {seed}"),
        )
    ]

    plan["baseline (lecture setup)"] = [
        ("s3_finetune__vgg16_flatten_frozen__seed42",
         "python s3_finetuning.py --backbone vgg16 --head flatten --frozen-only"),
    ]
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commands", action="store_true")
    args = parser.parse_args()

    have = {path.stem for path in common.RESULTS_DIR.glob("*.json")}
    plan = expected()

    todo: list[str] = []
    seen: set[str] = set()

    if not args.commands:
        print(f"{len(have)} result artefacts in results/\n")

    for stage, items in plan.items():
        done = [r for r, _ in items if r in have]
        missing = [(r, c) for r, c in items if r not in have]
        if not args.commands:
            mark = "OK  " if not missing else "-- "
            print(f"{mark} {stage:26s} {len(done)}/{len(items)}")
            for run_id, _ in missing:
                print(f"        missing: {run_id}")
        for _, cmd in missing:
            if cmd not in seen:      # one command can produce several runs
                seen.add(cmd)
                todo.append(cmd)

    if args.commands:
        for cmd in todo:
            print(cmd)
        return

    print()
    if not todo:
        print("nothing missing — run `python report.py` and build the PDF.")
        return

    print("to finish, run only these (an ablation command re-runs every arm it owns,\n"
          "so a partially-complete ablation repeats the arms it already did):\n")
    for cmd in todo:
        print(f"  !{cmd}")
    print("\n  !python report.py")


if __name__ == "__main__":
    main()
