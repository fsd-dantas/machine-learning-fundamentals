"""
Strategy 4 — Fine-tuning + data augmentation.

Strategy 3 with exactly one thing changed: the training stream is randomly distorted.
Everything else — backbone, head, split, schedule, seeds — is held fixed, so any
difference in test accuracy is attributable to the augmentation and to nothing else.
Augmentation is applied *only* to the training stream; validation and test images are
never distorted (`common.make_tf_dataset` enforces this by tying `augment` to `shuffle`).

This script answers the three open questions attached to strategy 4.

(b) **Optimiser** (`--ablation-optimizer`): Adam vs AdamW vs SGD+Nesterov vs RMSprop.
    SGD gets a 10x higher learning rate, because comparing optimisers at an learning
    rate tuned for Adam is a rigged fight, not an experiment.

(c) **Augmentation policy** (`--ablation-policy`):

    lecture   rotation 10 deg, zoom 0.15, shift 0.1  — the `ImageDataGenerator` from
              the course notebook, transcribed to Keras layers. Note what is *missing*:
              no horizontal flip. On CIFAR-10 that is the single most effective and
              most obviously label-preserving transform available (a mirrored truck is
              still a truck), so this policy is expected to leave accuracy on the table.
    flip      horizontal flip only — the cheapest possible policy.
    flip_crop horizontal flip + 12.5% random translation — the canonical CIFAR-10
              recipe used by essentially every published baseline since 2015.
    strong    flip + rotation + zoom + translation + contrast jitter — tests whether
              piling on distortions keeps helping or starts destroying the signal.

Every augmentation transform must be *label-preserving*. Vertical flip is deliberately
excluded: an upside-down horse is not a natural image, and CIFAR-10's test set contains
none, so training on them would only add noise.

Usage:
    python s4_augmentation.py --policy flip_crop --seed 42
    python s4_augmentation.py --ablation-policy      # 4 policies x 3 seeds
    python s4_augmentation.py --ablation-optimizer   # 4 optimisers x 3 seeds
"""

from __future__ import annotations

import argparse

import common
from s3_finetuning import run_one

POLICIES = ("lecture", "flip", "flip_crop", "strong")
OPTIMIZERS = ("adam", "adamw", "sgd", "rmsprop")


def build_policy(name: str, seed: int):
    """Return the augmentation pipeline for a policy, as Keras preprocessing layers.

    Keras layers are used instead of the notebook's `ImageDataGenerator` because they
    run on the GPU inside the `tf.data` graph. `ImageDataGenerator` augments on the CPU
    and becomes the bottleneck the moment the images are upsampled to 224x224 — it is
    also deprecated in Keras 3.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    spec = {
        "lecture": [
            layers.RandomRotation(10 / 360, seed=seed),      # 10 degrees
            layers.RandomZoom(0.15, seed=seed),
            layers.RandomTranslation(0.1, 0.1, seed=seed),
        ],
        "flip": [
            layers.RandomFlip("horizontal", seed=seed),
        ],
        "flip_crop": [
            layers.RandomFlip("horizontal", seed=seed),
            layers.RandomTranslation(0.125, 0.125, seed=seed),
        ],
        "strong": [
            layers.RandomFlip("horizontal", seed=seed),
            layers.RandomRotation(15 / 360, seed=seed),
            layers.RandomZoom(0.15, seed=seed),
            layers.RandomTranslation(0.125, 0.125, seed=seed),
            layers.RandomContrast(0.2, seed=seed),
        ],
    }[name]
    return keras.Sequential(spec, name=name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="mobilenetv2",
                        choices=list(common.BACKBONES))
    parser.add_argument("--head", default="gap", choices=["flatten", "gmp", "gap"])
    parser.add_argument("--policy", default="flip_crop", choices=POLICIES)
    parser.add_argument("--optimizer", default="adam", choices=OPTIMIZERS)
    parser.add_argument("--seed", type=int, default=common.SEED)
    parser.add_argument("--epochs-frozen", type=int, default=20)
    parser.add_argument("--epochs-finetune", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--unfreeze-layers", type=int, default=30)
    parser.add_argument("--ablation-policy", action="store_true",
                        help="open question 4(c): 4 policies x 3 seeds")
    parser.add_argument("--ablation-optimizer", action="store_true",
                        help="open question 4(b): 4 optimisers x 3 seeds")
    args = parser.parse_args()

    device = common.configure_gpu()
    splits = common.load_splits()
    print(f"device: {device} · {splits.summary()}")

    # Augmentation slows convergence (each epoch shows the net a harder problem), so
    # strategy 4 gets more epochs than strategy 3. Early stopping on val accuracy
    # keeps that from becoming an unfair advantage: a model that stops improving stops
    # training, whatever its epoch budget says.
    base = dict(epochs_frozen=args.epochs_frozen, epochs_finetune=args.epochs_finetune,
                batch_size=args.batch_size, frozen_only=False,
                unfreeze_layers=args.unfreeze_layers, device=device,
                strategy="s4_augment")

    if args.ablation_policy:
        for policy in POLICIES:
            for seed in common.SEEDS:
                run_one(args.backbone, args.head, seed, splits,
                        augment=build_policy(policy, seed), optimizer=args.optimizer,
                        label=f"policy_{policy}", **base)
    elif args.ablation_optimizer:
        for optimizer in OPTIMIZERS:
            for seed in common.SEEDS:
                run_one(args.backbone, args.head, seed, splits,
                        augment=build_policy(args.policy, seed), optimizer=optimizer,
                        label=f"optimizer_{optimizer}", **base)
    else:
        run_one(args.backbone, args.head, args.seed, splits,
                augment=build_policy(args.policy, args.seed), optimizer=args.optimizer,
                label=f"{args.backbone}_{args.head}_{args.policy}", **base)


if __name__ == "__main__":
    main()
