"""
Strategy 3 — Fine-tuning an ImageNet-pretrained CNN on CIFAR-10.

Two phases, in this order, because the order is not optional:

  Phase 1 — backbone frozen, only the new head trains (Adam, 1e-3). The head starts
            at random init; letting its large, noisy gradients flow into pretrained
            convolutional weights would destroy the very features being transferred.
  Phase 2 — the top block is unfrozen and the whole thing trains at a 100x smaller
            learning rate (1e-5), nudging the highest-level (most dataset-specific)
            filters toward CIFAR-10 while the generic early layers stay put.

The lecture notebook sets `vgg_conv.trainable = False` and never unfreezes anything.
That is *feature extraction with a trainable MLP head* — strategy 2 with a deeper
classifier — not fine-tuning. Phase 2 is the difference, and the report quantifies it.

Open question 4(a) — *does replacing `Flatten()` with `GlobalMaxPooling2D()` change
the result significantly?* — is answered by the `--head` ablation. The two are not
cosmetic variants: on MobileNetV2 at 224px the final feature map is 7x7x1280, so

    Flatten -> Dense(512):  62,720 x 512 = 32.1M parameters
    GlobalMaxPool -> Dense(512): 1,280 x 512 =  0.66M parameters

a 49x difference in head capacity, fitted on 10,000 images. Pooling also discards
spatial position, which for object classification is largely a nuisance variable.

BatchNorm caveat: when the top block is unfrozen, its BatchNormalization layers are
deliberately kept in inference mode. Letting them update their running statistics on
small fine-tuning batches is a well-known way to silently wreck a pretrained network.

Usage:
    python s3_finetuning.py --backbone mobilenetv2 --head gap --seed 42
    python s3_finetuning.py --ablation-head        # flatten vs gmp vs gap, 3 seeds
    python s3_finetuning.py --backbone vgg16 --head flatten --frozen-only  # lecture setup
"""

from __future__ import annotations

import argparse

import numpy as np

import common

HEADS = ("flatten", "gmp", "gap")


def build_model(backbone: common.Backbone, head: str, seed: int, *,
                dropout: float = 0.3):
    from tensorflow import keras
    from tensorflow.keras import layers

    init = keras.initializers.GlorotUniform(seed=seed)
    base = backbone.build(pooling=None)
    base.trainable = False

    inputs = keras.Input(shape=(backbone.resolution, backbone.resolution, 3))
    x = base(inputs, training=False)  # keeps BatchNorm in inference mode
    x = {
        "flatten": layers.Flatten(),
        "gmp": layers.GlobalMaxPooling2D(),
        "gap": layers.GlobalAveragePooling2D(),
    }[head](x)
    x = layers.Dense(512, activation="relu", kernel_initializer=init)(x)
    x = layers.Dropout(dropout, seed=seed)(x)
    outputs = layers.Dense(common.N_CLASSES, activation="softmax",
                           dtype="float32", kernel_initializer=init)(x)
    return keras.Model(inputs, outputs, name=f"{backbone.key}_{head}"), base


def unfreeze_top(base, n_layers: int) -> None:
    """Unfreeze the last `n_layers`, except BatchNorm — see the module docstring."""
    from tensorflow.keras import layers as klayers

    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers[-n_layers:]:
        if isinstance(layer, klayers.BatchNormalization):
            layer.trainable = False


def trainable_params(model) -> int:
    return sum(int(np.prod(w.shape)) for w in model.trainable_weights)


def run_one(backbone_key: str, head: str, seed: int, splits: common.Splits, *,
            epochs_frozen: int, epochs_finetune: int, batch_size: int,
            frozen_only: bool, unfreeze_layers: int, device: str,
            optimizer: str = "adam", augment=None, label: str | None = None,
            strategy: str = "s3_finetune") -> None:
    """One full two-phase run. Shared with strategy 4, which passes `augment`."""
    from tensorflow import keras

    common.set_seed(seed)
    backbone = common.BACKBONES[backbone_key]
    res = backbone.resolution
    preprocess = backbone.preprocess()

    train_ds = common.make_tf_dataset(
        splits.x_train, splits.y_train, image_size=res, preprocess=preprocess,
        batch_size=batch_size, shuffle=True, augment=augment, seed=seed)
    val_ds = common.make_tf_dataset(splits.x_val, splits.y_val, image_size=res,
                                    preprocess=preprocess, batch_size=batch_size)
    test_ds = common.make_tf_dataset(splits.x_test, splits.y_test, image_size=res,
                                     preprocess=preprocess, batch_size=batch_size)

    model, base = build_model(backbone, head, seed)
    params_head = trainable_params(model)

    def make_optimizer(lr: float):
        return {
            "adam": lambda: keras.optimizers.Adam(lr),
            "adamw": lambda: keras.optimizers.AdamW(lr, weight_decay=1e-4),
            "sgd": lambda: keras.optimizers.SGD(lr * 10, momentum=0.9, nesterov=True),
            "rmsprop": lambda: keras.optimizers.RMSprop(lr),
        }[optimizer]()

    early = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=8, mode="max", restore_best_weights=True)

    history_val: list[float] = []
    with common.Timer() as timer:
        # ---- Phase 1: head only -------------------------------------------------
        model.compile(optimizer=make_optimizer(1e-3),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        h1 = model.fit(train_ds, validation_data=val_ds, epochs=epochs_frozen,
                       callbacks=[early], verbose=2)
        history_val += h1.history["val_accuracy"]

        # ---- Phase 2: unfreeze the top block ------------------------------------
        if not frozen_only:
            unfreeze_top(base, unfreeze_layers)
            model.compile(optimizer=make_optimizer(1e-5),
                          loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"])
            print(f"phase 2: {trainable_params(model):,} trainable params")
            h2 = model.fit(train_ds, validation_data=val_ds, epochs=epochs_finetune,
                           callbacks=[early], verbose=2)
            history_val += h2.history["val_accuracy"]

    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)

    common.score_run(
        strategy=strategy,
        label=label or f"{backbone_key}_{head}" + ("_frozen" if frozen_only else ""),
        seed=seed,
        config={"backbone": backbone.display, "head": head, "optimizer": optimizer,
                "phase2_finetuning": not frozen_only,
                "unfrozen_layers": 0 if frozen_only else unfreeze_layers,
                "head_params": params_head, "batch_size": batch_size,
                "augmentation": augment.name if augment is not None else None},
        y_true=splits.y_test,
        y_pred=y_pred,
        input_resolution=res,
        params_total=model.count_params(),
        params_trainable=trainable_params(model),
        epochs_run=len(history_val),
        best_epoch=int(np.argmax(history_val)) + 1,
        train_seconds=timer.seconds,
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="mobilenetv2",
                        choices=list(common.BACKBONES))
    parser.add_argument("--head", default="gap", choices=HEADS)
    parser.add_argument("--seed", type=int, default=common.SEED)
    parser.add_argument("--epochs-frozen", type=int, default=15)
    parser.add_argument("--epochs-finetune", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--unfreeze-layers", type=int, default=30)
    parser.add_argument("--frozen-only", action="store_true",
                        help="skip phase 2 — reproduces the lecture notebook's setup")
    parser.add_argument("--ablation-head", action="store_true",
                        help="open question 4(a): flatten vs gmp vs gap over 3 seeds")
    args = parser.parse_args()

    device = common.configure_gpu()
    splits = common.load_splits()
    print(f"device: {device} · {splits.summary()}")

    kwargs = dict(epochs_frozen=args.epochs_frozen, epochs_finetune=args.epochs_finetune,
                  batch_size=args.batch_size, frozen_only=args.frozen_only,
                  unfreeze_layers=args.unfreeze_layers, device=device)

    if args.ablation_head:
        # 3 heads x 3 seeds: a head difference is only reportable if it survives
        # the seed-to-seed spread it has to be measured against.
        for head in HEADS:
            for seed in common.SEEDS:
                run_one(args.backbone, head, seed, splits, **kwargs)
    else:
        run_one(args.backbone, args.head, args.seed, splits, **kwargs)


if __name__ == "__main__":
    main()
