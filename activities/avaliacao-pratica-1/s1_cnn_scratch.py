"""
Strategy 1 — CNN trained from scratch on CIFAR-10 (32x32, no pretrained weights).

The baseline every transfer-learning strategy has to beat. It is also the only
strategy that sees the images at their native 32x32 resolution: a from-scratch
network has no ImageNet statistics to match, so upsampling would only waste
compute. That resolution gap is a declared property of the comparison, not a
defect — it is what "from scratch" structurally means here.

Architecture: a VGG-style stack (3 conv blocks, doubling widths, batch norm,
progressive dropout) with a global-average-pooled head. ~1.2M parameters, all
trainable, trained from random init on the 10,000-image budget.

Usage:
    python s1_cnn_scratch.py --seed 42
    python s1_cnn_scratch.py --seed 42 --augment   # scratch + augmentation
"""

from __future__ import annotations

import argparse

import numpy as np

import common


def build_model(seed: int):
    from tensorflow import keras
    from tensorflow.keras import layers

    init = keras.initializers.HeNormal(seed=seed)

    def block(x, filters, dropout):
        for _ in range(2):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                              kernel_initializer=init)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(2)(x)
        return layers.Dropout(dropout, seed=seed)(x)

    inputs = keras.Input(shape=(32, 32, 3))
    x = block(inputs, 32, 0.20)
    x = block(x, 64, 0.30)
    x = block(x, 128, 0.40)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu", kernel_initializer=init)(x)
    x = layers.Dropout(0.5, seed=seed)(x)
    # float32 output: with mixed_float16 the softmax must not run in half precision
    outputs = layers.Dense(common.N_CLASSES, activation="softmax",
                           dtype="float32", kernel_initializer=init)(x)
    return keras.Model(inputs, outputs, name="cnn_scratch")


def augmentation_layers(seed: int):
    from tensorflow import keras
    from tensorflow.keras import layers

    return keras.Sequential([
        layers.RandomFlip("horizontal", seed=seed),
        layers.RandomTranslation(0.1, 0.1, seed=seed),
    ], name="augment")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=common.SEED)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--augment", action="store_true",
                        help="train with horizontal flip + translation")
    args = parser.parse_args()

    from tensorflow import keras

    device = common.configure_gpu()
    common.set_seed(args.seed)
    splits = common.load_splits()
    print(f"device: {device} · {splits.summary()}")

    label = "cnn_scratch_augmented" if args.augment else "cnn_scratch"
    augment = augmentation_layers(args.seed) if args.augment else None

    # scale to [0,1]; a from-scratch net has no pretrained statistics to honour
    def preprocess(images):
        return images / 255.0

    train_ds = common.make_tf_dataset(
        splits.x_train, splits.y_train, image_size=32, preprocess=preprocess,
        batch_size=args.batch_size, shuffle=True, augment=augment, seed=args.seed)
    val_ds = common.make_tf_dataset(
        splits.x_val, splits.y_val, image_size=32, preprocess=preprocess,
        batch_size=args.batch_size)
    test_ds = common.make_tf_dataset(
        splits.x_test, splits.y_test, image_size=32, preprocess=preprocess,
        batch_size=args.batch_size)

    model = build_model(args.seed)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    early = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=15, mode="max", restore_best_weights=True)
    reduce = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5)

    with common.Timer() as timer:
        history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                            callbacks=[early, reduce], verbose=2)

    val_acc = history.history["val_accuracy"]
    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)

    common.score_run(
        strategy="s1_scratch",
        label=label,
        seed=args.seed,
        config={"architecture": "vgg-style 3-block CNN", "optimizer": "Adam(1e-3)",
                "batch_size": args.batch_size, "augmentation": bool(args.augment),
                "pretrained": False},
        y_true=splits.y_test,
        y_pred=y_pred,
        input_resolution=32,
        params_total=model.count_params(),
        params_trainable=sum(int(np.prod(w.shape)) for w in model.trainable_weights),
        epochs_run=len(val_acc),
        best_epoch=int(np.argmax(val_acc)) + 1,
        train_seconds=timer.seconds,
        device=device,
    )


if __name__ == "__main__":
    main()
