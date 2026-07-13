"""
Strategy 2 — Pretrained CNN as a frozen feature extractor + shallow classifier.

The backbone is never trained: images go through it once, the globally-averaged
activations become a fixed-length vector, and a shallow model (SVM or MLP) is
fitted on those vectors. This is transfer learning at its cheapest — one forward
pass over the data, then a classical scikit-learn problem.

Open question 2(a) — *does swapping the CNN for a simpler one such as MobileNet
significantly change the result?* — is answered here by holding the classifier,
the data and the protocol fixed and varying only the backbone. MobileNetV2
(3.5M params, 0.6 GFLOPs) against ResNet50 (25.6M, 8.2) and InceptionV3
(23.9M, 11.5): a ~14x parameter gap and a ~19x compute gap. If accuracy barely
moves, the honest engineering conclusion is that the heavy backbones are not
buying anything on this task.

Extracted features are cached to `data/features_<backbone>.npz`, so the classifier
grid can be re-run without touching the GPU again.

Usage:
    python s2_feature_extraction.py --backbone mobilenetv2 --classifier svm
    python s2_feature_extraction.py --ablation          # all backbones x all classifiers
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import common


def extract_features(backbone: common.Backbone, splits: common.Splits, *,
                     image_size: int, batch_size: int = 64) -> dict[str, np.ndarray]:
    """Run the frozen backbone once over every split; cache the result to disk."""
    cache = common.DATA_DIR / f"features_{backbone.key}_{image_size}.npz"
    if cache.exists():
        print(f"[{backbone.display}] using cached features: {cache.name}")
        z = np.load(cache)
        return {k: z[k] for k in z.files}

    res = image_size
    model = backbone.build(pooling="avg", input_shape=(res, res, 3))
    model.trainable = False
    preprocess = backbone.preprocess()

    out: dict[str, np.ndarray] = {}
    for name, x, y in (("train", splits.x_train, splits.y_train),
                       ("val", splits.x_val, splits.y_val),
                       ("test", splits.x_test, splits.y_test)):
        ds = common.make_tf_dataset(x, y, image_size=res, preprocess=preprocess,
                                    batch_size=batch_size)
        with common.Timer() as timer:
            feats = model.predict(ds, verbose=0)
        print(f"[{backbone.display}] {name}: {feats.shape} in {timer.seconds:.0f}s")
        out[f"x_{name}"] = feats.astype(np.float32)

    np.savez_compressed(cache, **out)
    return out


def build_classifier(kind: str, seed: int):
    """The two shallow heads used in the lecture example, wrapped in a scaler.

    Deep features are not standardised by construction, and both an RBF-kernel SVM
    and a gradient-trained MLP are scale-sensitive — so the scaler is fitted on the
    training split only and applied to val/test, never fitted on them.
    """
    if kind == "svm":
        est = SVC(C=10.0, gamma="scale", kernel="rbf", random_state=seed)
    elif kind == "mlp":
        est = MLPClassifier(hidden_layer_sizes=(512,), activation="relu",
                            max_iter=300, early_stopping=True, n_iter_no_change=10,
                            random_state=seed)
    else:
        raise ValueError(f"unknown classifier: {kind}")
    return Pipeline([("scaler", StandardScaler()), ("clf", est)])


def run_one(backbone_key: str, classifier: str, seed: int, splits: common.Splits,
            *, image_size: int) -> None:
    backbone = common.BACKBONES[backbone_key]
    common.set_seed(seed)

    feats = extract_features(backbone, splits, image_size=image_size)
    model = build_classifier(classifier, seed)

    with common.Timer() as timer:
        model.fit(feats["x_train"], splits.y_train)

    val_acc = float(np.mean(model.predict(feats["x_val"]) == splits.y_val))
    y_pred = model.predict(feats["x_test"])
    print(f"[{backbone.display}/{classifier}] val accuracy {val_acc:.4f}")

    common.score_run(
        strategy="s2_features",
        label=f"{backbone_key}_{classifier}",
        seed=seed,
        config={"backbone": backbone.display, "classifier": classifier,
                "feature_dim": int(feats["x_train"].shape[1]),
                "backbone_params_millions": backbone.params_millions,
                "backbone_gflops": backbone.gflops,
                "backbone_trained": False, "val_accuracy": val_acc},
        y_true=splits.y_test,
        y_pred=y_pred,
        input_resolution=image_size,
        params_total=int(backbone.params_millions * 1e6),
        params_trainable=0,  # the backbone is frozen; only the shallow head is fitted
        train_seconds=timer.seconds,
        framework="tensorflow+sklearn",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="mobilenetv2",
                        choices=list(common.BACKBONES))
    parser.add_argument("--classifier", default="svm", choices=["svm", "mlp"])
    parser.add_argument("--seed", type=int, default=common.SEED)
    parser.add_argument("--image-size", type=int, default=common.WORKING_RESOLUTION)
    parser.add_argument("--ablation", action="store_true",
                        help="open question 2(a): every backbone x every classifier")
    args = parser.parse_args()

    device = common.configure_gpu()
    splits = common.load_splits()
    print(f"device: {device} · {splits.summary()} · {args.image_size}px")

    if args.ablation:
        # Deterministic shallow heads on cached features: the backbone comparison is
        # decided by the features, not by initialisation luck, so one seed suffices
        # here. The seed sweep is spent where it matters — on the trained networks.
        for backbone, clf in itertools.product(
                ["mobilenetv2", "resnet50", "inceptionv3"], ["svm", "mlp"]):
            run_one(backbone, clf, args.seed, splits, image_size=args.image_size)
    else:
        run_one(args.backbone, args.classifier, args.seed, splits,
                image_size=args.image_size)


if __name__ == "__main__":
    main()
