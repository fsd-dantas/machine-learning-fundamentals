"""
Avaliação Prática 1 — CIFAR-10 image classification: shared experimental protocol.
Aprendizagem de Máquina · PPGIA/PUC-PR · MSc 2026

Every strategy (s1..s5) imports this module. It is the single source of truth for
the data splits, the evaluation, and the statistics — so that five very different
models, two frameworks (TensorFlow and PyTorch) and several ablations all remain
comparable to each other.

Three design decisions are load-bearing:

1. **Equal budget.** All strategies see the *same* stratified 10,000-image training
   subsample and the *same* 2,000-image validation set, and are scored once on the
   *full official* 10,000-image CIFAR-10 test set. A strategy cannot win by being
   fed more data than its rivals.

2. **Splits are cached to disk as `.npz`.** Written by TensorFlow on first run, read
   back with plain NumPy afterwards. The PyTorch ViT script therefore trains on
   byte-identical data without importing TensorFlow.

3. **Evaluation happens on saved predictions, not on models.** Each run persists its
   10,000 test predictions into its result JSON. Metrics, confusion matrices and the
   paired McNemar test are all computed downstream from those integers, which makes
   the protocol framework-agnostic and the report regenerable without a GPU.

Images are held as uint8 32x32 (~180 MB for all of CIFAR-10) and resized/preprocessed
on the fly inside the input pipeline. Materialising them at 224x224 float32 — as the
lecture notebooks do for their 1,000-image dataset — would cost ~36 GB here.
"""

from __future__ import annotations

import json
import platform
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# --------------------------------------------------------------------------- #
# Protocol constants — do not change without re-running every strategy.
# --------------------------------------------------------------------------- #

SEED = 42
"""Primary seed. Single-run results are reported at this seed."""

SEEDS: tuple[int, ...] = (42, 7, 2024)
"""Seeds for repeated runs. Any claim of the form 'X beats Y' must be backed by
the mean +/- std over these three, never by a single run."""

TRAIN_PER_CLASS = 1_000  # -> 10,000 training images
VAL_PER_CLASS = 200      # ->  2,000 validation images
N_CLASSES = 10

CLASS_NAMES: tuple[str, ...] = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"
CHECKPOINT_DIR = HERE / "checkpoints"
SPLITS_CACHE = DATA_DIR / "cifar10_splits.npz"

for _d in (DATA_DIR, RESULTS_DIR, CHECKPOINT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Backbone registry
# --------------------------------------------------------------------------- #

WORKING_RESOLUTION = 128
"""Input size for every pretrained CNN (strategies 2-4).

Not the ImageNet-native 224. CIFAR-10 images are 32x32: upsampling them to 224
adds no information whatsoever — it fabricates pixels by interpolation — while
costing ~3x the compute of 128. 128 is still a 4x upsample of the source, it is an
officially supported MobileNetV2 input size, and it keeps the whole sweep inside a
free-tier GPU session. It is applied identically to strategies 2, 3 and 4, so it is
a controlled constant of the comparison, not a variable.

The ViT (strategy 5) stays at 224: its patch grid and position embeddings are fixed
by the pretrained checkpoint.
"""


@dataclass(frozen=True)
class Backbone:
    """An ImageNet-pretrained convolutional backbone from `keras.applications`.

    `resolution` is the native ImageNet input size, kept for reference; the
    experiments run at `WORKING_RESOLUTION`. `params_millions` and `gflops` are what
    make the accuracy-vs-cost trade-off in the report legible — MobileNetV2 is ~50x
    cheaper per image than VGG16, which is the whole point of open question 2(a).
    """

    key: str
    display: str
    resolution: int
    params_millions: float
    gflops: float
    feature_dim: int

    def build(self, *, include_top: bool = False, pooling: str | None = None,
              input_shape: tuple[int, int, int] | None = None):
        from tensorflow.keras import applications

        ctor = {
            "mobilenetv2": applications.MobileNetV2,
            "resnet50": applications.ResNet50,
            "inceptionv3": applications.InceptionV3,
            "vgg16": applications.VGG16,
        }[self.key]
        shape = input_shape or (self.resolution, self.resolution, 3)
        return ctor(include_top=include_top, weights="imagenet",
                    pooling=pooling, input_shape=shape)

    def preprocess(self) -> Callable:
        from tensorflow.keras import applications

        return {
            "mobilenetv2": applications.mobilenet_v2.preprocess_input,
            "resnet50": applications.resnet50.preprocess_input,
            "inceptionv3": applications.inception_v3.preprocess_input,
            "vgg16": applications.vgg16.preprocess_input,
        }[self.key]


BACKBONES: dict[str, Backbone] = {
    "mobilenetv2": Backbone("mobilenetv2", "MobileNetV2", 224, 3.5, 0.60, 1280),
    "resnet50": Backbone("resnet50", "ResNet50", 224, 25.6, 8.20, 2048),
    "inceptionv3": Backbone("inceptionv3", "InceptionV3", 299, 23.9, 11.5, 2048),
    "vgg16": Backbone("vgg16", "VGG16", 224, 138.4, 30.9, 512),
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

@dataclass
class Splits:
    x_train: np.ndarray  # uint8 (10000, 32, 32, 3)
    y_train: np.ndarray  # int64 (10000,)
    x_val: np.ndarray    # uint8 (2000, 32, 32, 3)
    y_val: np.ndarray
    x_test: np.ndarray   # uint8 (10000, 32, 32, 3) — official CIFAR-10 test set
    y_test: np.ndarray

    def summary(self) -> str:
        return (f"train {self.x_train.shape[0]:,} · val {self.x_val.shape[0]:,} · "
                f"test {self.x_test.shape[0]:,} (official split)")


def seed_keras_cache() -> str | None:
    """Point Keras at a locally-attached CIFAR-10 instead of downloading it.

    `cifar10.load_data()` fetches 170 MB from cs.toronto.edu, which on Kaggle has been
    observed at ~35 KB/s — **78 minutes**, before a single epoch runs. That is longer than
    the entire training sweep, and it is spent every time the session restarts.

    Keras caches the extracted archive at `~/.keras/datasets/cifar-10-batches-py`, and
    `get_file(..., untar=True)` skips the download outright when that directory already
    exists. So if the dataset is attached as a Kaggle input (or left over in a Colab
    session), copying it into place turns 78 minutes into 2 seconds.

    Returns the source it found, or None if it will have to download after all.
    """
    cache = Path.home() / ".keras" / "datasets"
    target = cache / "cifar-10-batches-py"
    if target.is_dir():
        return "keras cache"

    import shutil
    import tarfile

    for base in (Path("/kaggle/input"), Path("/content"), HERE / "data"):
        if not base.is_dir():
            continue

        for found in base.rglob("cifar-10-batches-py"):
            if found.is_dir() and (found / "data_batch_1").exists():
                cache.mkdir(parents=True, exist_ok=True)
                shutil.copytree(found, target)
                return str(found)

        for archive in base.rglob("cifar-10-python.tar.gz"):
            cache.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive) as tar:
                tar.extractall(cache)
            if target.is_dir():
                return str(archive)

    return None


def load_splits(*, rebuild: bool = False) -> Splits:
    """Return the fixed splits, building and caching them on first call.

    The subsample is stratified and drawn with a dedicated RNG seeded at `SEED`,
    independent of the training seed: every seed of every strategy trains on the
    *same* 10,000 images. Only weight init and augmentation vary across seeds, so
    a difference between two runs cannot be an artefact of a luckier training set.
    """
    if SPLITS_CACHE.exists() and not rebuild:
        z = np.load(SPLITS_CACHE)
        return Splits(**{k: z[k] for k in
                         ("x_train", "y_train", "x_val", "y_val", "x_test", "y_test")})

    source = seed_keras_cache()
    print(f"CIFAR-10: {source}" if source
          else "CIFAR-10: downloading (170 MB — attach it as an input to skip this)")

    from tensorflow.keras.datasets import cifar10

    (x_full, y_full), (x_test, y_test) = cifar10.load_data()
    y_full = y_full.ravel().astype(np.int64)
    y_test = y_test.ravel().astype(np.int64)

    rng = np.random.default_rng(SEED)
    train_idx, val_idx = [], []
    for c in range(N_CLASSES):
        idx = np.flatnonzero(y_full == c)
        rng.shuffle(idx)
        need = TRAIN_PER_CLASS + VAL_PER_CLASS
        if idx.size < need:
            raise RuntimeError(f"class {c}: {idx.size} images, need {need}")
        train_idx.append(idx[:TRAIN_PER_CLASS])
        val_idx.append(idx[TRAIN_PER_CLASS:need])

    train_idx = rng.permutation(np.concatenate(train_idx))
    val_idx = rng.permutation(np.concatenate(val_idx))

    splits = Splits(
        x_train=x_full[train_idx], y_train=y_full[train_idx],
        x_val=x_full[val_idx], y_val=y_full[val_idx],
        x_test=x_test, y_test=y_test,
    )
    np.savez_compressed(SPLITS_CACHE, **asdict(splits))
    return splits


def make_tf_dataset(x: np.ndarray, y: np.ndarray, *, image_size: int,
                    preprocess: Callable, batch_size: int = 32,
                    shuffle: bool = False, augment=None, seed: int = SEED):
    """uint8 32x32 -> resized, preprocessed, batched `tf.data.Dataset`.

    Resizing happens per batch on the accelerator. `augment` is a Keras layer (or
    Sequential of layers) applied *after* resizing and *only* when `shuffle=True`
    (i.e. on the training stream), never on validation or test.
    """
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((x, y.astype(np.int32)))
    if shuffle:
        ds = ds.shuffle(len(x), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)

    def _prep(images, labels):
        images = tf.image.resize(tf.cast(images, tf.float32),
                                 (image_size, image_size), method="bilinear")
        if augment is not None and shuffle:
            images = augment(images, training=True)
        return preprocess(images), labels

    return ds.map(_prep, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if importable) TensorFlow / PyTorch.

    Note: this does not buy bitwise determinism on GPU — cuDNN convolution kernels
    are non-deterministic by default, and forcing determinism costs a large slowdown.
    That residual noise is precisely why every comparative claim is repeated over
    `SEEDS` and reported with a standard deviation.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def configure_gpu() -> str:
    """Enable memory growth (so TF does not pre-allocate the whole T4) and mixed
    precision (roughly 2x throughput on the T4's tensor cores). Returns a label."""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass  # already initialised
    if gpus:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        return tf.config.experimental.get_device_details(gpus[0]).get(
            "device_name", "GPU")
    return "CPU"


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Preferred over the normal approximation: it stays inside [0, 1] and behaves
    near the boundaries, where accuracies of 0.95+ live. On the 10,000-image test
    set the half-width is about +/-0.6 pp at 90% accuracy — which is exactly the
    resolution limit of any claim made in the report.
    """
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (float(centre - half), float(centre + half))


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact McNemar test on two models' predictions over the same test set.

    This — not Wilcoxon over folds — is the correct test here. The two models are
    evaluated on *identical* samples, so their errors are paired; the test looks
    only at the discordant pairs (a right & b wrong, vs a wrong & b right) and
    asks whether that split is fairer than a coin flip. The exact binomial form is
    used because the chi-square approximation misbehaves when discordances are few.
    """
    a_ok = pred_a == y_true
    b_ok = pred_b == y_true
    n01 = int(np.sum(a_ok & ~b_ok))   # a right, b wrong
    n10 = int(np.sum(~a_ok & b_ok))   # a wrong, b right
    n_disc = n01 + n10
    p = 1.0 if n_disc == 0 else float(
        stats.binomtest(min(n01, n10), n_disc, 0.5).pvalue)
    return {
        "n_a_correct_b_wrong": n01,
        "n_a_wrong_b_correct": n10,
        "n_discordant": n_disc,
        "p_value": p,
        "significant_at_05": bool(p < 0.05),
    }


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #

@dataclass
class RunResult:
    """One trained configuration, scored once on the official test set.

    `test_predictions` is carried in the artefact on purpose: it is what makes the
    paired McNemar test, the confusion matrix and the whole report reproducible
    from the committed JSONs alone, with no GPU and no retraining.
    """

    strategy: str
    label: str
    seed: int
    config: dict
    accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    accuracy_ci95: tuple[float, float]
    per_class_accuracy: dict
    confusion_matrix: list
    test_predictions: list
    input_resolution: int
    params_total: int
    params_trainable: int
    epochs_run: int
    best_epoch: int
    train_seconds: float
    framework: str = "tensorflow"
    environment: dict = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        return f"{self.strategy}__{self.label}__seed{self.seed}"

    def save(self) -> Path:
        path = RESULTS_DIR / f"{self.run_id}.json"
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False),
                        encoding="utf-8")
        return path


def score_run(*, strategy: str, label: str, seed: int, config: dict,
              y_true: np.ndarray, y_pred: np.ndarray, input_resolution: int,
              params_total: int = 0, params_trainable: int = 0,
              epochs_run: int = 0, best_epoch: int = 0, train_seconds: float = 0.0,
              framework: str = "tensorflow", device: str = "") -> RunResult:
    """Turn raw test predictions into the full, persisted result record."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    correct = int(np.sum(y_pred == y_true))
    n = int(y_true.size)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))

    per_class = {
        CLASS_NAMES[c]: float(cm[c, c] / cm[c].sum()) if cm[c].sum() else 0.0
        for c in range(N_CLASSES)
    }

    result = RunResult(
        strategy=strategy,
        label=label,
        seed=seed,
        config=config,
        accuracy=correct / n,
        macro_f1=float(f1_score(y_true, y_pred, average="macro")),
        macro_precision=float(precision_score(y_true, y_pred, average="macro",
                                              zero_division=0)),
        macro_recall=float(recall_score(y_true, y_pred, average="macro",
                                        zero_division=0)),
        accuracy_ci95=wilson_ci(correct, n),
        per_class_accuracy=per_class,
        confusion_matrix=cm.tolist(),
        test_predictions=y_pred.astype(int).tolist(),
        input_resolution=input_resolution,
        params_total=int(params_total),
        params_trainable=int(params_trainable),
        epochs_run=int(epochs_run),
        best_epoch=int(best_epoch),
        train_seconds=float(train_seconds),
        framework=framework,
        environment={"device": device, "python": platform.python_version()},
    )
    path = result.save()
    lo, hi = result.accuracy_ci95
    print(f"\n[{result.run_id}]")
    print(f"  accuracy   {result.accuracy:.4f}  (95% CI {lo:.4f}-{hi:.4f})")
    print(f"  macro-F1   {result.macro_f1:.4f}")
    print(f"  trainable  {result.params_trainable:,} params @ {input_resolution}px")
    print(f"  train time {result.train_seconds/60:.1f} min")
    print(f"  saved      {path.parent.name}/{path.name}")
    return result


def load_results(strategy: str | None = None) -> list[dict]:
    """Load every persisted run, optionally filtered by strategy prefix."""
    runs = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        if strategy is None or run["strategy"] == strategy:
            runs.append(run)
    return runs


def aggregate(runs: Iterable[dict]) -> list[dict]:
    """Collapse repeated seeds of the same configuration into mean +/- std.

    A configuration with a single seed reports std = None rather than 0.0: those
    are different statements, and printing 0.0 would imply a stability that was
    never measured.
    """
    by_label: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        by_label.setdefault((run["strategy"], run["label"]), []).append(run)

    rows = []
    for (strategy, label), group in by_label.items():
        accs = np.array([g["accuracy"] for g in group])
        f1s = np.array([g["macro_f1"] for g in group])
        first = group[0]
        rows.append({
            "strategy": strategy,
            "label": label,
            "n_seeds": len(group),
            "accuracy_mean": float(accs.mean()),
            "accuracy_std": float(accs.std(ddof=1)) if len(group) > 1 else None,
            "macro_f1_mean": float(f1s.mean()),
            "macro_f1_std": float(f1s.std(ddof=1)) if len(group) > 1 else None,
            "best_seed_accuracy": float(accs.max()),
            "input_resolution": first["input_resolution"],
            "params_trainable": first["params_trainable"],
            "train_minutes_mean": float(np.mean([g["train_seconds"] for g in group]) / 60),
            "config": first["config"],
            "framework": first["framework"],
        })
    return sorted(rows, key=lambda r: r["accuracy_mean"], reverse=True)


class Timer:
    """Wall-clock timer. Training cost is a reported variable, not an afterthought:
    the best accuracy at 50x the compute is a different answer to the same question."""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start
        return False
