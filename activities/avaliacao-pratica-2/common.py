"""
Avaliação Prática 2 — text classification (LSTM vs Transformer): shared protocol.
Aprendizagem de Máquina · PPGIA/PUC-PR · MSc 2026

Every model (m0..m2) imports this module. It owns the corpus, the label maps, the
holdout, the evaluation and the statistics, so that a Keras LSTM and a PyTorch BERT
remain comparable to each other and to the classical baseline.

The corpus needs real work before it can be trained on, and each step below is a
finding in its own right rather than housekeeping:

1. **Encoding.** The CSVs are latin-1, not UTF-8, and their delimiter is `;` with a
   junk quoted column in the middle (`texto;";";classe`). Read as UTF-8 they raise;
   read carelessly they yield mojibake, and an LSTM would then learn a vocabulary of
   corrupted tokens.

2. **Duplicates.** Concatenating the two parts yields 2,732 rows but only 2,440
   distinct texts: **292 duplicates (10.7%)**. Splitting before de-duplicating would
   put a text in train and its copy in test — the model would be scored on material it
   memorised. De-duplication happens here, before any split, and it is the single most
   consequential line in this file.

3. **Conflicting labels.** Four texts appear with *different* emotion labels. Notably,
   all four conflicts are negative-vs-negative (raiva/desgosto, desgosto/medo,
   tristeza/medo): the annotators disagree about *which* negative emotion, never about
   the valence. So the binary task is clean, while the multiclass task carries
   irreducible label noise — which is exactly where its confusion matrix will bleed.
   The four are dropped, uniformly, from every task.

4. **The floor.** After de-duplication the majority class is 57.7% (binary) and 24.8%
   (multiclass). Every accuracy in the report is quoted against those numbers, because
   a 62% binary classifier is not "good", it is barely better than answering
   "negativo" to everything.

Statistical helpers (Wilson interval, exact McNemar, seeded runs, result records) are
deliberately duplicated from `avaliacao-pratica-1/common.py` rather than shared through
an import: that activity's code is frozen while its GPU sweep runs. Unifying the two
into one protocol package is the obvious follow-up once both are delivered.
"""

from __future__ import annotations

import json
import platform
import random
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# Protocol constants
# --------------------------------------------------------------------------- #

SEED = 42
SEEDS: tuple[int, ...] = (42, 7, 2024)
"""The assignment prescribes a single 70/30 holdout. On 2,436 texts that is a test set
of ~730, where the binomial standard error at 80% accuracy is ~1.5 pp — wide enough
that one split can rank two models by luck alone. The holdout is honoured exactly as
specified, then repeated over three seeds so the report can say how much of a gap is
signal."""

TEST_SIZE = 0.30      # as prescribed
VAL_SIZE = 0.15       # carved out of the 70% train, for early stopping only

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"
CSV_FILES = ("g1_v1_ws.csv", "g1_v2_ws.csv")

for _d in (DATA_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

BERT_MODEL = "neuralmind/bert-base-portuguese-cased"

# The instructor's mapping, transcribed verbatim from the lecture notebook.
# `neutro -> positivo` and `surpresa -> positivo` are semantically strained (neutral is
# not positive; surprise has no fixed valence), so `binary_valence` re-runs the task
# without them as a sensitivity check. If the LSTM-vs-BERT conclusion holds under both,
# it does not depend on a debatable labelling decision.
VALENCE_MAP = {
    "neutro": "positivo", "alegria": "positivo", "surpresa": "positivo",
    "medo": "negativo", "raiva": "negativo", "desgosto": "negativo",
    "tristeza": "negativo",
}

TASKS = ("binary", "multiclass", "binary_valence")


@dataclass
class Corpus:
    texts: np.ndarray
    labels: np.ndarray      # integer-encoded
    class_names: list[str]
    task: str

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def majority_rate(self) -> float:
        """The floor. Any model that does not clear this has learned nothing."""
        return float(np.bincount(self.labels).max() / len(self.labels))


def _read_raw() -> pd.DataFrame:
    frames = []
    for name in CSV_FILES:
        path = DATA_DIR / name
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. The corpus is instructor-provided and is not "
                f"committed to this repository — upload {', '.join(CSV_FILES)} into "
                f"{DATA_DIR} (the Colab notebook has a cell for it)."
            )
        # latin-1, not UTF-8; ';' delimiter with a quoted junk column in the middle.
        frame = pd.read_csv(path, sep=";", encoding="latin-1")[["texto", "classe"]]
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df["texto"] = df["texto"].astype(str).str.strip()
    df["classe"] = df["classe"].str.strip().str.lower()
    return df


def load_corpus(task: str = "binary", *, verbose: bool = True) -> Corpus:
    """Concatenate, de-duplicate, drop conflicts, encode labels — in that order."""
    if task not in TASKS:
        raise ValueError(f"unknown task: {task}")

    df = _read_raw()
    n_raw = len(df)

    # Normalised key: same text differing only in casing or whitespace is the same text,
    # and would leak just as effectively across the split.
    df["key"] = (df["texto"].str.lower()
                 .str.replace(r"\s+", " ", regex=True).str.strip())

    conflicting = df.groupby("key")["classe"].nunique()
    conflicting = set(conflicting[conflicting > 1].index)
    df = df[~df["key"].isin(conflicting)]

    df = df.drop_duplicates("key", keep="first").reset_index(drop=True)

    if task == "multiclass":
        labels = df["classe"]
    else:
        labels = df["classe"].map(VALENCE_MAP)
        if task == "binary_valence":
            # neither positive nor valence-bearing: dropped rather than forced
            keep = ~df["classe"].isin({"neutro", "surpresa"})
            df, labels = df[keep], labels[keep]

    class_names = sorted(labels.unique())
    encoded = labels.map({name: i for i, name in enumerate(class_names)}).to_numpy()
    corpus = Corpus(texts=df["texto"].to_numpy(), labels=encoded.astype(np.int64),
                    class_names=class_names, task=task)

    if verbose:
        print(f"[{task}] {n_raw} rows -> {len(corpus.texts)} texts "
              f"({n_raw - len(df) - len(conflicting)} duplicates, "
              f"{len(conflicting)} label conflicts dropped)")
        print(f"[{task}] classes: " + " · ".join(
            f"{name} {int(np.sum(corpus.labels == i))}"
            for i, name in enumerate(class_names)))
        print(f"[{task}] majority-class floor: {corpus.majority_rate():.4f}")
    return corpus


@dataclass
class Splits:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

    def summary(self) -> str:
        return (f"train {len(self.x_train)} · val {len(self.x_val)} · "
                f"test {len(self.x_test)}")


def make_splits(corpus: Corpus, seed: int = SEED) -> Splits:
    """The prescribed 70/30 holdout, stratified, with a validation carve-out.

    Two departures from the lecture notebook, both necessary:

    * **Stratified.** With `raiva` down to 196 examples, an unstratified split can hand
      the test set a class proportion the model never saw in training, and the macro-F1
      then measures the split rather than the model.
    * **A real validation set.** The notebook passes its 30% holdout as
      `validation_data` and *also* reports accuracy on it. That is model selection on
      the test set: the reported number is optimistic by construction. Here early
      stopping watches a validation slice carved out of the training 70%, and the test
      set is read exactly once.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        corpus.texts, corpus.labels, test_size=TEST_SIZE, random_state=seed,
        stratify=corpus.labels)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=VAL_SIZE, random_state=seed, stratify=y_train)
    return Splits(x_train, y_train, x_val, y_val, x_test, y_test)


def set_seed(seed: int) -> None:
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


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval — stays inside [0,1] and behaves near the boundaries."""
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (float(centre - half), float(centre + half))


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact McNemar on paired predictions over the same test set.

    The LSTM and BERT see identical test texts, so their errors are paired and only the
    discordant pairs carry information. On a ~730-text test set this matters: a 3 pp gap
    is roughly 22 texts, and whether that is signal or luck is not something eyeballing
    two accuracies can answer.
    """
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    n01 = int(np.sum(a_ok & ~b_ok))
    n10 = int(np.sum(~a_ok & b_ok))
    n_disc = n01 + n10
    p = 1.0 if n_disc == 0 else float(
        stats.binomtest(min(n01, n10), n_disc, 0.5).pvalue)
    return {"n_a_correct_b_wrong": n01, "n_a_wrong_b_correct": n10,
            "n_discordant": n_disc, "p_value": p,
            "significant_at_05": bool(p < 0.05)}


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #

@dataclass
class RunResult:
    task: str
    model: str
    label: str
    seed: int
    config: dict
    accuracy: float
    macro_f1: float
    weighted_f1: float
    macro_precision: float
    macro_recall: float
    accuracy_ci95: tuple[float, float]
    majority_floor: float
    class_names: list[str]
    per_class_f1: dict
    confusion_matrix: list
    test_predictions: list
    params_trainable: int
    epochs_run: int
    train_seconds: float
    framework: str = "tensorflow"
    environment: dict = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        return f"{self.task}__{self.model}__seed{self.seed}"

    def save(self) -> Path:
        path = RESULTS_DIR / f"{self.run_id}.json"
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False),
                        encoding="utf-8")
        return path


def score_run(*, task: str, model: str, seed: int, config: dict, corpus: Corpus,
              y_true: np.ndarray, y_pred: np.ndarray, label: str | None = None,
              params_trainable: int = 0, epochs_run: int = 0,
              train_seconds: float = 0.0, framework: str = "tensorflow",
              device: str = "") -> RunResult:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    correct = int(np.sum(y_pred == y_true))
    n = int(y_true.size)
    names = corpus.class_names
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    per_class = dict(zip(names, [float(v) for v in f1_score(
        y_true, y_pred, average=None, labels=list(range(len(names))),
        zero_division=0)]))

    result = RunResult(
        task=task, model=model, label=label or model, seed=seed, config=config,
        accuracy=correct / n,
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        macro_precision=float(precision_score(y_true, y_pred, average="macro",
                                              zero_division=0)),
        macro_recall=float(recall_score(y_true, y_pred, average="macro",
                                        zero_division=0)),
        accuracy_ci95=wilson_ci(correct, n),
        majority_floor=corpus.majority_rate(),
        class_names=names,
        per_class_f1=per_class,
        confusion_matrix=cm.tolist(),
        test_predictions=y_pred.astype(int).tolist(),
        params_trainable=int(params_trainable),
        epochs_run=int(epochs_run),
        train_seconds=float(train_seconds),
        framework=framework,
        environment={"device": device, "python": platform.python_version()},
    )
    path = result.save()
    lo, hi = result.accuracy_ci95
    lift = (result.accuracy - result.majority_floor) * 100
    print(f"\n[{result.run_id}]")
    print(f"  accuracy   {result.accuracy:.4f}  (95% CI {lo:.4f}-{hi:.4f})")
    print(f"  macro-F1   {result.macro_f1:.4f}")
    print(f"  vs floor   {lift:+.1f} pp over majority class "
          f"({result.majority_floor:.4f})")
    print(f"  train      {result.train_seconds:.0f}s · {result.params_trainable:,} params")
    print(f"  saved      {path.parent.name}/{path.name}")
    return result


def load_results(task: str | None = None) -> list[dict]:
    runs = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        if task is None or run["task"] == task:
            runs.append(run)
    return runs


def aggregate(runs: Iterable[dict]) -> list[dict]:
    """Collapse seeds of the same (task, model) into mean ± std.

    A single-seed configuration reports std = None, never 0.0: the second is a claim of
    stability that was never measured.
    """
    grouped: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        grouped.setdefault((run["task"], run["model"]), []).append(run)

    rows = []
    for (task, model), group in grouped.items():
        accs = np.array([g["accuracy"] for g in group])
        f1s = np.array([g["macro_f1"] for g in group])
        first = group[0]
        rows.append({
            "task": task, "model": model, "n_seeds": len(group),
            "accuracy_mean": float(accs.mean()),
            "accuracy_std": float(accs.std(ddof=1)) if len(group) > 1 else None,
            "macro_f1_mean": float(f1s.mean()),
            "macro_f1_std": float(f1s.std(ddof=1)) if len(group) > 1 else None,
            "majority_floor": first["majority_floor"],
            "params_trainable": first["params_trainable"],
            "train_seconds_mean": float(np.mean([g["train_seconds"] for g in group])),
            "framework": first["framework"],
        })
    return sorted(rows, key=lambda r: (r["task"], -r["accuracy_mean"]))


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start
        return False
