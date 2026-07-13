"""
Baselines — majority class and TF-IDF + linear SVM.

Neither is asked for by the assignment, and the report is much weaker without them.

**Majority class** is not a model, it is the floor: 57.7% (binary) and 24.8%
(multiclass). Quote an accuracy without it and a reader cannot tell learning from
arithmetic — a binary classifier at 62% has captured almost nothing, yet reads as a
passing grade.

**TF-IDF + linear SVM** is the ceiling that deep learning has to clear to justify
itself. On ~1,450 short news headlines, a bag of character and word n-grams fitted by a
convex solver is a genuinely strong competitor: an LSTM must learn its embeddings from
scratch on that same tiny corpus, and 1,450 examples is not enough to learn what a word
means. This mirrors Atividade 1, where a logistic-regression baseline beat every tree
ensemble on the same principle — the right inductive bias beats the fancier algorithm.

Character n-grams (3-5) are included alongside word n-grams because Portuguese is
morphologically rich: `mata`, `matou`, `matando` share a stem that a word-level model
treats as three unrelated tokens.

Fitted on the training split only — the same data the networks see — so the comparison
is not tilted by giving the baseline the validation slice as a bonus.

Usage:
    python m0_baselines.py --task binary
    python m0_baselines.py --task multiclass --seed 7
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, make_union
from sklearn.svm import LinearSVC

import common


def build_tfidf_svm(seed: int) -> Pipeline:
    features = make_union(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                        sublinear_tf=True, lowercase=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                        sublinear_tf=True, lowercase=True),
    )
    return Pipeline([
        ("features", features),
        ("clf", LinearSVC(C=1.0, class_weight="balanced", random_state=seed,
                          max_iter=5000)),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="binary", choices=common.TASKS)
    parser.add_argument("--seed", type=int, default=common.SEED)
    args = parser.parse_args()

    common.set_seed(args.seed)
    corpus = common.load_corpus(args.task)
    splits = common.make_splits(corpus, args.seed)
    print(f"{splits.summary()}\n")

    for name, model in (("majority", DummyClassifier(strategy="most_frequent")),
                        ("tfidf_svm", build_tfidf_svm(args.seed))):
        with common.Timer() as timer:
            model.fit(splits.x_train, splits.y_train)
        y_pred = model.predict(splits.x_test)

        n_features = 0
        if name == "tfidf_svm":
            n_features = int(sum(
                len(v.vocabulary_) for _, v in model.named_steps[
                    "features"].transformer_list))

        common.score_run(
            task=args.task, model=name, seed=args.seed,
            config={"vectorizer": "word(1,2) + char_wb(3,5) TF-IDF",
                    "classifier": "LinearSVC(class_weight=balanced)",
                    "n_features": n_features} if name == "tfidf_svm"
            else {"strategy": "most_frequent"},
            corpus=corpus, y_true=splits.y_test, y_pred=y_pred,
            params_trainable=n_features, epochs_run=0,
            train_seconds=timer.seconds, framework="sklearn",
        )


if __name__ == "__main__":
    main()
