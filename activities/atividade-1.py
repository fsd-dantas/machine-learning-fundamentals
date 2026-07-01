"""
Atividade 1 — Shallow Models: Classification & Regression
Aprendizagem de Máquina · PPGIA/PUC-PR · MSc 2026

Reproducible experiment script for the comparative study described in
`atividade-1.md`. Running this file regenerates every number in the Results
tables, the per-fold matrices used for statistical testing, and the answers
to questions A.1-A.4 and B.1.

Usage:
    python atividade-1.py

Dependencies: numpy, pandas, scikit-learn, scipy, xgboost.
Determinism: a single SEED governs all stochastic components; leakage is
prevented by fitting the StandardScaler inside each CV fold via Pipeline.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    BaggingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")  # silence convergence chatter for clean tables

SEED = 42
N_SPLITS = 5


def needs_scaling(name: str) -> bool:
    """Distance/gradient-based learners require standardisation; trees/NB do not."""
    return name in {"Logistic Regression", "k-NN", "SVM", "SVR", "MLP",
                    "Linear (Ridge)"}


def make_pipeline(name: str, estimator):
    """Wrap scale-sensitive estimators with a fold-local StandardScaler."""
    if needs_scaling(name):
        return Pipeline([("scaler", StandardScaler()), ("est", estimator)])
    return estimator


# --------------------------------------------------------------------------- #
# Part A — Classification (Breast Cancer Wisconsin)
# --------------------------------------------------------------------------- #
def part_a():
    data = load_breast_cancer()
    X, y = data.data, data.target

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "k-NN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(random_state=SEED),
        "MLP": MLPClassifier(max_iter=1000, random_state=SEED),
        "Random Forest": RandomForestClassifier(random_state=SEED),
        "Bagging": BaggingClassifier(random_state=SEED),
        "AdaBoost": AdaBoostClassifier(random_state=SEED),
        "XGBoost": XGBClassifier(
            eval_metric="logloss", random_state=SEED, verbosity=0
        ),
    }

    scoring = {
        "accuracy": "accuracy",
        "f1": "f1_macro",
        "precision": "precision_macro",
        "recall": "recall_macro",
    }

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    rows, per_fold_acc = [], {}

    for name, clf in classifiers.items():
        res = cross_validate(
            make_pipeline(name, clf), X, y, cv=cv, scoring=scoring
        )
        per_fold_acc[name] = res["test_accuracy"]
        rows.append(
            {
                "Classifier": name,
                "Accuracy": res["test_accuracy"].mean(),
                "F1": res["test_f1"].mean(),
                "Precision": res["test_precision"].mean(),
                "Recall": res["test_recall"].mean(),
            }
        )

    df = pd.DataFrame(rows).set_index("Classifier")
    print("\n=== Part A — Classification Results (mean over 5 folds) ===\n")
    print(df.round(4).to_string())

    ranking = df["Accuracy"].sort_values(ascending=False)
    best, second = ranking.index[0], ranking.index[1]
    stat, p = wilcoxon(per_fold_acc[best], per_fold_acc[second])
    print(f"\n[A.1] Best: {best} ({ranking.iloc[0]:.4f}) | "
          f"2nd: {second} ({ranking.iloc[1]:.4f})")
    print(f"[A.1] Wilcoxon signed-rank (per-fold accuracy, best vs 2nd): "
          f"W={stat:.3f}, p={p:.4f} -> "
          f"{'significant' if p < 0.05 else 'NOT significant'} at alpha=0.05")
    return df


# --------------------------------------------------------------------------- #
# Part B — Regression (Diabetes)
# --------------------------------------------------------------------------- #
def part_b():
    data = load_diabetes()
    X, y = data.data, data.target
    target_range = y.max() - y.min()

    regressors = {
        "Linear (Ridge)": Ridge(random_state=SEED),
        "Decision Tree": DecisionTreeRegressor(random_state=SEED),
        "k-NN": KNeighborsRegressor(),
        "SVR": SVR(),
        "MLP": MLPRegressor(max_iter=2000, random_state=SEED),
        "Random Forest": RandomForestRegressor(random_state=SEED),
        "Bagging": BaggingRegressor(random_state=SEED),
        "XGBoost": XGBRegressor(random_state=SEED, verbosity=0),
    }

    scoring = {"r2": "r2", "mae": "neg_mean_absolute_error"}
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    rows = []

    for name, reg in regressors.items():
        res = cross_validate(
            make_pipeline(name, reg), X, y, cv=cv, scoring=scoring
        )
        mae = -res["test_mae"].mean()
        rows.append(
            {
                "Regressor": name,
                "R2": res["test_r2"].mean(),
                "MAE": mae,
                "MAE_%range": 100 * mae / target_range,
            }
        )

    df = pd.DataFrame(rows).set_index("Regressor")
    print("\n=== Part B — Regression Results (mean over 5 folds) ===\n")
    print(df.round(4).to_string())

    best_r2 = df["R2"].idxmax()
    best_mae = df["MAE"].idxmin()
    print(f"\n[B.1] Highest R2: {best_r2} ({df['R2'].max():.4f})")
    print(f"[B.1] Lowest MAE: {best_mae} ({df['MAE'].min():.4f})")
    print(f"[B.1] Rankings consistent: {best_r2 == best_mae}")
    return df


if __name__ == "__main__":
    print(f"SEED={SEED} | folds={N_SPLITS} | "
          f"numpy {np.__version__} | pandas {pd.__version__}")
    part_a()
    part_b()
