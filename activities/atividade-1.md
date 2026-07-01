# Atividade 1 — Shallow Models: Classification & Regression

> **Course:** Aprendizagem de Máquina · PPGIA / PUC-PR · MSc 2026  
> **Instructor:** Prof. Alceu de Souza Britto Jr. (alceu@ppgia.pucpr.br)  
> **Related modules:** [Module 2 — Experimental Protocols](../modules/02-protocols.md) · [Module 3 — Shallow Techniques](../modules/03-shallow.md)

---

## Overview

This activity applies the shallow supervised learning techniques from Module 3 under the experimental protocol from Module 2. Two tasks are addressed: binary classification (Part A) and regression (Part B). Both use publicly available benchmark datasets and 5-fold cross-validation as the evaluation protocol.

The objective is not only to obtain results, but to develop the capacity to **design a rigorous comparative experiment**, interpret metric differences with statistical awareness, and reason about method selection given dataset characteristics.

---

## Part A — Classification

### Dataset: Breast Cancer Wisconsin

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository |
| Instances | 569 |
| Features | 30 continuous (derived from digitised FNA images) |
| Target | Binary: 0 = malignant, 1 = benign |
| Class distribution | 212 malignant (37.3%), 357 benign (62.7%) |

Features encode geometric and textural properties of cell nuclei: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension — each computed as mean, standard error, and worst (largest) value across the image.

### Classifiers

The following nine classifiers are compared:

| Classifier | Module section | Key hyperparameters to tune |
|---|---|---|
| Decision Tree | [03-shallow § 5](../modules/03-shallow.md#5-tree-based-methods) | Max depth, criterion |
| k-Nearest Neighbours | [03-shallow § 2](../modules/03-shallow.md#2-instance-based-methods) | $k$, distance metric |
| Naive Bayes | [03-shallow § 3](../modules/03-shallow.md#3-probabilistic-methods) | `var_smoothing` (Gaussian NB) |
| SVM | [03-shallow § 4](../modules/03-shallow.md#4-geometric-methods) | $C$, kernel, $\gamma$ |
| MLP | [03-shallow § 6](../modules/03-shallow.md#mlp-as-a-shallow-baseline) | Architecture, $\eta$ |
| Random Forest | [03-shallow § 6](../modules/03-shallow.md#random-forest) | $T$, $m$, depth |
| Bagging | [03-shallow § 6](../modules/03-shallow.md#bagging-bootstrap-aggregating) | $T$, base learner |
| AdaBoost | [03-shallow § 6](../modules/03-shallow.md#adaboost) | $T$, learning rate |
| XGBoost | [03-shallow § 6](../modules/03-shallow.md#xgboost-extreme-gradient-boosting) | $\eta$, depth, $\lambda$ |

> A **Logistic Regression** baseline ([03-shallow § 1](../modules/03-shallow.md#1-linear-models)) is included in the analysis as the canonical linear reference every other classifier should be measured against.

### Experimental Protocol

- **Validation:** Stratified 5-fold cross-validation (stratification preserves class ratio per fold)
- **Metrics:** Accuracy, F1 score (macro), Precision (macro), Recall (macro)
- **Comparison:** Per-fold metric values enable statistical comparison across classifiers

### Questions

**A.1** — Which classifier achieves the best mean accuracy across folds? Is the difference statistically significant relative to the second-best? Apply an appropriate statistical test (see [Module 2 §4](../modules/02-protocols.md#4-comparing-classifiers-statistically)).

**A.2** — Analyse the confusion matrices for the two best classifiers. Which class (malignant / benign) is more frequently misclassified, and what are the implications for clinical deployment? Consider the relative costs of false positives and false negatives.

**A.3** — Compare the trade-off between precision and recall across classifiers. Which methods exhibit high precision at the cost of recall, and which exhibit the inverse? What does this reveal about their decision boundaries?

**A.4** — Ensemble methods (Random Forest, Bagging, AdaBoost, XGBoost) are expected to outperform their base learners. Does the experimental evidence support this? Analyse any exceptions and hypothesise why they occur.

---

## Part B — Regression

### Dataset: Diabetes

| Property | Value |
|---|---|
| Source | scikit-learn built-in (`sklearn.datasets.load_diabetes`) |
| Instances | 442 |
| Features | 10 numeric (age, sex, BMI, blood pressure, 6 serum measurements) — pre-normalised |
| Target | Continuous: disease progression measure one year after baseline (range 25–346) |

### Regressors

The following seven regressors are compared:

| Regressor | Module section | Key hyperparameters to tune |
|---|---|---|
| Decision Tree | [03-shallow § 5](../modules/03-shallow.md#decision-tree) | Max depth, min samples leaf |
| k-Nearest Neighbours | [03-shallow § 2](../modules/03-shallow.md#k-nearest-neighbours-k-nn) | $k$, distance metric |
| SVR | [03-shallow § 4](../modules/03-shallow.md#support-vector-machine-svm) | $C$, $\varepsilon$, kernel |
| MLP | [03-shallow § 6](../modules/03-shallow.md#mlp-as-a-shallow-baseline) | Architecture, $\eta$ |
| Random Forest | [03-shallow § 6](../modules/03-shallow.md#random-forest) | $T$, $m$, depth |
| Bagging | [03-shallow § 6](../modules/03-shallow.md#bagging-bootstrap-aggregating) | $T$, base learner |
| XGBoost | [03-shallow § 6](../modules/03-shallow.md#xgboost-extreme-gradient-boosting) | $\eta$, depth, $\lambda$ |

> A **Ridge (regularised linear) regression** baseline ([03-shallow § 1](../modules/03-shallow.md#1-linear-models)) is included as the canonical linear reference for the regression task.

### Experimental Protocol

- **Validation:** 5-fold cross-validation
- **Metrics:** R² (coefficient of determination), MAE (mean absolute error)
- **Note:** Because the target range spans ~320 units, report MAE both in absolute terms and as a percentage of the target range for interpretability.

### Questions

**B.1** — Which regressor achieves the highest mean R² and lowest mean MAE across folds? Are the two rankings consistent (i.e., does the best R² model also yield the lowest MAE)? If not, explain what this discrepancy reveals about the error distribution of each model.

---

## Implementation Notes

### Recommended Stack

```python
# Core
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Classifiers
from sklearn.linear_model import LogisticRegression   # linear baseline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

# Regressors
from sklearn.linear_model import Ridge                 # linear baseline
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from xgboost import XGBRegressor

# Metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import r2_score, mean_absolute_error
```

### Preprocessing Considerations

- Feature standardisation (`StandardScaler`) is **required** before Logistic Regression, Ridge, k-NN, SVM/SVR, and MLP. Tree-based methods and Gaussian Naive Bayes are invariant to monotonic feature scaling.
- Gaussian Naive Bayes is parameter-light: its only smoothing knob is `var_smoothing` (a variance floor for numerical stability), **not** the Laplace/additive $\alpha$ used by Multinomial/Bernoulli NB.
- Use `Pipeline(steps=[('scaler', StandardScaler()), ('clf', classifier)])` to prevent data leakage — the scaler must be fit only on training folds.

### Reproducibility

Set a fixed random seed for all stochastic components:

```python
SEED = 42
```

Pass `random_state=SEED` to all classifiers/regressors that accept it, and to `KFold`/`StratifiedKFold`.

---

## Results

> Generated by [`atividade-1.py`](atividade-1.py) — `SEED = 42`, stratified (Part A) / shuffled (Part B) 5-fold CV, scale-sensitive learners wrapped in a fold-local `StandardScaler` to prevent leakage. All values are means over the five folds. Re-run with `python atividade-1.py` to reproduce exactly.

### Part A — Classification Results

| Classifier | Accuracy | F1 (macro) | Precision (macro) | Recall (macro) |
|---|---|---|---|---|
| Logistic Regression *(linear baseline)* | 0.9737 | 0.9714 | 0.9773 | 0.9676 |
| Decision Tree | 0.9104 | 0.9028 | 0.9080 | 0.9000 |
| k-NN | 0.9631 | 0.9599 | 0.9664 | 0.9553 |
| Naive Bayes | 0.9385 | 0.9331 | 0.9428 | 0.9280 |
| **SVM** | **0.9772** | 0.9754 | 0.9783 | 0.9732 |
| **MLP** | **0.9772** | 0.9753 | 0.9797 | 0.9723 |
| Random Forest | 0.9561 | 0.9529 | 0.9557 | 0.9527 |
| Bagging | 0.9543 | 0.9512 | 0.9514 | 0.9520 |
| AdaBoost | 0.9525 | 0.9487 | 0.9523 | 0.9469 |
| XGBoost | 0.9631 | 0.9604 | 0.9635 | 0.9592 |

### Part B — Regression Results

| Regressor | R² | MAE | MAE (% of target range) |
|---|---|---|---|
| **Linear (Ridge)** *(linear baseline)* | **0.4791** | 44.24 | 13.78% |
| Decision Tree | −0.1325 | 63.64 | 19.82% |
| k-NN | 0.3912 | 46.76 | 14.57% |
| SVR | 0.1492 | 58.89 | 18.35% |
| **MLP** | 0.4686 | **44.05** | 13.72% |
| Random Forest | 0.4294 | 46.69 | 14.55% |
| Bagging | 0.3761 | 48.81 | 15.21% |
| XGBoost | 0.3290 | 50.20 | 15.64% |

---

## Answers

### Part A

**A.1 — Best classifier and significance.**
SVM and MLP tie at the top with mean accuracy **0.9772**; SVM edges MLP on F1/recall, MLP on precision. Taking SVM and MLP as the top two, a **Wilcoxon signed-rank test** on the five per-fold accuracies gives $W = 4.0$, $p = 0.875$ — **not significant** at $\alpha = 0.05$. The conclusion is that the two are statistically indistinguishable on this dataset; reporting either as "the winner" would over-interpret a difference smaller than fold-to-fold noise. Notably, the **Logistic Regression baseline (0.9737)** is within ~0.4 pp of the top and *outperforms every tree-based ensemble* — a textbook reminder that Breast Cancer Wisconsin is close to linearly separable, so capacity beyond a linear boundary buys almost nothing.

**A.2 — Confusion-matrix analysis of the two best (pooled out-of-fold predictions).**

| | SVM | MLP |
|---|---|---|
| Malignant (0) → predicted benign — **false negatives** | 9 / 212 | 10 / 212 |
| Benign (1) → predicted malignant — false positives | 4 / 357 | 3 / 357 |

For **both** models the error rate on the **malignant** class (≈4.3–4.7%) is higher than on the benign class (≈0.8–1.1%): the **malignant class is misclassified more often**. This is the clinically dangerous direction — a missed malignancy (false negative) defers treatment, whereas a false positive triggers a follow-up biopsy. A screening deployment should therefore **shift the decision threshold to raise recall on the malignant class**, accepting more false positives, rather than optimising raw accuracy.

**A.3 — Precision–recall trade-off.**
Across every classifier, macro-precision exceeds macro-recall (e.g. MLP 0.9797 vs 0.9723; Logistic 0.9773 vs 0.9676), i.e. models are slightly **conservative about declaring the malignant class** — consistent with A.2. Decision Tree shows the narrowest precision–recall gap but the lowest level of both (high variance, axis-aligned boundary overfitting). The probabilistic Naive Bayes shows higher precision than recall, reflecting its conditional-independence bias on the correlated nucleus features. The geometric/margin models (SVM, MLP, Logistic) dominate because the decision surface separating the two nucleus-feature clusters is close to linear.

**A.4 — Do ensembles beat their base learners?**
Partly. Random Forest (0.9561), Bagging (0.9543) and AdaBoost (0.9525) **clearly beat a single Decision Tree (0.9104)** — bagging's variance reduction is exactly what a high-variance tree needs. But the headline expectation *fails*: the tree ensembles **do not beat the non-tree models** — SVM, MLP, Logistic Regression and even plain k-NN all outrank them. The reason is that ensembling only repairs the *inductive bias of its base learner*; axis-aligned splits remain a poor fit for a near-linear boundary, so a forest of trees cannot overcome the basic mismatch. XGBoost (0.9631) underperforms the linear baseline here for the same reason, compounded by limited data (569 samples) for a high-capacity booster. **Lesson: ensemble gains are relative to the base learner, not absolute; the right inductive bias beats the fancier algorithm.**

### Part B

**B.1 — Best R², best MAE, and ranking consistency.**
The rankings are **not consistent**. **Ridge** achieves the highest mean **R² (0.4791)**, but **MLP** achieves the lowest mean **MAE (44.05)**; Ridge's MAE (44.24) is a close second. R² and MAE measure different things: R² is driven by **squared** error and so is dominated by the largest residuals, while MAE weights every error linearly. A model can explain marginally more total variance (higher R²) yet make slightly larger *typical* errors (higher MAE) if it trades many small errors for fewer large ones — which is what separates Ridge from MLP here. The practical reading: on this low-signal dataset (best R² ≈ 0.48, MAE ≈ 14% of range for several models) a **regularised linear model is the right default** — Decision Tree's negative R² (−0.13) shows a single unregularised tree is *worse than predicting the mean*, and the high-capacity learners (MLP, RF, XGBoost) earn no meaningful advantage over Ridge. Always report R² **and** MAE: each alone would crown a different model.

---

*[← README](../README.md) · [Module 2 — Experimental Protocols](../modules/02-protocols.md) · [Module 3 — Shallow Techniques](../modules/03-shallow.md)*
