# Module 2 — Experimental Protocols

> **Colour:** ![#d29922](https://placehold.co/12x12/d29922/d29922.png) Amber  
> **Prerequisites:** [Module 1 — Foundations](01-foundations.md) (probability, expectation, hypothesis testing)  
> **Connects to:** [Module 3 — Shallow Techniques](03-shallow.md) · [Module 4 — Descriptive Models](04-descriptive.md) · [Module 5 — Deep Techniques](05-deep.md)

---

## Overview

Experimental Protocols is the methodology layer of ML research. It answers the question: **how do we measure whether a model works, and whether the difference between two models is real?** Without a rigorous experimental protocol, results are not reproducible, comparisons are not valid, and conclusions drawn from them are scientifically unsound.

This module covers the design of evaluation experiments — how to split data, how to estimate generalisation error without bias, which metrics to compute for which task types, and how to make statistically defensible claims when comparing methods.

---

## How to Use This Module (Exam Prep)

**After this module you should be able to:**
- Explain **why** you never evaluate on the training set.
- Draw and describe **k-fold cross-validation**, including stratification.
- Build a **confusion matrix** and compute accuracy, precision, recall, F1, specificity from raw counts.
- Choose the right metric for the task (and explain when accuracy lies).
- Say which **statistical test** to use to compare methods, and why.

**⭐ High-yield for exams:** confusion-matrix metrics (compute by hand) · precision vs. recall trade-off · how k-fold CV works · when accuracy is misleading · R² can be negative.

**If you only read one thing:** §3 Evaluation Metrics — the confusion-matrix calculations are the single most commonly examined skill in the course.

**Suggested time:** ~75 min reading + ~45 min on the [Worked Examples](#worked-examples) and [Self-Check](#self-check).

---

## 1. The Generalisation Problem

A model trained on data $\mathcal{D}_\text{train}$ must perform well on unseen data drawn from the same distribution. **Generalisation error** is the expected loss on the true data-generating distribution $p(\mathbf{x}, y)$:

$$\mathcal{E}_\text{gen} = \mathbb{E}_{(\mathbf{x},y) \sim p}[\mathcal{L}(f(\mathbf{x}), y)]$$

This is estimated empirically using held-out data. The choice of estimation procedure directly affects how much the estimate can be trusted.

**Common pitfall:** Evaluating on the training set. The training error is a biased (optimistic) estimator of generalisation error — the bias grows with model capacity.

---

## 2. Data Partitioning Strategies

### Holdout

The dataset is split once into train and test sets (typically 70/30 or 80/20). Simple but high variance — the result depends heavily on which samples land in which split, especially for small datasets.

### k-Fold Cross-Validation

> **💡 Intuition:** A single train/test split wastes data and depends on luck — you might get an easy test set. k-fold CV gives **every sample a turn at being tested** while keeping training and testing separate, then averages. You trade compute (train $k$ times) for a more trustworthy, lower-variance estimate.
>
> ```
> 5-Fold CV  (■ = test fold, □ = train folds)
>   Round 1:  ■ □ □ □ □   → score₁
>   Round 2:  □ ■ □ □ □   → score₂
>   Round 3:  □ □ ■ □ □   → score₃
>   Round 4:  □ □ □ ■ □   → score₄
>   Round 5:  □ □ □ □ ■   → score₅
>                              CV estimate = mean(score₁..score₅)
> ```

The dataset is partitioned into $k$ equally-sized folds. The model is trained $k$ times, each time using $k-1$ folds for training and 1 fold for evaluation. The final estimate is the mean over folds:

$$\hat{\mathcal{E}}_\text{CV} = \frac{1}{k} \sum_{i=1}^k \mathcal{E}_i$$

**Standard choice:** $k = 5$ or $k = 10$. For Atividade 1, $k = 5$ is specified.

**Bias–variance of CV:** Lower $k$ → higher bias, lower variance. Higher $k$ → lower bias, higher variance. Leave-one-out ($k = n$) is the lowest-bias estimator but computationally expensive and high-variance.

### Stratified k-Fold

Folds are constructed such that each fold preserves the class distribution of the full dataset. **Required** for imbalanced datasets — otherwise some folds may contain no minority-class samples.

### Repeated k-Fold

k-fold repeated $r$ times with different random splits. Reduces variance of the CV estimate at the cost of $r \times k$ training runs. Common in rigorous benchmarking.

---

## 3. Evaluation Metrics

### Classification

For a binary classifier with predictions $\hat{y} \in \{0, 1\}$ and ground truth $y$:

| Metric | Formula | Interpretation |
|---|---|---|
| Accuracy | $\frac{TP + TN}{n}$ | Fraction correctly classified; misleading under class imbalance |
| Precision | $\frac{TP}{TP + FP}$ | Of predicted positives, how many are truly positive |
| Recall (Sensitivity) | $\frac{TP}{TP + FN}$ | Of true positives, how many were retrieved |
| F1 Score | $\frac{2 \cdot \text{Prec} \cdot \text{Rec}}{\text{Prec} + \text{Rec}}$ | Harmonic mean; balances precision and recall |
| Specificity | $\frac{TN}{TN + FP}$ | True negative rate |

The **confusion matrix** is the primary diagnostic tool — aggregate metrics can mask systematic errors on specific classes.

> **💡 Intuition:** Every prediction lands in one of four boxes. **TP/TN** = correct. **FP** = false alarm (predicted positive, wasn't). **FN** = miss (predicted negative, was positive). **Precision** asks *"when I shout positive, am I right?"*; **recall** asks *"of all real positives, how many did I catch?"* A spam filter wants high precision (don't trash real mail); a cancer screen wants high recall (don't miss a tumour).

> **📝 Worked Example — compute every metric from a confusion matrix**
> A classifier on 100 patients (positive = "has disease") produces:
>
> ```
>                  Predicted +    Predicted −
>   Actual +          TP = 40       FN = 10
>   Actual −          FP =  5       TN = 45
> ```
>
> - **Accuracy** $= \frac{TP+TN}{n} = \frac{40+45}{100} = \mathbf{0.85}$
> - **Precision** $= \frac{TP}{TP+FP} = \frac{40}{45} \approx \mathbf{0.889}$ — of those flagged, 89% truly had it
> - **Recall** $= \frac{TP}{TP+FN} = \frac{40}{50} = \mathbf{0.80}$ — caught 80% of real cases (missed 10)
> - **Specificity** $= \frac{TN}{TN+FP} = \frac{45}{50} = \mathbf{0.90}$
> - **F1** $= \frac{2 \cdot 0.889 \cdot 0.80}{0.889 + 0.80} \approx \mathbf{0.842}$ — harmonic mean of precision & recall
>
> **Watch the trade-off:** lowering the decision threshold would catch more of the 10 misses (↑recall) but raise false alarms (↓precision). F1 balances the two.

**ROC and AUC:** The Receiver Operating Characteristic curve plots TPR vs. FPR across all decision thresholds. Area Under the Curve (AUC) summarises discrimination ability independently of threshold choice.

### Regression

For a regressor with predictions $\hat{y}_i$ and ground truth $y_i$:

| Metric | Formula | Interpretation |
|---|---|---|
| MAE | $\frac{1}{n}\sum_i \|y_i - \hat{y}_i\|$ | Mean absolute error; robust to outliers |
| MSE | $\frac{1}{n}\sum_i (y_i - \hat{y}_i)^2$ | Mean squared error; penalises large errors quadratically |
| RMSE | $\sqrt{\text{MSE}}$ | Same units as target; interpretable |
| R² | $1 - \frac{\sum_i(y_i - \hat{y}_i)^2}{\sum_i(y_i - \bar{y})^2}$ | Proportion of variance explained; 1.0 is perfect |

**Common pitfall:** R² can be negative (model worse than predicting the mean) and is sensitive to the range of the target variable — always report alongside MAE or RMSE.

---

## 4. Comparing Classifiers Statistically

A single accuracy value from a single train/test split is not evidence of superiority. The correct procedure:

1. Compute per-fold metric values for each method (gives $k$ paired observations)
2. Test the null hypothesis that performance distributions are equal
3. Report effect size, not just significance

### Paired t-test (two classifiers, one dataset)

$$t = \frac{\bar{d}}{s_d / \sqrt{k}}$$

where $\bar{d}$ is the mean of per-fold differences and $s_d$ is their standard deviation. Valid under approximate normality of the differences.

### Wilcoxon Signed-Rank Test

Non-parametric alternative to the paired t-test. Preferred when the normality assumption is doubtful (common with small $k$).

### Multiple Classifier Comparison (Friedman + Nemenyi)

When comparing $m > 2$ classifiers across multiple datasets:

1. **Friedman test:** Rank classifiers per dataset; test whether mean ranks are equal (non-parametric ANOVA on ranks).
2. **Nemenyi post-hoc test:** Pairwise comparisons with family-wise error rate control.

This is the standard protocol in ML benchmarking literature (Demšar, 2006).

> **💡 Intuition — which test when?** Two methods, one dataset → **paired t-test** (if differences look normal) or **Wilcoxon** (if not). Many methods, many datasets → **Friedman** (are they all equal?) then **Nemenyi** (which pairs differ?). The golden rule: *a difference you can't defend with a test is not a result.*

---

## Worked Examples

Collected for revision: **all five confusion-matrix metrics from raw counts** (§3) and the **5-fold CV diagram** (§2). The confusion-matrix calculation is the most heavily examined skill in this module — practise it until the four boxes (TP/FP/FN/TN) are automatic.

---

## Self-Check

<details>
<summary><strong>Q1.</strong> Your model scores 95% accuracy on a dataset that is 95% one class. Should you be impressed? (§3)</summary>

No — a "always predict the majority class" baseline also scores 95%. Report precision/recall/F1 and compare against that trivial baseline.
</details>

<details>
<summary><strong>Q2.</strong> Confusion matrix: TP=80, FP=20, FN=10, TN=90. Compute precision and recall. (§3)</summary>

Precision $= 80/(80+20) = \mathbf{0.80}$. Recall $= 80/(80+10) \approx \mathbf{0.889}$. (This model is more likely to miss a false alarm than a real positive.)
</details>

<details>
<summary><strong>Q3.</strong> Why use <em>stratified</em> k-fold instead of plain k-fold? (§2)</summary>

Stratification keeps each fold's class proportions equal to the full dataset. Without it, on an imbalanced dataset a fold might contain few or zero minority-class samples, making that fold's metrics meaningless.
</details>

<details>
<summary><strong>Q4.</strong> Can R² be negative? What would that mean? (§3)</summary>

Yes. R² < 0 means the model predicts **worse than just outputting the mean** of the target — a real red flag (we saw it for the single Decision Tree in [Atividade 1](../activities/atividade-1.md#part-b--regression)).
</details>

<details>
<summary><strong>Q5.</strong> You want to <em>never miss a fraud</em>, even at the cost of false alarms. Optimise precision or recall? (§3)</summary>

**Recall** — you want to catch every true positive (fraud). High recall tolerates more false positives; the cost of a miss outweighs the cost of a false alarm.
</details>

---

## 🔑 Quick Revision

| Concept | One-line takeaway |
|---|---|
| Don't test on train | Training error is optimistically biased; the gap grows with model capacity |
| k-fold CV | Train $k$ times, every sample tested once, average the scores |
| Stratified k-fold | Preserves class ratios per fold — use it for imbalance/classification |
| Accuracy | $\frac{TP+TN}{n}$ — **lies under class imbalance** |
| Precision | $\frac{TP}{TP+FP}$ — "when I say positive, am I right?" |
| Recall | $\frac{TP}{TP+FN}$ — "did I catch all the positives?" |
| F1 | harmonic mean of precision & recall — balances the two |
| R² / MAE | R² = variance explained (can be < 0); always report MAE too |
| Significance | Wilcoxon (2 methods) · Friedman+Nemenyi (many) |

---

## Professor's References

> Full entries in [`../references.md`](../references.md).

- BISHOP, C. M. *Pattern Recognition and Machine Learning*. Springer, 2006. — Chapter 1.3 (model selection and the curse of dimensionality).
- DUDA, R. O.; HART, P. E.; STORK, D. G. *Pattern Classification*. 2. ed. Wiley-Interscience, 2000. — Chapter 9 (algorithm-independent machine learning).

---

## Extended Reading

- DEMŠAR, J. Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research*, v. 7, p. 1–30, 2006. Available at: <https://www.jmlr.org/papers/v7/demsar06a.html>. — The canonical reference for multi-classifier statistical comparison.
- KOHAVI, R. A study of cross-validation and bootstrap for accuracy estimation and model selection. In: *IJCAI*, 1995. p. 1137–1145. — Empirical analysis of CV variants.
- JAPKOWICZ, N.; SHAH, M. *Evaluating Learning Algorithms: A Classification Perspective*. Cambridge University Press, 2011. — Comprehensive treatment of evaluation methodology.

---

## Connected Activities

- [Atividade 1 — Part A](../activities/atividade-1.md#part-a--classification): 5-fold CV, accuracy/F1/precision/recall across nine classifiers. Questions A.1–A.4 directly exercise this module.
- [Atividade 1 — Part B](../activities/atividade-1.md#part-b--regression): 5-fold CV, R²/MAE across seven regressors. Question B.1.

---

*[← Module 1 — Foundations](01-foundations.md) · [README](../README.md) · [Module 3 — Shallow Techniques →](03-shallow.md)*
