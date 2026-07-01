# Module 3 — Shallow Techniques

> **Colour:** ![#f0883e](https://placehold.co/12x12/f0883e/f0883e.png) Orange-red  
> **Prerequisites:** [Module 1 — Foundations](01-foundations.md) · [Module 2 — Experimental Protocols](02-protocols.md)  
> **Connects to:** [Module 4 — Descriptive Models](04-descriptive.md) · [Module 5 — Deep Techniques](05-deep.md) · [Atividade 1](../activities/atividade-1.md)

---

## Overview

Shallow techniques are supervised learning methods that operate on hand-crafted feature representations — as opposed to deep methods that learn representations end-to-end. Despite the label "shallow", many of these methods (SVM, ensemble methods, gradient boosting) remain state-of-the-art on structured/tabular data and constitute the empirical baseline against which deep models must be justified.

This module covers supervised classifiers and regressors, organised by their inductive bias: linear, geometric, probabilistic, tree-based, instance-based, and ensemble methods.

---

## How to Use This Module (Exam Prep)

**After this module you should be able to:**
- Match each method to its **inductive bias** (the assumption it makes about the data).
- **Compute information gain** to choose a decision-tree split — by hand.
- Classify a point with **k-NN** and with **Naive Bayes** from numbers.
- Explain **bagging vs. boosting** (variance vs. bias) and why Random Forest decorrelates trees.
- Name the **key hyperparameters** of every method and what they control.

**⭐ High-yield for exams:** information gain / entropy split (compute it) · bagging vs. boosting · the kernel trick · why ensembles need diverse base learners · k-NN/Naive Bayes by hand.

**If you only read one thing:** §5 Tree-Based Methods (information gain is the most-computed exam task) and §6 Ensemble Methods.

**Suggested time:** ~2.5 h reading + ~1 h on the [Worked Examples](#worked-examples) and [Self-Check](#self-check). This is the largest module — budget accordingly.

---

## 1. Linear Models

Linear models are the foundational baselines of supervised learning: the smallest hypothesis class that is still useful, the reference every more complex method must beat, and the bridge from the Maximum Likelihood framework of [Module 1 §3](01-foundations.md#3-probability--statistics) to the parametric learning of [Module 5](05-deep.md).

### Linear Regression

Models the target as a linear function of the features, $\hat{y} = \mathbf{w}^\top \mathbf{x} + b$, fit by minimising the residual sum of squares:

$$\min_{\mathbf{w}, b} \sum_{i=1}^n (y_i - \mathbf{w}^\top \mathbf{x}_i - b)^2$$

The **normal equations** give the closed-form solution $\hat{\mathbf{w}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$, which exists uniquely when the Gram matrix $\mathbf{X}^\top \mathbf{X}$ is invertible (cf. [Module 1 §1](01-foundations.md#1-linear-algebra) — rank deficiency signals multicollinearity). Under the assumption of i.i.d. Gaussian noise, the least-squares solution **is** the Maximum Likelihood estimate.

**Regularised variants** trade a small increase in bias for a reduction in variance:

| Variant | Penalty added to loss | Effect |
|---|---|---|
| Ridge | $\lambda \|\mathbf{w}\|_2^2$ | Shrinks weights; stabilises ill-conditioned $\mathbf{X}^\top\mathbf{X}$ |
| Lasso | $\lambda \|\mathbf{w}\|_1$ | Induces sparsity — performs feature selection |
| Elastic Net | $\lambda_1 \|\mathbf{w}\|_1 + \lambda_2 \|\mathbf{w}\|_2^2$ | Compromise between the two |

**Key hyperparameters:** Regularisation strength $\lambda$ (and the L1/L2 mix for Elastic Net).

### Logistic Regression

Despite the name, this is the canonical **probabilistic linear classifier**. It passes the linear score through the logistic (sigmoid) function to produce a calibrated class probability:

$$P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}$$

Parameters are fit by minimising the **cross-entropy** (negative log-likelihood) — the direct MLE consequence of a Bernoulli likelihood:

$$\mathcal{L}(\mathbf{w}, b) = -\sum_{i=1}^n \left[ y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

There is no closed form; the convex objective is optimised by gradient-based methods (e.g. L-BFGS, SGD). Multiclass problems use the **softmax** generalisation. Logistic regression is the conceptual atom of the neural network: a single neuron with a sigmoid activation and a cross-entropy loss (cf. [Module 5 §1](05-deep.md#1-biological-motivation-and-the-perceptron)).

**Key hyperparameters:** Regularisation type (L1/L2) and strength $C = 1/\lambda$, solver, class weights (for imbalance).

**Inductive bias:** Linear decision boundary; log-odds linear in the features. Strong, interpretable baseline whenever classes are approximately linearly separable.

**Common pitfall:** Standardise features before applying regularisation — the penalty is not scale-invariant, so unscaled features are penalised unequally.

---

## 2. Instance-Based Methods

### k-Nearest Neighbours (k-NN)

A non-parametric method: the prediction for a query $\mathbf{x}$ is derived from the $k$ closest training points under a chosen distance metric.

**Classification:** majority vote among $k$ neighbours.  
**Regression:** mean (or distance-weighted mean) of $k$ neighbours' target values.

$$\hat{y} = \frac{1}{k} \sum_{i \in \mathcal{N}_k(\mathbf{x})} y_i$$

**Key hyperparameters:** $k$ (neighbourhood size), distance metric (Euclidean, Manhattan, Minkowski).

**Inductive bias:** Smoothness — nearby points have similar labels. Fails in high dimensions (curse of dimensionality) where distances become uninformative.

**Common pitfall:** Failing to standardise features before computing distances. Large-magnitude features dominate the distance computation regardless of their predictive relevance.

> **💡 Intuition:** k-NN has **no training phase** — it just memorises the data and, at prediction time, asks "who are my nearest neighbours, and what are they?" Small $k$ = jagged, noise-sensitive boundary; large $k$ = smooth, but blurs real detail.

> **📝 Worked Example — classify a point with k-NN ($k=3$)**
> Training points (feature, class): $A(1, \text{🔴})$, $B(2, \text{🔴})$, $C(4, \text{🔵})$, $D(5, \text{🔵})$, $E(6, \text{🔵})$. Classify query $q = 3$.
> Distances: $|3-1|=2$, $|3-2|=1$, $|3-4|=1$, $|3-5|=2$, $|3-6|=3$.
> The **3 nearest** are $B(d{=}1,🔴)$, $C(d{=}1,🔵)$, $A(d{=}2,🔴)$ → votes: 🔴 2, 🔵 1 → **predict 🔴**.
> Note: with $k=1$ it would be a tie-break between $B$ and $C$; with $k=5$ the 🔵 majority (3 vs 2) would flip it to 🔵. **$k$ changes the answer — that's why you tune it.**

---

## 3. Probabilistic Methods

### Naive Bayes

Applies Bayes' theorem under the conditional independence assumption:

$$P(y \mid \mathbf{x}) \propto P(y) \prod_{j=1}^d P(x_j \mid y)$$

The "naive" assumption — that features are independent given the class — is rarely true in practice but yields surprisingly competitive performance, especially in text classification and high-dimensional sparse data.

**Variants:** Gaussian NB (continuous features), Multinomial NB (count features), Bernoulli NB (binary features).

**Key hyperparameters:** Smoothing parameter $\alpha$ (Laplace/additive smoothing) to handle zero-probability estimates.

> **💡 Intuition:** "Naive" = pretend every feature is independent given the class. It's usually false, but the maths becomes a simple product and it works shockingly well — especially for text. **Why smoothing?** If a word never appeared with a class in training, $P(\text{word}\mid\text{class})=0$ would zero out the *entire* product. Laplace smoothing adds a tiny count so nothing is impossible.

> **📝 Worked Example — Naive Bayes spam (with counts)**
> Priors: $P(\text{spam})=0.4$, $P(\text{ham})=0.6$. Word likelihoods: $P(\text{"free"}\mid\text{spam})=0.8$, $P(\text{"free"}\mid\text{ham})=0.1$; $P(\text{"meeting"}\mid\text{spam})=0.1$, $P(\text{"meeting"}\mid\text{ham})=0.5$. Classify the email **"free meeting"**:
> - Spam score: $0.4 \times 0.8 \times 0.1 = 0.032$
> - Ham score: $0.6 \times 0.1 \times 0.5 = 0.030$
>
> Spam (0.032) > Ham (0.030) → **predict spam** (barely). We compare the *unnormalised* products because the denominator $P(\mathbf{x})$ is the same for both classes.

---

## 4. Geometric Methods

### Support Vector Machine (SVM)

Finds the maximum-margin hyperplane separating classes:

$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 \; \forall i$$

The **soft-margin** extension introduces slack variables $\xi_i \geq 0$ and a regularisation parameter $C$ controlling the bias–variance trade-off:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_i \xi_i$$

> **💡 Intuition:** SVM draws the boundary that sits in the **widest possible gap** between the classes — the "street" with the most room on both sides. Only the points on the kerb (the **support vectors**) matter; the rest could be deleted without changing the line. A bigger $C$ means "tolerate fewer mistakes" (narrower street, risk overfit); smaller $C$ means "allow some slack" (wider street, more bias).

**Kernel trick:** Maps inputs to a higher-dimensional feature space implicitly via a kernel function $K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^\top \phi(\mathbf{x}_j)$. Common kernels: RBF (Gaussian), polynomial, sigmoid.

> **💡 Intuition (kernel trick):** Some data can't be separated by a straight line in its original space — but *can* be after lifting it into more dimensions. The trick: you never actually compute those high-dimensional coordinates; the kernel gives you the dot product directly. Classic picture: two rings that aren't linearly separable in 2D become separable by a plane once you add a "height" dimension based on distance from centre.

**SVR (regression):** Uses an $\varepsilon$-insensitive loss — errors within a tube of width $\varepsilon$ around the prediction are ignored.

**Key hyperparameters:** $C$ (regularisation), kernel type, $\gamma$ (RBF bandwidth), $\varepsilon$ (SVR tube width).

---

## 5. Tree-Based Methods

### Decision Tree

Recursively partitions the feature space using axis-aligned splits. At each node, the split is chosen to maximise an impurity reduction criterion.

#### ID3 Algorithm

ID3 (Iterative Dichotomiser 3, Quinlan 1986) is the foundational decision tree learning algorithm. It selects splits using **Information Gain** — the reduction in entropy achieved by splitting on attribute $A$:

$$\text{Entropy}(S) = -\sum_{k=1}^{K} p_k \log_2 p_k$$

where $p_k$ is the proportion of examples in class $k$ in set $S$. Entropy is 0 when all examples belong to one class (pure node) and maximal at $\log_2 K$ when classes are uniformly distributed.

**Information Gain** of splitting $S$ on attribute $A$:

$$\text{Gain}(S, A) = \text{Entropy}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \cdot \text{Entropy}(S_v)$$

where $S_v$ is the subset of $S$ for which $A = v$. ID3 greedily selects the attribute $A^* = \arg\max_A \text{Gain}(S, A)$ at each node.

> **💡 Intuition:** Entropy measures *mixed-up-ness*. A node that is all one class has entropy 0 (perfectly pure); a 50/50 node has entropy 1 (maximally confused). A good split is one that produces **purer** children — information gain is simply *how much confusion you removed*. The tree greedily grabs the most informative question first.

> **📝 Worked Example — pick the best split with information gain** ⭐ *(the classic exam question)*
> 14 days, target **Play?** = 9 Yes / 5 No. We test splitting on **Wind ∈ {Weak, Strong}**.
>
> **Step 1 — parent entropy** (9 Yes, 5 No out of 14):
> $$\text{Entropy}(S) = -\tfrac{9}{14}\log_2\tfrac{9}{14} - \tfrac{5}{14}\log_2\tfrac{5}{14} = 0.410 + 0.530 = \mathbf{0.940}$$
>
> **Step 2 — child entropies.** Weak (8 days: 6 Yes, 2 No); Strong (6 days: 3 Yes, 3 No):
> $$\text{Entropy}(\text{Weak}) = -\tfrac{6}{8}\log_2\tfrac{6}{8} - \tfrac{2}{8}\log_2\tfrac{2}{8} = \mathbf{0.811}, \qquad \text{Entropy}(\text{Strong}) = -\tfrac{3}{6}\log_2\tfrac{3}{6}-\tfrac{3}{6}\log_2\tfrac{3}{6} = \mathbf{1.000}$$
>
> **Step 3 — weighted child entropy, then gain:**
> $$\tfrac{8}{14}(0.811) + \tfrac{6}{14}(1.000) = 0.463 + 0.429 = 0.892 \;\Rightarrow\; \text{Gain}(S,\text{Wind}) = 0.940 - 0.892 = \mathbf{0.048}$$
>
> **Reading it:** Wind removes only 0.048 bits of confusion — a weak split. In the full Play-Tennis dataset, **Outlook** scores ≈ 0.247, so ID3 picks Outlook for the root. **Method to memorise: parent entropy − weighted average of child entropies = gain; biggest gain wins.**

**Limitations of ID3:** Biased toward attributes with many values (high cardinality); no pruning; handles only categorical attributes. **C4.5** (Quinlan 1993) extends ID3 with Gain Ratio (normalises by split information), continuous attribute handling, and pruning. **CART** uses Gini impurity instead of entropy and produces binary splits.

**Impurity criteria comparison:**

| Criterion | Formula | Algorithm |
|---|---|---|
| Entropy / Information Gain | $-\sum_k p_k \log_2 p_k$ | ID3, C4.5 |
| Gini impurity | $1 - \sum_k p_k^2$ | CART |
| Variance reduction | $\text{Var}(y) - \sum_c w_c \text{Var}(y_c)$ | CART (regression) |

**Key hyperparameters:** Maximum depth, minimum samples per leaf, impurity criterion.

**Inductive bias:** Axis-aligned decision boundaries; high variance (prone to overfitting) but zero bias on training data when fully grown.

---

## 6. Ensemble Methods

Ensemble methods combine multiple base learners to reduce variance (bagging), bias (boosting), or both (stacking).

> **💡 Intuition — bagging vs. boosting (exam favourite):**
> - **Bagging** (Random Forest): train many models **in parallel** on different bootstrap samples and **average** them. Like polling many independent experts — averaging cancels their random errors. Attacks **variance**. Base learners should be high-variance (deep trees).
> - **Boosting** (AdaBoost, XGBoost): train models **in sequence**, each one focusing on the mistakes of the last. Like a student drilling the questions they keep getting wrong. Attacks **bias**. Base learners are weak (shallow trees/stumps).
>
> ```
>   BAGGING (parallel, vote)          BOOSTING (sequential, reweight errors)
>   data ─┬─ bootstrap ─ tree₁         data ─ model₁ ─► errors ─ model₂ ─► errors ─ model₃
>         ├─ bootstrap ─ tree₂                 (each model fixes the previous one's misses)
>         └─ bootstrap ─ tree₃
>              └─► average / majority vote
> ```

### Bagging (Bootstrap Aggregating)

Trains $T$ base models on bootstrap samples of the training data. Predictions are aggregated by majority vote (classification) or mean (regression):

$$\hat{y} = \frac{1}{T} \sum_{t=1}^T f_t(\mathbf{x})$$

Reduces variance without increasing bias. Most effective with high-variance base learners (e.g., deep decision trees).

### Random Forest

Bagging of decision trees with an additional feature subsampling step: at each split, only a random subset of $m \leq d$ features is considered. This decorrelates the trees, reducing ensemble variance further.

**Key hyperparameters:** Number of trees $T$, features per split $m$ (typically $\sqrt{d}$ for classification, $d/3$ for regression), tree depth.

### AdaBoost

Sequential ensemble: each model $f_t$ focuses on the examples misclassified by the previous ensemble. Sample weights are updated multiplicatively after each round:

$$w_i^{(t+1)} \propto w_i^{(t)} \cdot \exp(-\alpha_t y_i f_t(\mathbf{x}_i))$$

where $\alpha_t = \frac{1}{2} \ln\frac{1 - \varepsilon_t}{\varepsilon_t}$ is the weight of model $t$ and $\varepsilon_t$ is its weighted error. **Note on encoding:** this update assumes labels and predictions in $\{-1, +1\}$, so that $y_i f_t(\mathbf{x}_i) = +1$ for a correct prediction (weight decreases) and $-1$ for an error (weight increases). The $\{0, 1\}$ encoding used elsewhere in this course must be mapped to $\{-1, +1\}$ for this formula to hold.

**Inductive bias:** Reduces bias progressively; sensitive to outliers (noisy labels get high weight).

### XGBoost (Extreme Gradient Boosting)

Gradient boosting with second-order Taylor expansion of the loss, regularised objective, and several systems-level optimisations (column subsampling, parallel split finding, cache-aware access):

$$\mathcal{L}^{(t)} = \sum_i \left[ g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \Omega(f_t)$$

where $g_i = \partial_{\hat{y}} \ell(y_i, \hat{y})$ and $h_i = \partial^2_{\hat{y}} \ell(y_i, \hat{y})$ are first and second derivatives of the loss.

**Key hyperparameters:** Learning rate $\eta$, max depth, subsample ratio, $\lambda$ (L2 regularisation), $\alpha$ (L1 regularisation), number of rounds.

### Heterogeneous Ensembles and Classifier Combination

While homogeneous ensembles (Bagging, Boosting, Random Forest) combine multiple instances of the **same** base learner type, **heterogeneous ensembles** combine fundamentally different classifiers — e.g., a Decision Tree, a k-NN, and a Naive Bayes — exploiting the diversity of their distinct inductive biases.

**Motivation:** Classifiers that make different types of errors on different regions of the feature space are better candidates for combination than classifiers with correlated errors. Diversity is the key ingredient.

#### Combination Rules (Fixed)

Given $T$ classifiers $\{f_1, \ldots, f_T\}$ and $K$ classes, fixed combination rules aggregate outputs without learning additional parameters:

| Rule | Input type | Formula / logic |
|---|---|---|
| **Majority vote** | Class labels | $\hat{y} = \arg\max_k \sum_t \mathbb{1}[f_t(\mathbf{x}) = k]$ |
| **Weighted vote** | Class labels | $\hat{y} = \arg\max_k \sum_t w_t \cdot \mathbb{1}[f_t(\mathbf{x}) = k]$, $w_t$ set by validation accuracy |
| **Sum rule** | Posterior probabilities | $\hat{y} = \arg\max_k \sum_t P_t(k \mid \mathbf{x})$ |
| **Product rule** | Posterior probabilities | $\hat{y} = \arg\max_k \prod_t P_t(k \mid \mathbf{x})$ — assumes classifier independence |
| **Max rule** | Posterior probabilities | $\hat{y} = \arg\max_k \max_t P_t(k \mid \mathbf{x})$ |
| **Borda count** | Rankings | Sum of rank positions across classifiers |

**Sum rule** is empirically robust and theoretically well-motivated when classifiers produce calibrated probability estimates. **Product rule** is theoretically optimal under classifier independence (rarely holds in practice) but degrades badly when one classifier assigns near-zero probability.

#### Stacking (Stacked Generalisation)

A **trainable** combination method: a meta-learner $g$ is trained on the out-of-fold predictions of the base classifiers:

1. Split training data into $k$ folds.
2. For each fold, train all base classifiers on the remaining $k-1$ folds and collect predictions on the held-out fold.
3. Assemble a new dataset where each row is the vector of base classifier predictions; train $g$ on this dataset.
4. At test time: obtain base classifier predictions, feed to $g$ for the final prediction.

Stacking can learn non-linear combinations and asymmetric weighting — at the cost of added complexity and risk of overfitting the meta-learner.

#### Classifier Selection vs. Fusion

- **Fusion:** All classifiers contribute to every prediction (rules above).
- **Selection:** For a given query $\mathbf{x}$, select the single most competent classifier based on local accuracy in the neighbourhood of $\mathbf{x}$ (Dynamic Classifier Selection, DCS) or combine a subset (Dynamic Ensemble Selection, DES).

**Common pitfall:** Combining classifiers that are highly correlated (e.g., five Random Forests trained on the same features) yields negligible improvement — diversity is the prerequisite for ensemble gain.

---

### MLP as a Shallow Baseline

A Multilayer Perceptron with one or two hidden layers appears in this module as a **transition model** — shallow in the sense that feature engineering is still largely manual, but introducing the parametric learning paradigm that Module 5 develops deeply.

$$\mathbf{h} = \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1), \quad \hat{y} = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2$$

**Key hyperparameters:** Layer sizes, activation function ($\sigma$: ReLU, tanh, logistic), learning rate, regularisation (L2, dropout), batch size, epochs.

---

## Classifier / Regressor Summary

| Method | Type | Inductive Bias | Key Hyperparameters |
|---|---|---|---|
| Linear / Ridge / Lasso Regression | Regressor | Linear response; sparsity (L1) | $\lambda$, L1/L2 mix |
| Logistic Regression | Classifier | Linear log-odds; calibrated probabilities | $C$, penalty, solver |
| k-NN | Both | Smoothness, locality | $k$, distance metric |
| Naive Bayes | Classifier | Feature independence given class | Smoothing $\alpha$ |
| SVM / SVR | Both | Maximum margin / $\varepsilon$-tube | $C$, kernel, $\gamma$, $\varepsilon$ |
| Decision Tree | Both | Axis-aligned splits | Max depth, min samples |
| Random Forest | Both | Bagged decorrelated trees | $T$, $m$, depth |
| Bagging | Both | Variance reduction via bootstrap | $T$, base learner |
| AdaBoost | Classifier | Sequential bias reduction | $T$, base learner |
| XGBoost | Both | Regularised gradient boosting | $\eta$, depth, $\lambda$, $\alpha$ |
| MLP | Both | Universal approximation | Architecture, $\eta$, regularisation |

---

## Worked Examples

Collected for revision: **k-NN classification** (§2), **Naive Bayes spam** (§3), and — most important — **information gain for a tree split** (§5). The information-gain calculation is the single most likely hand-computation on the exam; practise it on a fresh dataset until the *parent − weighted children* recipe is automatic.

---

## Self-Check

<details>
<summary><strong>Q1.</strong> A node has 4 Yes and 4 No. What is its entropy, and is it a good or bad node? (§5)</summary>

$-\tfrac{4}{8}\log_2\tfrac{4}{8} - \tfrac{4}{8}\log_2\tfrac{4}{8} = 1.0$ bit — **maximum entropy**, the worst (most mixed) possible node. A perfect split would drive children toward entropy 0.
</details>

<details>
<summary><strong>Q2.</strong> One sentence each: how does bagging reduce error vs. how does boosting? (§6)</summary>

**Bagging** averages many independent high-variance models to cancel their random errors (↓variance). **Boosting** trains models in sequence, each correcting the previous one's mistakes (↓bias).
</details>

<details>
<summary><strong>Q3.</strong> Why does Random Forest sample features at each split, not just rows? (§6)</summary>

Row bootstrapping alone leaves trees correlated (they all latch onto the same strong feature). Random feature subsets **decorrelate** the trees, so averaging removes more variance.
</details>

<details>
<summary><strong>Q4.</strong> You combine five identical Random Forests on the same data. Big improvement? (§6)</summary>

No — they make **correlated** errors. Ensembles only help when base learners are **diverse**. Combining near-identical models adds cost, not accuracy.
</details>

<details>
<summary><strong>Q5.</strong> Naive Bayes assumes feature independence, which is usually false. Why use it anyway? (§3)</summary>

It still ranks the correct class highest surprisingly often, needs little data, trains instantly, and excels on high-dimensional sparse data (text). For *classification* you only need the right argmax, not calibrated probabilities.
</details>

<details>
<summary><strong>Q6.</strong> Which methods need feature scaling: Decision Tree, k-NN, SVM, Naive Bayes? (§2, §4)</summary>

**k-NN and SVM** need scaling (distance/margin based). **Decision Tree and Gaussian Naive Bayes** do not (threshold splits / per-feature distributions are scale-invariant).
</details>

---

## 🔑 Quick Revision

| Method | Bias / idea in 5 words | Key knob |
|---|---|---|
| Linear / Logistic Reg. | straight-line boundary, interpretable baseline | $\lambda$ / $C$ |
| k-NN | "vote of nearest neighbours" | $k$ (+ scaling!) |
| Naive Bayes | "independent features, multiply probabilities" | smoothing $\alpha$ |
| SVM | "widest-margin street; kernels for curves" | $C$, kernel, $\gamma$ |
| Decision Tree | "greedy purest split (info gain)" | depth, criterion |
| Random Forest | "many decorrelated trees, vote" | $T$, $m$ |
| Bagging | "parallel models, average → ↓variance" | $T$, base learner |
| AdaBoost / XGBoost | "sequential, fix prior errors → ↓bias" | $\eta$, depth, $T$ |
| MLP | "stacked neurons, universal approximator" | architecture, $\eta$ |

**Two mantras:** *Information gain = parent entropy − weighted child entropy.* *Bagging fights variance (parallel); boosting fights bias (sequential).*

---

## Professor's References

> Full entries in [`../references.md`](../references.md).

- BISHOP, C. M. *Pattern Recognition and Machine Learning*. Springer, 2006. — Chapters 3–7 (linear models, kernel methods, sparse models).
- DUDA, R. O.; HART, P. E.; STORK, D. G. *Pattern Classification*. 2. ed. Wiley-Interscience, 2000. — Chapters 4–8.
- BREIMAN, L. Random forests. *Machine Learning*, v. 45, n. 1, p. 5–32, 2001.
- FREUND, Y.; SCHAPIRE, R. E. A decision-theoretic generalization of on-line learning and an application to boosting. *Journal of Computer and System Sciences*, v. 55, n. 1, p. 119–139, 1997.
- CHEN, T.; GUESTRIN, C. XGBoost: A scalable tree boosting system. In: *KDD*, 2016. p. 785–794. Available at: <https://arxiv.org/abs/1603.02754>.

---

## Extended Reading

- FRIEDMAN, J.; HASTIE, T.; TIBSHIRANI, R. *The Elements of Statistical Learning*. 2. ed. Springer, 2009. — Freely available at <https://hastie.su.domains/ElemStatLearn/>. Chapters 9–10 (trees and boosting), 12 (SVM).
- VAPNIK, V. N. *The Nature of Statistical Learning Theory*. 2. ed. Springer, 2000. — Original SVM theoretical foundation.
- SCHÖLKOPF, B.; SMOLA, A. J. *Learning with Kernels*. MIT Press, 2002. — Comprehensive kernel methods reference.

---

## Connected Activities

- [Atividade 1 — Part A](../activities/atividade-1.md#part-a--classification): Ten classifiers from this module (Logistic Regression, Decision Tree, k-NN, Naive Bayes, SVM, MLP, Random Forest, Bagging, AdaBoost, XGBoost).
- [Atividade 1 — Part B](../activities/atividade-1.md#part-b--regression): Eight regressors (Ridge, Decision Tree, k-NN, SVR, MLP, Random Forest, Bagging, XGBoost).

---

*[← Module 2 — Experimental Protocols](02-protocols.md) · [README](../README.md) · [Module 4 — Descriptive Models →](04-descriptive.md)*
