# Module 1 — Foundations

> **Colour:** ![#3fb950](https://placehold.co/12x12/3fb950/3fb950.png) Green  
> **Prerequisites:** Calculus (derivatives, gradients), basic programming (Python/NumPy)  
> **Connects to:** [Module 2 — Experimental Protocols](02-protocols.md) · [Module 3 — Shallow Techniques](03-shallow.md)

---

## Overview

Foundations establishes the mathematical substrate on which all Machine Learning methods are built. Three pillars are covered: **linear algebra** (the language of data representation and transformations), **analytical geometry** (the spatial intuition behind distance, margin, and projection), and **probability & statistics** (the formal framework for uncertainty, inference, and model evaluation).

These are not prerequisites to be skimmed — they are the precise tools invoked when reading algorithm derivations, interpreting loss functions, or understanding why a method fails on a given dataset. Every subsequent module assumes fluency with the notation and results introduced here.

---

## How to Use This Module (Exam Prep)

**After this module you should be able to:**
- Read ML notation fluently — $\mathbf{X} \in \mathbb{R}^{n \times d}$, vectors, matrices, norms — without slowing down.
- State **and apply** Bayes' theorem to a numeric problem.
- Explain the **bias–variance decomposition** and say what each term means.
- Compute a Euclidean / Manhattan distance and justify why feature standardisation matters.
- Define **MLE** and connect it to cross-entropy (classification) and squared-error (regression) losses.

**⭐ High-yield for exams:** Bayes' theorem (compute it) · bias–variance decomposition (explain each term) · why scaling matters for distance/gradient methods · MLE → loss connection.

**If you only read one thing:** §3 Probability & Statistics — it powers Naive Bayes, every evaluation metric, and every loss function in the course.

**Suggested time:** ~90 min reading + ~45 min on the [Worked Examples](#worked-examples) and [Self-Check](#self-check).

---

## 1. Linear Algebra

### Vectors and Matrices

A dataset of $n$ samples with $d$ features is represented as a matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$, where each row $\mathbf{x}_i \in \mathbb{R}^d$ is a feature vector. A target vector $\mathbf{y} \in \mathbb{R}^n$ collects the labels or response values.

Key operations:

- **Matrix–vector product:** $\mathbf{Xw}$ computes a linear combination of features weighted by $\mathbf{w}$ — the core of every linear model.
- **Transpose:** $\mathbf{X}^\top \mathbf{X} \in \mathbb{R}^{d \times d}$ is the Gram matrix; its invertibility determines whether the normal equations have a unique solution.
- **Rank:** $\text{rank}(\mathbf{X}) < d$ signals multicollinearity — features are linearly dependent.

### Eigendecomposition

For a symmetric positive semi-definite matrix $\mathbf{A}$:

$$\mathbf{A} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^\top$$

where $\mathbf{Q}$ contains orthonormal eigenvectors and $\mathbf{\Lambda} = \text{diag}(\lambda_1, \ldots, \lambda_d)$ contains non-negative eigenvalues. This decomposition underlies **Principal Component Analysis (PCA)** and the analysis of covariance structures.

### Singular Value Decomposition (SVD)

> **💡 Intuition:** *Every* linear map is just **rotate → stretch → rotate**. SVD makes that explicit: $\mathbf{V}^\top$ rotates the data, $\mathbf{\Sigma}$ stretches each axis by a singular value, and $\mathbf{U}$ rotates again. The largest singular values point along the directions where the data varies most — which is exactly why SVD is the engine behind PCA and image/data compression (keep the big singular values, throw away the small ones).

Any matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$ decomposes as:

$$\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top$$

where $\mathbf{U} \in \mathbb{R}^{n \times n}$ and $\mathbf{V} \in \mathbb{R}^{d \times d}$ are orthogonal, and $\mathbf{\Sigma} \in \mathbb{R}^{n \times d}$ is rectangular-diagonal (non-negative singular values on the main diagonal, zeros elsewhere). SVD is the numerical backbone of PCA, low-rank approximation, and pseudo-inverse computation.

### Norms

- **$\ell_2$ (Euclidean):** $\|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2}$ — used in k-NN distance and SVM margin.
- **$\ell_1$ (Manhattan):** $\|\mathbf{x}\|_1 = \sum_i |x_i|$ — induces sparsity in Lasso regularisation.
- **Frobenius:** $\|\mathbf{A}\|_F = \sqrt{\sum_{i,j} a_{ij}^2}$ — matrix analogue of the $\ell_2$ norm.

**Common pitfall:** Failing to normalise features before computing distances causes high-magnitude features to dominate — directly affecting k-NN, SVM, and gradient-based methods.

---

## 2. Analytical Geometry

### Distance and Similarity

The choice of distance metric is a modelling decision, not a default. Common metrics:

| Metric | Formula | Typical use |
|--------|---------|-------------|
| Euclidean | $\sqrt{\sum_i (x_i - y_i)^2}$ | k-NN, k-means, SVM (RBF) |
| Manhattan | $\sum_i |x_i - y_i|$ | Robust to outliers; high-dimensional data |
| Cosine | $1 - \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|}$ | Text/sparse vectors; direction over magnitude |
| Mahalanobis | $\sqrt{(\mathbf{x}-\mathbf{y})^\top \mathbf{S}^{-1} (\mathbf{x}-\mathbf{y})}$ | Accounts for feature covariance |

> **📝 Worked Example — why standardisation is not optional**
> Two patients described by `[age (years), income ($)]`: $\mathbf{x} = [40,\ 50000]$, $\mathbf{y} = [42,\ 50500]$.
> Euclidean distance: $\sqrt{(40-42)^2 + (50000-50500)^2} = \sqrt{4 + 250000} \approx \mathbf{500.0}$.
> The age difference (2 years) is **completely drowned out** by income — distance is essentially "income only." After standardising each feature to mean 0, variance 1, both contribute on equal footing. **This is why k-NN, SVM, and MLP require `StandardScaler`, while trees (which split one feature at a time) do not.**

### Hyperplanes and Margins

A hyperplane in $\mathbb{R}^d$ is defined by:

$$\mathbf{w}^\top \mathbf{x} + b = 0$$

Points on either side satisfy $\mathbf{w}^\top \mathbf{x} + b > 0$ or $< 0$ — this is the geometric basis of linear classifiers. The **margin** is the perpendicular distance between the hyperplane and the nearest data points, equal to $\frac{2}{\|\mathbf{w}\|_2}$ in the SVM formulation.

### Projections

The orthogonal projection of $\mathbf{x}$ onto a unit vector $\mathbf{u}$ is $(\mathbf{x}^\top \mathbf{u})\mathbf{u}$. PCA finds the directions $\mathbf{u}_1, \ldots, \mathbf{u}_k$ that maximise projected variance — reducing dimensionality while retaining maximal information.

---

## 3. Probability & Statistics

### Probability Fundamentals

- **Random variable:** A mapping from outcomes to real values. Discrete (Bernoulli, Categorical) or continuous (Gaussian, Exponential).
- **Joint, marginal, conditional:** $P(X, Y) = P(Y \mid X) \cdot P(X)$ — the chain rule underpins generative models.
- **Bayes' theorem:** $P(Y \mid X) = \frac{P(X \mid Y) \cdot P(Y)}{P(X)}$ — the foundation of Naive Bayes and Bayesian inference.

> **💡 Intuition:** Bayes' theorem *updates a belief with evidence*. You start with a prior $P(Y)$ (how likely the class is before seeing data), multiply by the likelihood $P(X \mid Y)$ (how well the class explains the evidence), and renormalise. The classic exam trap: a test that is "99% accurate" for a *rare* disease still produces mostly false alarms — because the prior is tiny.

> **📝 Worked Example — Bayes' theorem (the disease-test classic)**
> A disease affects **1%** of people: $P(D) = 0.01$. A test detects it with **99%** sensitivity, $P(+\mid D)=0.99$, and has a **5%** false-positive rate, $P(+\mid \neg D)=0.05$. You test positive. What is $P(D \mid +)$?
>
> 1. **Total probability of a positive test:**
>    $$P(+) = P(+\mid D)P(D) + P(+\mid \neg D)P(\neg D) = (0.99)(0.01) + (0.05)(0.99) = 0.0099 + 0.0495 = 0.0594$$
> 2. **Apply Bayes:**
>    $$P(D \mid +) = \frac{P(+\mid D)P(D)}{P(+)} = \frac{0.0099}{0.0594} \approx \mathbf{0.167}$$
>
> **Lesson:** Despite a "99% accurate" test, a positive result means only a **~17% chance** of actually having the disease, because the disease is rare. This is *exactly* why accuracy is misleading under class imbalance (see [Module 2 §3](02-protocols.md#3-evaluation-metrics)).

### Key Distributions

| Distribution | Parameters | ML relevance |
|---|---|---|
| Bernoulli | $p$ | Binary classification labels |
| Categorical | $\mathbf{p} \in \Delta^{K-1}$ | Multiclass labels; softmax output |
| Gaussian | $\mu, \sigma^2$ | Feature modelling; LDA; noise assumptions |
| Multivariate Gaussian | $\boldsymbol{\mu}, \boldsymbol{\Sigma}$ | Gaussian mixture models; QDA |

### Expectation and Variance

$$\mathbb{E}[X] = \int x \, p(x) \, dx \qquad \text{Var}[X] = \mathbb{E}[(X - \mathbb{E}[X])^2]$$

The **bias–variance decomposition** of the expected generalisation error of an estimator $\hat{f}$:

$$\mathbb{E}[(y - \hat{f}(\mathbf{x}))^2] = \text{Bias}^2[\hat{f}] + \text{Var}[\hat{f}] + \sigma^2_\varepsilon$$

where $\sigma^2_\varepsilon$ is irreducible noise. This identity is the theoretical motivation for regularisation, ensemble methods, and model complexity control.

> **💡 Intuition:** **Bias** = error from the model being too *simple* to capture the pattern (underfitting). **Variance** = error from the model being too *sensitive* to the particular training sample (overfitting). You cannot drive both to zero; tuning model complexity trades one for the other. The sweet spot is the bottom of the U-curve.
>
> ```
>  error
>   ^
>   |  \                              /   ← total error (what you measure)
>   |   \                          /
>   |    \.                      ./
>   |      \.        *         ./        * = sweet spot (best complexity)
>   |        \..   variance ../
>   |   bias    \........../
>   |  (high when                (variance high when
>   |   too simple)               too complex)
>   +-------------------------------------> model complexity
>      underfit        good        overfit
> ```
>
> **Exam-ready one-liner:** *low bias + high variance = overfitting; high bias + low variance = underfitting; regularisation and ensembles buy variance reduction at a small bias cost.*

### Maximum Likelihood Estimation (MLE)

Given i.i.d. observations $\{\mathbf{x}_i\}_{i=1}^n$ and a parametric model $p(\mathbf{x} \mid \boldsymbol{\theta})$:

$$\hat{\boldsymbol{\theta}}_{\text{MLE}} = \arg\max_{\boldsymbol{\theta}} \sum_{i=1}^n \log p(\mathbf{x}_i \mid \boldsymbol{\theta})$$

MLE is the derivation basis for cross-entropy loss (classification) and mean squared error (regression under Gaussian noise).

### Hypothesis Testing and Significance

When comparing classifier performance across multiple datasets or folds, statistical tests are required:

- **Paired t-test:** Appropriate for comparing two classifiers on the same dataset with k-fold CV results.
- **Wilcoxon signed-rank test:** Non-parametric alternative; preferred when normality of differences cannot be assumed.
- **Friedman test + Nemenyi post-hoc:** Standard for comparing multiple classifiers across multiple datasets.

**Common pitfall:** Reporting accuracy differences without a significance test overstates the evidence for superiority — a difference of 1–2% on a single dataset is rarely statistically meaningful.

---

## Worked Examples

The boxes above are collected here for revision: **Bayes' theorem (disease test)** in §3, **distance & standardisation** in §2, and the **bias–variance U-curve** in §3. Re-derive each one from a blank page before the exam — recognising a formula is not the same as being able to apply it.

---

## Self-Check

Try each before expanding. If you can't answer in ~1 minute, re-read the linked section.

<details>
<summary><strong>Q1.</strong> A spam filter is "98% accurate," but 98% of email is <em>not</em> spam. Is the filter useful? (§3)</summary>

Not necessarily — a model that labels **everything "not spam"** also scores 98% accuracy while catching zero spam. Accuracy is misleading under class imbalance; you need precision/recall ([Module 2 §3](02-protocols.md#3-evaluation-metrics)).
</details>

<details>
<summary><strong>Q2.</strong> Disease prevalence is 2%, test sensitivity 95%, false-positive rate 10%. Roughly what is P(disease | positive)? (§3)</summary>

$P(+) = (0.95)(0.02) + (0.10)(0.98) = 0.019 + 0.098 = 0.117$; so $P(D\mid+) = 0.019/0.117 \approx \mathbf{0.16}$ (~16%). Same lesson as the worked example: a rare condition keeps the posterior low.
</details>

<details>
<summary><strong>Q3.</strong> A model gets 0% training error but 40% test error. Bias or variance problem? Name two fixes. (§3)</summary>

**High variance** (overfitting). Fixes: add regularisation, get more data, reduce model complexity, or use an ensemble (bagging).
</details>

<details>
<summary><strong>Q4.</strong> Why does k-NN need feature scaling but a decision tree does not? (§1–2)</summary>

k-NN ranks neighbours by **distance**, which sums over all features — large-magnitude features dominate (see worked example). A tree splits on **one feature at a time** using thresholds, so any monotonic rescaling leaves the splits unchanged.
</details>

<details>
<summary><strong>Q5.</strong> In one sentence, how does MLE connect to the loss you minimise when training? (§3)</summary>

Minimising a loss = maximising likelihood under an assumed noise model: **Gaussian noise → squared error (regression)**; **Bernoulli/categorical → cross-entropy (classification)**.
</details>

---

## 🔑 Quick Revision

| Concept | One-line takeaway |
|---|---|
| $\mathbf{X} \in \mathbb{R}^{n \times d}$ | $n$ rows (samples), $d$ columns (features) — memorise this shape |
| SVD | rotate → stretch → rotate; big singular values = main directions (powers PCA) |
| Norms | $\ell_2$ = Euclidean (margins, k-NN); $\ell_1$ = sparsity (Lasso) |
| Distance + scaling | large-magnitude features dominate distance → **standardise** for k-NN/SVM/MLP |
| Bayes' theorem | posterior ∝ likelihood × prior; rare priors keep posteriors low |
| Bias–variance | simple = bias/underfit; complex = variance/overfit; aim for the U-curve bottom |
| MLE | Gaussian → MSE; Bernoulli/categorical → cross-entropy |
| Significance tests | Wilcoxon (2 methods), Friedman+Nemenyi (many) — never trust a 1–2% gap alone |

---

## Professor's References

> Full entries in [`../references.md`](../references.md).

- BISHOP, C. M. *Pattern Recognition and Machine Learning*. Springer, 2006. — Chapters 1–2 (probability, decision theory) and Appendix C (linear algebra).
- DUDA, R. O.; HART, P. E.; STORK, D. G. *Pattern Classification*. 2. ed. Wiley-Interscience, 2000. — Chapter 2 (Bayesian decision theory).
- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. MIT Press, 2016. — Part I: Applied Math and Machine Learning Basics (Chapters 2–5). Available at: <https://www.deeplearningbook.org>.

---

## Extended Reading

- STRANG, G. *Introduction to Linear Algebra*. 5. ed. Wellesley-Cambridge Press, 2016. — The definitive accessible treatment; Chapters 6–7 for eigenvalues and SVD.
- BLITZSTEIN, J. K.; HWANG, J. *Introduction to Probability*. 2. ed. CRC Press, 2019. — Rigorous but readable; freely available at <https://projects.iq.harvard.edu/stat110>.
- PETERSEN, K. B.; PEDERSEN, M. S. *The Matrix Cookbook*. Technical University of Denmark, 2012. — Dense reference for matrix identities and derivatives. Available at: <https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf>.
- GRINSTEAD, C. M.; SNELL, J. L. *Introduction to Probability*. American Mathematical Society, 2006. — Open access at <https://math.dartmouth.edu/~prob/prob/prob.pdf>.

---

## Connected Activities

No activity in this course directly targets Module 1 in isolation — foundational concepts are exercised implicitly through all experimental work. The following connections are most direct:

- [Atividade 1 — Part A](../activities/atividade-1.md#part-a--classification): Feature space geometry (distance metrics, decision boundaries) is operative in k-NN and SVM.
- [Atividade 1 — Part B](../activities/atividade-1.md#part-b--regression): Bias–variance trade-off is observable empirically across the regressor comparison.

---

*← [README](../README.md) · [Module 2 — Experimental Protocols →](02-protocols.md)*
