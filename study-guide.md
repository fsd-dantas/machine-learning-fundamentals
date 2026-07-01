# Exam Study Guide

> A practical plan for using this repository to prepare for the **Aprendizagem de Máquina** exam (PPGIA/PUC-PR). Start here, not at Module 1.

---

## How this repo is built for studying

Every module now follows the same five-part learning structure — use them in this order:

1. **How to Use This Module** — learning objectives + ⭐ high-yield topics + "if you only read one thing."
2. **💡 Intuition** boxes — plain-language explanation *before* the maths. Read these first if a formula scares you.
3. **📝 Worked Examples** — real numbers plugged into the formulas. **Re-derive these from a blank page** — this is the difference between recognising and being able to *do*.
4. **✅ Self-Check** — questions with hidden answers. If you can't answer in ~1 minute, re-read the linked section.
5. **🔑 Quick Revision** — a one-table cheat sheet per module for the night before.

Supporting files: **[glossary.md](glossary.md)** (every acronym, one line each) and **[flashcards.md](flashcards.md)** (Q/A for spaced repetition).

---

## Suggested study order & time budget

The modules have prerequisites, so follow the numeric order. Rough budget for a from-scratch pass:

| Order | Module | Read | Practice | Why this order |
|---|---|---|---|---|
| 1 | [01 — Foundations](modules/01-foundations.md) | 90 min | 45 min | Notation + Bayes + bias–variance underpin everything |
| 2 | [02 — Protocols](modules/02-protocols.md) | 75 min | 45 min | You must know *how results are measured* before judging methods |
| 3 | [03 — Shallow](modules/03-shallow.md) | 2.5 h | 60 min | The core of the course; largest module |
| 4 | [04 — Descriptive](modules/04-descriptive.md) | 75 min | 40 min | Unsupervised — lighter, mostly conceptual |
| 5 | [05 — Deep](modules/05-deep.md) | 2 h | 45 min | Builds on the MLP/perceptron from Module 3 |

**Total ≈ 13 hours** for a thorough first pass. Split across a week, that's < 2 h/day.

**Crunched for time (one evening)?** Read every module's **How to Use** + **Quick Revision** + **Worked Examples** only (~2 h), then drill [flashcards.md](flashcards.md).

---

## ⭐ The highest-yield topics (where exam points live)

These recur across the modules' high-yield flags. If you master nothing else, master these:

1. **Compute information gain** for a decision-tree split — [Module 3 §5](modules/03-shallow.md#5-tree-based-methods). *Most likely hand-calculation on the exam.*
2. **Confusion-matrix metrics** (accuracy/precision/recall/F1/specificity) from counts — [Module 2 §3](modules/02-protocols.md#3-evaluation-metrics).
3. **Bias–variance** — name each term, link to over/underfitting — [Module 1 §3](modules/01-foundations.md#3-probability--statistics).
4. **Bagging vs. boosting** — parallel/variance vs. sequential/bias — [Module 3 §6](modules/03-shallow.md#6-ensemble-methods).
5. **Bayes' theorem** with a numeric prior — [Module 1 §3](modules/01-foundations.md#3-probability--statistics).
6. **Why accuracy lies under class imbalance** — [Module 2 §3](modules/02-protocols.md#3-evaluation-metrics).
7. **k-fold cross-validation** mechanics + stratification — [Module 2 §2](modules/02-protocols.md#2-data-partitioning-strategies).
8. **Perceptron + XOR** → why we need depth — [Module 5 §1](modules/05-deep.md#1-biological-motivation-and-the-perceptron).
9. **PCA** = max-variance directions; PCA vs. LDA vs. t-SNE — [Module 4 §2](modules/04-descriptive.md#2-dimensionality-reduction).
10. **The kernel trick** and the maximum-margin idea — [Module 3 §4](modules/03-shallow.md#4-geometric-methods).

---

## Hand-computations to practise until automatic

The exam may ask you to *calculate*, not just explain. Drill these five:

| Computation | Recipe | Where |
|---|---|---|
| Information gain | parent entropy − weighted child entropy; biggest wins | [M3 §5](modules/03-shallow.md#5-tree-based-methods) |
| Confusion-matrix metrics | precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = harmonic mean | [M2 §3](modules/02-protocols.md#3-evaluation-metrics) |
| Bayes' theorem | posterior = (likelihood × prior) / total prob. of evidence | [M1 §3](modules/01-foundations.md#3-probability--statistics) |
| Gradient-descent step | $w \leftarrow w - \eta\,\nabla\mathcal{L}$ | [M5 §3](modules/05-deep.md#3-backpropagation-and-gradient-based-optimisation) |
| k-means iteration | assign to nearest centroid → move centroid to mean → repeat | [M4 §1](modules/04-descriptive.md#1-clustering) |

---

## One-line formula sheet (cram card)

- **Entropy:** $H(S) = -\sum_k p_k \log_2 p_k$  (0 = pure, max = uniform)
- **Information gain:** $\text{Gain} = H(\text{parent}) - \sum_v \frac{|S_v|}{|S|} H(S_v)$
- **Precision / Recall:** $\frac{TP}{TP+FP}$ / $\frac{TP}{TP+FN}$ · **F1** = $\frac{2PR}{P+R}$
- **Accuracy:** $\frac{TP+TN}{n}$ (beware imbalance) · **R²** = $1 - \frac{SS_{res}}{SS_{tot}}$ (can be < 0)
- **Bayes:** $P(Y\mid X) = \frac{P(X\mid Y)P(Y)}{P(X)}$
- **Bias–variance:** $\mathbb{E}[\text{err}] = \text{Bias}^2 + \text{Var} + \sigma^2_\varepsilon$
- **Gradient descent:** $w \leftarrow w - \eta\,\nabla\mathcal{L}$
- **Variance explained (PCA):** $\lambda_j / \sum_k \lambda_k$ (keep components to ~95%)
- **Perceptron update:** $\mathbf{w} \leftarrow \mathbf{w} + \eta\,y_i\,\mathbf{x}_i$ (on a mistake)
- **Softmax / sigmoid:** multiclass / binary output activation

---

## Self-test routine (the night before)

1. Cover every **🔑 Quick Revision** table; reproduce it from memory.
2. Do all **✅ Self-Check** questions without expanding answers.
3. Re-derive the five **hand-computations** above on blank paper.
4. Run one pass of [flashcards.md](flashcards.md).
5. List, per method, its **inductive bias** and **key hyperparameter** ([Module 3 summary table](modules/03-shallow.md#classifier--regressor-summary)).

If you can do all five cold, you're ready.

---

*[← README](README.md) · [Glossary](glossary.md) · [Flashcards](flashcards.md)*
