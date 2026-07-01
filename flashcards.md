# Flashcards

> Spaced-repetition Q/A covering the whole course. Two ways to use:
> 1. **On GitHub/Obsidian:** read the question, answer in your head, expand to check.
> 2. **In Anki:** the `Question; Answer` table at the bottom imports directly (semicolon-separated, one card per row).

How to drill: go through once, star anything you miss, and re-test only the missed cards the next day. Repeat until the missed pile is empty.

---

## Module 1 — Foundations

<details><summary>What shape is the data matrix <strong>X</strong>, and what do rows/columns mean?</summary>

$\mathbf{X} \in \mathbb{R}^{n \times d}$ — $n$ rows (samples), $d$ columns (features).
</details>

<details><summary>State Bayes' theorem and name its three parts.</summary>

$P(Y\mid X) = \frac{P(X\mid Y)P(Y)}{P(X)}$ — posterior ∝ likelihood × prior, divided by evidence.
</details>

<details><summary>Write the bias–variance decomposition and define each term.</summary>

$\mathbb{E}[\text{err}] = \text{Bias}^2 + \text{Var} + \sigma^2_\varepsilon$. Bias = too simple (underfit); Variance = too sensitive (overfit); $\sigma^2_\varepsilon$ = irreducible noise.
</details>

<details><summary>Which models need feature standardisation and which don't?</summary>

Need it: k-NN, SVM/SVR, MLP, regularised linear (distance/gradient based). Don't: decision trees, Gaussian Naive Bayes.
</details>

<details><summary>MLE → which losses?</summary>

Gaussian noise → squared error (regression); Bernoulli/categorical → cross-entropy (classification).
</details>

<details><summary>SVD in one phrase.</summary>

Any matrix = rotate → stretch → rotate; biggest singular values = main directions (powers PCA).
</details>

---

## Module 2 — Experimental Protocols

<details><summary>Why never evaluate on the training set?</summary>

Training error is optimistically biased; it underestimates generalisation error, and the bias grows with model capacity.
</details>

<details><summary>How does k-fold CV work?</summary>

Split into $k$ folds; train $k$ times each leaving one fold out for testing; average the $k$ scores. Every sample is tested exactly once.
</details>

<details><summary>Precision vs. recall — definitions.</summary>

Precision = TP/(TP+FP) ("when I say positive, am I right?"). Recall = TP/(TP+FN) ("did I catch all positives?").
</details>

<details><summary>What is F1 and when do you prefer it over accuracy?</summary>

Harmonic mean of precision & recall: $\frac{2PR}{P+R}$. Prefer it under class imbalance, where accuracy is misleading.
</details>

<details><summary>When is stratified k-fold required?</summary>

For imbalanced classification — it keeps each fold's class proportions equal to the whole dataset.
</details>

<details><summary>Can R² be negative? Meaning?</summary>

Yes — the model predicts worse than outputting the target mean.
</details>

<details><summary>Which significance test for two methods vs. many methods?</summary>

Two methods, one dataset: paired t-test or Wilcoxon. Many methods, many datasets: Friedman + Nemenyi.
</details>

---

## Module 3 — Shallow Techniques

<details><summary>Formula for information gain.</summary>

Gain = Entropy(parent) − Σ (|S_v|/|S|)·Entropy(child_v). ID3 picks the attribute with the highest gain.
</details>

<details><summary>Entropy of a node with classes 6 Yes / 2 No (out of 8)?</summary>

$-\tfrac{6}{8}\log_2\tfrac{6}{8} - \tfrac{2}{8}\log_2\tfrac{2}{8} \approx 0.811$.
</details>

<details><summary>Bagging vs. boosting in one line each.</summary>

Bagging: parallel models on bootstrap samples, averaged → ↓variance. Boosting: sequential models each fixing prior errors → ↓bias.
</details>

<details><summary>Why does Random Forest subsample features at each split?</summary>

To decorrelate the trees; averaging decorrelated trees removes more variance than averaging correlated ones.
</details>

<details><summary>What is the kernel trick?</summary>

Implicitly computing dot products in a higher-dimensional space, letting a linear SVM form non-linear boundaries without computing the coordinates.
</details>

<details><summary>Naive Bayes core assumption and why smoothing is used.</summary>

Features are conditionally independent given the class. Laplace smoothing ($\alpha$) prevents a single zero likelihood from zeroing the whole product.
</details>

<details><summary>What makes an ensemble actually help?</summary>

Diversity — base learners must make different (uncorrelated) errors. Combining near-identical models adds cost, not accuracy.
</details>

<details><summary>What is logistic regression (despite the name)?</summary>

A probabilistic linear classifier: sigmoid of a linear score, trained by minimising cross-entropy (Bernoulli MLE).
</details>

---

## Module 4 — Descriptive Models

<details><summary>The two repeating steps of k-means.</summary>

Assign each point to the nearest centroid → move each centroid to the mean of its points → repeat until assignments stop changing.
</details>

<details><summary>k-means weaknesses (name two).</summary>

Needs $K$ up front; sensitive to initialisation; assumes spherical, equal-sized clusters.
</details>

<details><summary>When to use DBSCAN over k-means?</summary>

Arbitrary cluster shapes and/or outliers — DBSCAN is density-based and labels noise; no $K$ needed.
</details>

<details><summary>What does PCA maximise, and is it supervised?</summary>

The variance retained along orthogonal components; unsupervised (ignores labels). Components are eigenvectors of the covariance matrix.
</details>

<details><summary>PCA vs. LDA vs. t-SNE.</summary>

PCA = unsupervised max-variance (linear). LDA = supervised max class-separation (linear). t-SNE = non-linear visualisation, not reusable on new points.
</details>

<details><summary>Eigenvalues [6,2.5,1,0.4,0.1]: components to keep for ≥95% variance?</summary>

Cumulative: 60%, 85%, 95% → keep the first 3.
</details>

---

## Module 5 — Deep Techniques

<details><summary>Why can't a single perceptron learn XOR, and what fixes it?</summary>

XOR isn't linearly separable — no single line splits the classes. A hidden layer (stacked perceptrons) combines multiple lines into a non-linear boundary.
</details>

<details><summary>Forward pass vs. backward pass.</summary>

Forward: compute prediction + loss. Backward: chain-rule the gradient of the loss w.r.t. every weight (assign blame) for the optimiser to update.
</details>

<details><summary>Gradient-descent update rule.</summary>

$w \leftarrow w - \eta\,\nabla\mathcal{L}$. Too-large $\eta$ diverges; too-small crawls.
</details>

<details><summary>Activation for hidden layer / binary output / multiclass output?</summary>

ReLU / sigmoid / softmax.
</details>

<details><summary>What problem do LSTM gates solve?</summary>

Vanishing/exploding gradients over long sequences — the cell state + forget/input/output gates preserve information across many time steps.
</details>

<details><summary>Why do CNNs have far fewer parameters than fully connected nets on images?</summary>

Parameter sharing — one kernel slides over the whole image instead of separate weights per pixel.
</details>

<details><summary>One advantage and one cost of self-attention vs. RNN.</summary>

Advantage: parallel processing, one-step long-range links. Cost: $O(n^2)$ memory/compute in sequence length.
</details>

---

## Anki import block

Copy the rows below into Anki (File → Import → "Fields separated by: Semicolon"). Front `;` Back.

```
What shape is X and what are rows/cols?; X ∈ R^(n×d): n samples (rows), d features (cols)
State Bayes' theorem.; P(Y|X) = P(X|Y)P(Y)/P(X) — posterior ∝ likelihood × prior
Bias–variance decomposition?; E[err] = Bias^2 + Var + irreducible noise
Which models need feature scaling?; k-NN, SVM/SVR, MLP, regularised linear — NOT trees or Gaussian NB
MLE maps to which losses?; Gaussian→MSE (regression); Bernoulli/categorical→cross-entropy
Why never test on the training set?; Training error is optimistically biased; grows with capacity
How does k-fold CV work?; Train k times leaving one fold out each; average k scores
Precision vs recall?; Precision=TP/(TP+FP); Recall=TP/(TP+FN)
F1 score?; Harmonic mean of precision and recall = 2PR/(P+R)
When is stratified k-fold required?; Imbalanced classification — preserves per-fold class ratios
Can R² be negative?; Yes — model worse than predicting the mean
Test for two methods vs many?; Two: paired t-test/Wilcoxon; Many: Friedman + Nemenyi
Information gain formula?; Gain = H(parent) − Σ weighted H(children); ID3 picks max gain
Bagging vs boosting?; Bagging: parallel, averaged → ↓variance. Boosting: sequential, fix errors → ↓bias
Why subsample features in Random Forest?; Decorrelates trees → averaging removes more variance
What is the kernel trick?; Implicit dot products in higher-D space → non-linear SVM boundary
Naive Bayes assumption + why smoothing?; Conditional feature independence; smoothing avoids zero-probability wipeout
What makes an ensemble help?; Diversity — base learners must make uncorrelated errors
What is logistic regression?; Probabilistic linear classifier: sigmoid of linear score, cross-entropy loss
Two steps of k-means?; Assign to nearest centroid; move centroid to mean; repeat
When DBSCAN over k-means?; Arbitrary shapes / outliers; no K needed
What does PCA maximise? Supervised?; Variance along orthogonal components; unsupervised
PCA vs LDA vs t-SNE?; PCA=unsup variance; LDA=sup class-separation; t-SNE=nonlinear viz (not reusable)
Why can't a perceptron learn XOR? Fix?; Not linearly separable; add a hidden layer
Forward vs backward pass?; Forward: predict+loss; Backward: chain-rule gradients (assign blame)
Gradient-descent update?; w ← w − η∇L; big η diverges, small η crawls
Activations: hidden/binary/multiclass?; ReLU / sigmoid / softmax
What do LSTM gates solve?; Vanishing/exploding gradients over long sequences
Why fewer params in CNN vs FC net?; Parameter sharing — one kernel slides over the whole image
Self-attention advantage/cost vs RNN?; +parallel, long-range in one step; −O(n²) compute
```

---

*[← README](README.md) · [Study Guide](study-guide.md) · [Glossary](glossary.md)*
