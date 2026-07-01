# Module 4 — Descriptive Models

> **Colour:** ![#bc8cff](https://placehold.co/12x12/bc8cff/bc8cff.png) Violet  
> **Prerequisites:** [Module 1 — Foundations](01-foundations.md) (linear algebra, probability, distance metrics)  
> **Connects to:** [Module 3 — Shallow Techniques](03-shallow.md) · [Module 5 — Deep Techniques](05-deep.md)

---

## Overview

Descriptive (unsupervised) models discover structure in data without labelled examples. Rather than optimising a prediction target, they seek to reveal the geometry of the data distribution: compact groups (clustering), low-dimensional manifolds (dimensionality reduction), and latent generative factors (density estimation).

These techniques are foundational in exploratory data analysis, feature engineering, data compression, and as pre-processing stages for supervised pipelines. In the research context of this study plan, unsupervised methods are directly applicable to network traffic segmentation, anomaly detection without labelled attack data, and topology analysis in SDN environments.

---

## How to Use This Module (Exam Prep)

**After this module you should be able to:**
- Explain what "unsupervised" means and when you'd use it (no labels).
- Run one iteration of **k-means** by hand and state its assumptions/weaknesses.
- Compare **k-means vs. hierarchical vs. DBSCAN** — what each can and can't do.
- Explain **PCA** as variance maximisation and compute proportion of variance explained.
- Distinguish **PCA vs. LDA** (unsupervised vs. supervised) and **PCA vs. t-SNE** (linear vs. non-linear, reusable vs. visualisation-only).

**⭐ High-yield for exams:** k-means algorithm + its assumptions · choosing $K$ (elbow/silhouette) · DBSCAN's core/border/noise points · PCA = eigenvectors of the covariance · PCA vs. LDA vs. t-SNE differences.

**If you only read one thing:** §1 Clustering (k-means mechanics) and the **PCA** part of §2.

**Suggested time:** ~75 min reading + ~40 min on the [Worked Examples](#worked-examples) and [Self-Check](#self-check).

---

## 1. Clustering

Clustering partitions a dataset $\mathcal{D} = \{\mathbf{x}_i\}_{i=1}^n$ into groups such that intra-cluster similarity is high and inter-cluster similarity is low. The definition of "similarity" is determined by the algorithm's inductive bias.

### k-Means

Minimises total within-cluster variance (sum of squared distances to centroids):

$$\min_{\{\mu_k\}, \{C_k\}} \sum_{k=1}^K \sum_{\mathbf{x}_i \in C_k} \|\mathbf{x}_i - \boldsymbol{\mu}_k\|^2$$

**Algorithm (Lloyd's):**
1. Initialise $K$ centroids (random, or k-means++ for better initialisation).
2. Assign each point to its nearest centroid.
3. Recompute centroids as the mean of assigned points.
4. Repeat steps 2–3 until convergence.

**Key hyperparameters:** $K$ (number of clusters), initialisation strategy, distance metric.

> **💡 Intuition:** k-means alternates two cheap steps until nothing moves: **(1) assign** each point to the nearest centroid, **(2) move** each centroid to the mean of its points. Like dropping $K$ flags, sending each person to their closest flag, then moving each flag to the middle of its crowd — repeat. It only finds *spherical, equal-sized* blobs, and the result depends on where you drop the flags.

> **📝 Worked Example — one k-means iteration ($K=2$)**
> 1-D points: $\{1, 2, 3, 10, 11, 12\}$. Initial centroids $\mu_1 = 1$, $\mu_2 = 11$.
> 1. **Assign** (nearest centroid): $\{1,2,3\} \to \mu_1$ (distances 0,1,2 beat 10,9,8); $\{10,11,12\} \to \mu_2$.
> 2. **Update**: $\mu_1 = \text{mean}(1,2,3) = \mathbf{2}$; $\mu_2 = \text{mean}(10,11,12) = \mathbf{11}$.
> 3. **Re-assign**: no point changes cluster → **converged**. Final clusters $\{1,2,3\}$ and $\{10,11,12\}$.
> Try $\mu_1=1, \mu_2=2$ instead and you reach a *worse* solution — proof that **initialisation matters** (hence k-means++ and multiple restarts).

**Common pitfalls:** Sensitivity to initialisation (mitigated by k-means++, multiple restarts); assumes spherical clusters of equal size; $K$ must be specified a priori. The **elbow method** and **silhouette score** are standard heuristics for selecting $K$.

### Hierarchical Clustering

Builds a dendrogram — a tree of nested clusters — without requiring $K$ in advance. Two strategies:

- **Agglomerative (bottom-up):** Start with $n$ singleton clusters; merge the two most similar clusters at each step.
- **Divisive (top-down):** Start with one cluster containing all points; recursively split.

**Linkage criteria** define how inter-cluster distance is computed:

| Linkage | Distance between clusters $A$, $B$ |
|---|---|
| Single | $\min_{a \in A, b \in B} d(a, b)$ — susceptible to chaining |
| Complete | $\max_{a \in A, b \in B} d(a, b)$ — compact, sensitive to outliers |
| Average (UPGMA) | $\frac{1}{|A||B|} \sum_{a,b} d(a,b)$ — balanced |
| Ward | Minimises total within-cluster variance after merge |

### DBSCAN

Density-based clustering: clusters are contiguous regions of high density, separated by low-density regions.

- **Core point:** has at least $\text{MinPts}$ neighbours within radius $\varepsilon$.
- **Border point:** within $\varepsilon$ of a core point but not itself a core point.
- **Noise point:** neither core nor border — treated as outlier.

**Key hyperparameters:** $\varepsilon$ (neighbourhood radius), $\text{MinPts}$ (density threshold).

**Advantages over k-means:** Discovers arbitrary cluster shapes; automatically identifies outliers; does not require $K$.

### Cluster Validity Indices

| Index | Type | Formula sketch | Interpretation |
|---|---|---|---|
| Silhouette | Internal | $s_i = \frac{b_i - a_i}{\max(a_i, b_i)}$ | $\in [-1, 1]$; higher is better |
| Davies-Bouldin | Internal | Mean ratio of within-cluster to between-cluster distance | Lower is better |
| Adjusted Rand Index | External | Agreement with ground truth labels, corrected for chance | $\in [-1, 1]$; 1 is perfect |

---

## 2. Dimensionality Reduction

High-dimensional data is expensive to model, visualise, and store. Dimensionality reduction maps $\mathbf{x} \in \mathbb{R}^d$ to $\mathbf{z} \in \mathbb{R}^q$ with $q \ll d$, ideally preserving the structure relevant to downstream tasks.

### Principal Component Analysis (PCA)

Finds the $q$ orthogonal directions of maximum variance in the data. Given centred data matrix $\mathbf{X}$, the covariance matrix $\mathbf{C} = \frac{1}{n-1}\mathbf{X}^\top\mathbf{X}$ is decomposed as $\mathbf{C} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^\top$. The first $q$ eigenvectors (columns of $\mathbf{V}$) define the projection:

$$\mathbf{Z} = \mathbf{X} \mathbf{V}_q$$

> **💡 Intuition:** PCA finds new axes (principal components) ordered by how much the data *spreads* along them. Component 1 is the single direction of greatest variance; component 2 is the next, at right angles; and so on. Keep the first few and you compress the data while losing as little spread (information) as possible. The eigenvalue $\lambda_j$ literally *is* the variance captured by component $j$.

**Proportion of variance explained** by component $j$: $\frac{\lambda_j}{\sum_k \lambda_k}$. The **scree plot** shows this as a function of $j$.

> **📝 Worked Example — how many components to keep?**
> A 5-feature dataset gives covariance eigenvalues $\lambda = [6.0,\ 2.5,\ 1.0,\ 0.4,\ 0.1]$ (total = 10).
> - PC1 explains $6.0/10 = \mathbf{60\%}$; PC2 adds $2.5/10 = 25\%$ → cumulative **85%**; PC3 adds 10% → **95%**.
> - To retain **≥ 95% variance**, keep the **first 3 components** — reducing 5 dimensions to 3 while discarding only the last 5% (likely noise). That threshold rule is the standard exam answer for "how many components?"

**Key hyperparameters:** $q$ (number of components) or explained variance threshold (e.g., 95%).

**Common pitfall:** PCA is a linear method — it cannot capture non-linear manifold structure. For non-linear data, t-SNE or UMAP are preferred.

### Linear Discriminant Analysis (LDA)

Supervised dimensionality reduction: finds the projection that maximises between-class scatter relative to within-class scatter.

$$\mathbf{w}^* = \arg\max_\mathbf{w} \frac{\mathbf{w}^\top \mathbf{S}_B \mathbf{w}}{\mathbf{w}^\top \mathbf{S}_W \mathbf{w}}$$

where $\mathbf{S}_B$ is the between-class scatter matrix and $\mathbf{S}_W$ is the within-class scatter matrix. The maximum number of discriminant components is $\min(d, K-1)$ where $K$ is the number of classes.

### t-SNE

Stochastic Neighbour Embedding with Student-$t$ kernel: a non-linear technique for 2D/3D visualisation of high-dimensional data. Constructs a probability distribution over pairwise distances in the original space and optimises a lower-dimensional embedding to match it via KL divergence minimisation.

**Key hyperparameters:** Perplexity (controls effective neighbourhood size), learning rate, number of iterations.

**Limitation:** Not suitable as a pre-processing step for supervised models — the embedding is not deterministic and does not generalise to new points. Use for exploration and visualisation only.

### UMAP

Uniform Manifold Approximation and Projection: a manifold learning method that is faster than t-SNE, scales to larger datasets, and (unlike t-SNE) can be used for supervised dimensionality reduction and generalises to new data.

> **💡 Intuition — PCA vs. LDA vs. t-SNE (exam favourite):** **PCA** = unsupervised, keeps directions of max *variance*. **LDA** = supervised, keeps directions that best *separate classes*. **t-SNE/UMAP** = non-linear, for *visualising* clusters in 2-D — t-SNE can't be reused on new points, UMAP can.

---

## Worked Examples

Collected for revision: **one k-means iteration** (§1) and **PCA variance-explained / component selection** (§2). Re-run the k-means assign→update loop from scratch — that two-step cycle is the most likely hand-trace on the exam.

---

## Self-Check

<details>
<summary><strong>Q1.</strong> What two steps does k-means alternate, and when does it stop? (§1)</summary>

**Assign** each point to its nearest centroid, then **update** each centroid to the mean of its assigned points. It stops when assignments no longer change (convergence).
</details>

<details>
<summary><strong>Q2.</strong> Your clusters are crescent-shaped (non-spherical) with some outliers. k-means or DBSCAN? (§1)</summary>

**DBSCAN** — it finds arbitrary shapes by density and labels outliers as noise. k-means assumes round, equal-sized blobs and forces every point into a cluster.
</details>

<details>
<summary><strong>Q3.</strong> Eigenvalues [4, 3, 2, 1]. What % of variance do the first two components explain? (§2)</summary>

Total = 10; first two = 4 + 3 = 7 → **70%**.
</details>

<details>
<summary><strong>Q4.</strong> You have class labels and want a projection that separates classes best. PCA or LDA? (§2)</summary>

**LDA** — it's supervised and maximises between-class vs. within-class scatter. PCA ignores labels and only maximises total variance.
</details>

<details>
<summary><strong>Q5.</strong> Why should you not feed a t-SNE embedding into a classifier as features? (§2)</summary>

t-SNE is non-deterministic and **does not generalise to new points** — there's no reusable mapping. It's for visualisation only; use PCA or UMAP for pre-processing.
</details>

---

## 🔑 Quick Revision

| Concept | One-line takeaway |
|---|---|
| k-means | assign → update → repeat; spherical clusters; needs $K$; init-sensitive |
| Choosing $K$ | elbow method + silhouette score |
| Hierarchical | dendrogram, no $K$ upfront; linkage = single/complete/average/Ward |
| DBSCAN | density-based; finds any shape; flags noise; knobs $\varepsilon$, MinPts |
| PCA | eigenvectors of covariance; max variance; **unsupervised**; linear |
| LDA | max class separation; **supervised**; ≤ $K-1$ components |
| t-SNE / UMAP | non-linear visualisation; t-SNE viz-only, UMAP reusable |
| Variance explained | $\lambda_j / \sum_k \lambda_k$; keep components to reach ~95% |

---

## Professor's References

> Full entries in [`../references.md`](../references.md).

- BISHOP, C. M. *Pattern Recognition and Machine Learning*. Springer, 2006. — Chapter 9 (mixture models and EM), Chapter 12 (PCA and probabilistic PCA).
- DUDA, R. O.; HART, P. E.; STORK, D. G. *Pattern Classification*. 2. ed. Wiley-Interscience, 2000. — Chapter 10 (unsupervised learning and clustering).

---

## Extended Reading

- JAIN, A. K.; MURTY, M. N.; FLYNN, P. J. Data clustering: a review. *ACM Computing Surveys*, v. 31, n. 3, p. 264–323, 1999. — Comprehensive survey of clustering algorithms.
- ESTER, M. et al. A density-based algorithm for discovering clusters in large spatial databases with noise. In: *KDD*, 1996. p. 226–231. — Original DBSCAN paper.
- VAN DER MAATEN, L.; HINTON, G. Visualizing data using t-SNE. *Journal of Machine Learning Research*, v. 9, p. 2579–2605, 2008. Available at: <https://www.jmlr.org/papers/v9/vandermaaten08a.html>.
- McINNES, L.; HEALY, J.; MELVILLE, J. UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv*, 2018. Available at: <https://arxiv.org/abs/1802.03426>.
- FRIEDMAN, J.; HASTIE, T.; TIBSHIRANI, R. *The Elements of Statistical Learning*. 2. ed. Springer, 2009. — Chapter 14 (unsupervised learning). Available at: <https://hastie.su.domains/ElemStatLearn/>.

---

## Connected Activities

No activity in the current course sequence directly targets Module 4. However, dimensionality reduction (PCA) is a natural pre-processing step for the classification task in Atividade 1 Part A and may be explored as an extension.

---

*[← Module 3 — Shallow Techniques](03-shallow.md) · [README](../README.md) · [Module 5 — Deep Techniques →](05-deep.md)*
