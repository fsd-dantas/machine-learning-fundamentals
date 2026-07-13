# Machine Learning — Research Fundamentals Study Plan

> PPGIA / PUC-PR · MSc 2026 · Aprendizagem de Máquina  
> Fernando Dantas · [fsd-dantas.github.io/machine-learning-fundamentals](https://fsd-dantas.github.io/machine-learning-fundamentals)

<p align="center">
  <a href="https://fsd-dantas.github.io/machine-learning-fundamentals/pages/mapa-mental.html"><strong>Open the interactive mind map</strong></a>
</p>

---

> ### 📖 Preparing for the exam? **Start with the [Study Guide](study-guide.md).**
> It gives a module-by-module study order, time budget, the highest-yield topics, the hand-computations to practise, and a one-page formula sheet. Companion files: **[Glossary](glossary.md)** · **[Flashcards](flashcards.md)**.
>
> Every module now includes *Learning objectives*, *💡 Intuition* boxes, *📝 Worked Examples* (real numbers), a *✅ Self-Check* (collapsible answers), and a *🔑 Quick Revision* cheat sheet.

---

## Concept & Framing

This repository is a **personal research roadmap** for the foundational concepts of Machine Learning, built around the structure of the discipline as taught by Prof. Alceu de Souza Britto Jr. at PPGIA/PUC-PR, but reframed as a self-directed learning and research reference.

The guiding philosophy borrows from dependency-graph learning resources (e.g., [Metacademy](https://metacademy.org), [roadmap.sh/machine-learning](https://roadmap.sh/machine-learning)) and knowledge-atlas projects (e.g., [mrdbourke/machine-learning-roadmap](https://github.com/mrdbourke/machine-learning-roadmap)): **every concept has prerequisites, and every activity connects back to a cluster of foundational ideas**.

This site exists to:

1. Map the ML concept space as a navigable knowledge graph — following the canonical structure below
2. Provide curated references per concept cluster — professor-assigned and beyond
3. Document hands-on experimental work — shallow models, deep vision, and NLP — under one reproducible protocol
4. Serve as a public-facing academic portfolio page for the MSc journey

**Language:** English throughout, to reach broader audiences and align with international research norms. Original Portuguese course materials are credited faithfully.

---

## Repository Structure

The repository is **Markdown-first**: Markdown files are the source of truth, GitHub renders them natively, and Obsidian can mirror them as a local vault. The static HTML pages are a presentation layer and should be generated from the Markdown content as the repository matures, so they do not become a competing source of truth.

```
/
├── README.md                        # This file — landing page and specification
├── modules/
│   ├── 01-foundations.md            # Fundamentos (math, geometry, probability)
│   ├── 02-protocols.md              # Experimental Protocols (CV, metrics, design)
│   ├── 03-shallow.md                # Shallow Techniques (classifiers & regressors)
│   ├── 04-descriptive.md            # Descriptive Models (clustering, dim-reduction)
│   └── 05-deep.md                   # Deep Techniques (MLP, CNN, RNN+)
├── activities/
│   ├── atividade-1.md               # Activity 1 — Shallow models: classification & regression
│   ├── atividade-1.py               # Reproducible experiment script (regenerates all results)
│   ├── avaliacao-pratica-1.md       # Practical 1 — CIFAR-10: five deep-learning strategies
│   ├── avaliacao-pratica-1/         # Protocol, 5 strategy scripts, 4 ablations, report generator
│   ├── avaliacao-pratica-2.md       # Practical 2 — Text classification: LSTM vs Transformer
│   ├── avaliacao-pratica-2/         # Protocol, LSTM, BERTimbau, classical baseline, report generator
│   └── *-respostas.md               # pt-BR submissions (rendered to PDF for hand-in)
├── mind-map.json                    # SOURCE OF TRUTH for the concept graph (nodes, edges, prerequisites)
├── mind-map.md                      # Markdown mind map (Mermaid) — generated from mind-map.json
├── scripts/
│   ├── build_mindmap.py             # Regenerates mind-map.md + the HTML graph data from mind-map.json
│   ├── build_pdf.py                 # Markdown → PDF (pandoc + headless Chrome) for submissions
│   └── validate_activity_results.py # Checks generated results against the Markdown report
├── .github/workflows/
│   └── validate.yml                 # CI: dependencies + reproducibility checks
├── assets/
│   └── img/
│       ├── mindmap.png              # Legacy static mind map — dark variant
│       └── mindmap-light.png        # Legacy static mind map — light variant
├── study-guide.md                   # Exam-prep plan: study order, time budget, high-yield topics
├── glossary.md                      # Every acronym + core term, one line each
├── flashcards.md                    # Spaced-repetition Q/A (Anki-importable)
├── references.md                    # Full ABNT bibliography (professor's + extended)
└── requirements.txt                 # Python dependencies for reproducible activities
```

> **Rendering targets:** GitHub (primary) and Obsidian (secondary). The Markdown files are canonical; the interactive HTML view should follow the mind-map structure below.

---

## Mind Map Structure

Three views of the same concept graph, all generated from **one source of truth, [`mind-map.json`](mind-map.json)**, by [`scripts/build_mindmap.py`](scripts/build_mindmap.py) — so they can never drift:

1. the ASCII tree below (quick reference, hand-maintained);
2. the **[Markdown mind map](mind-map.md)** — Mermaid, renders on GitHub/Obsidian, and adds a prerequisite dependency graph;
3. the interactive **[`pages/mapa-mental.html`](pages/mapa-mental.html)** — JavaScript/SVG presentation layer.

Edit `mind-map.json`, run `python scripts/build_mindmap.py`, and both view 2 and view 3 update together (CI enforces this with `--check`).

```text
Aprendizagem de Máquina
├── Fundamentos
│   ├── Álgebra Linear
│   ├── Geometria
│   ├── Probabilidade & Estatística
│   ├── Bayes / MLE
│   └── Viés / Variância
├── Protocolos Experimentais
│   ├── Holdout / K-Fold CV
│   ├── Estratificação
│   ├── Métricas
│   ├── Matriz de Confusão
│   ├── Pipeline sem Vazamento
│   └── Wilcoxon / Friedman
├── Técnicas Rasas Preditivas
│   ├── Linear / Ridge / Lasso
│   ├── Regressão Logística
│   ├── k-NN / Naive Bayes
│   ├── SVM / SVR
│   ├── Árvores / ID3 / CART
│   ├── Ensembles
│   └── MLP Baseline
├── Modelos Descritivos
│   ├── K-Means
│   ├── Hierárquico
│   ├── DBSCAN
│   ├── Silhueta / Davies-Bouldin
│   ├── PCA / LDA
│   └── t-SNE / UMAP
└── Técnicas Profundas
    ├── Perceptron / XOR
    ├── Backprop / Otimização
    ├── MLP
    ├── CNN
    ├── RNN / LSTM / GRU
    ├── Redes Siamesas
    ├── Transformer
    └── Generativos / GPT / VAE / GAN
```

---

## Module Map

Each module is a self-contained Markdown file that follows a consistent content schema (detailed below). They map directly to the five top-level clusters in the mind map.

| # | Module | Theme | File |
|---|--------|-------|------|
| 1 | **Foundations** | Mathematics, geometry, probability theory | [`modules/01-foundations.md`](modules/01-foundations.md) |
| 2 | **Experimental Protocols** | Cross-validation, evaluation metrics, experimental design | [`modules/02-protocols.md`](modules/02-protocols.md) |
| 3 | **Shallow Techniques** | Supervised classifiers and regressors | [`modules/03-shallow.md`](modules/03-shallow.md) |
| 4 | **Descriptive Models** | Unsupervised learning, clustering, dimensionality reduction | [`modules/04-descriptive.md`](modules/04-descriptive.md) |
| 5 | **Deep Techniques** | MLP, CNN, RNN, modern architectures | [`modules/05-deep.md`](modules/05-deep.md) |

---

## Content Schema (per module file)

Every module file follows this structure, in order:

### 1. Concept Overview
A concise paragraph establishing what the module covers and how it connects to the broader ML landscape. Written at MSc level — assumes mathematical literacy.

### 2. Prerequisites
A short dependency list: concepts the reader should be comfortable with before entering this module. Links to the relevant preceding module file or external references.

### 3. Key Concepts & Techniques
A structured breakdown of the main ideas — presented as a scannable reference, not lecture notes. Each entry includes:
- Core definition (1–2 sentences)
- Formal notation where applicable (LaTeX inline or block)
- Key hyperparameters / design decisions
- Common pitfalls

### 4. Professor's References
Exact bibliographic entries from Prof. Alceu de Souza Britto Jr.'s syllabus, formatted in ABNT. Preserved verbatim as the authoritative reading list for this course. Linked to `references.md` for the full list.

### 5. Extended Reading
Curated external references beyond the syllabus — landmark papers, textbook chapters, open-access resources — selected for research-grade depth.

### 6. Connected Activities
Links to hands-on work in `/activities/` that exercise concepts from this module.

### Learning-Support Sections (exam prep)
Every module also carries a consistent set of pedagogical aids:
- **How to Use This Module** — learning objectives, ⭐ high-yield flags, "if you only read one thing," and a time budget.
- **💡 Intuition** — plain-language explanation placed *before* each heavy formula.
- **📝 Worked Examples** — formulas applied to real numbers, step by step.
- **✅ Self-Check** — questions with collapsible (`<details>`) answers.
- **🔑 Quick Revision** — a single cheat-sheet table per module.

---

## Index Page (this README)

`README.md` serves as the **concept graph entry point**. Beyond the mind map and module table above, it includes:

- **Mission statement** — why this study plan exists (above)
- **Research context** — how ML connects to the author's SDN/Smart Grid/VNF thesis work (below)
- **Activities index** — compact list of submitted work linking to `/activities/`
- **References** — link to the full `references.md`

---

## Scope & Boundaries

This roadmap covers the **supervised and unsupervised foundations** of classical and deep ML, in the sequence taught in the course. To keep it a faithful reference rather than an exhaustive textbook, the following are **deliberately out of scope** and noted here so readers know what they will *not* find:

- **Reinforcement learning** (MDPs, policy/value methods) — a distinct paradigm, not part of this course.
- **Probabilistic graphical models** (Bayesian networks, HMMs, CRFs) beyond the Naive Bayes special case.
- **Causal inference** and experimental causality.
- **MLOps / deployment** (serving, monitoring, drift) — referenced only where it touches the author's SDN research.

Within scope, the canonical baselines are covered explicitly: **linear and logistic regression** ([Module 3 §1](modules/03-shallow.md#1-linear-models)) anchor the model hierarchy, and the **perceptron** ([Module 5 §1](modules/05-deep.md#1-biological-motivation-and-the-perceptron)) anchors the neural one.

---

## Research Context

Fernando Dantas's MSc research at PPGIA/PUC-PR focuses on **Software-Defined Networking (SDN)**, **Smart Grid** orchestration, and **Virtual Network Functions (VNF)** management — primarily on the [ONOS](https://opennetworking.org/onos/) platform. Machine Learning is studied here as a foundational tool for intelligent network management. Concrete application areas:

| ML Technique | Research Application |
|---|---|
| Supervised classification | Anomaly detection in SDN traffic flows |
| Regression | Power-load forecasting in Smart Grid scenarios |
| Clustering | Network slice segmentation and topology grouping |
| RNN / LSTM | Time-series prediction for VNF auto-scaling decisions |

This framing is intentional: every technique studied in this course is evaluated not only on the benchmark datasets assigned in class, but also against its potential utility in the SDN/Smart Grid domain.

---

## Activities

### Atividade 1 — Shallow Models: Classification & Regression

Full write-up: [`activities/atividade-1.md`](activities/atividade-1.md)

**Part A — Classification (Breast Cancer Wisconsin)**
- 569 instances · 30 continuous attributes · Binary target (0 = malignant, 1 = benign)
- Classifiers: Logistic Regression *(linear baseline)*, Decision Tree, k-NN, Naive Bayes, SVM, MLP, Random Forest, Bagging, AdaBoost, XGBoost
- Protocol: stratified 5-fold cross-validation · Metrics: Accuracy, F1, Precision, Recall
- Result: SVM/MLP tie at **0.977** accuracy (difference not significant, Wilcoxon $p=0.875$); the linear baseline beats every tree ensemble

**Part B — Regression (Diabetes dataset)**
- 442 instances · 10 numeric attributes · Continuous target (range 25–346)
- Regressors: Ridge *(linear baseline)*, Decision Tree, k-NN, SVR, MLP, Random Forest, Bagging, XGBoost
- Protocol: 5-fold cross-validation · Metrics: R², MAE
- Result: Ridge wins R² (**0.479**) while MLP wins MAE (**44.05**) — a deliberate illustration of metric-dependent ranking

> Results are fully reproducible: [`activities/atividade-1.py`](activities/atividade-1.py) (`SEED = 42`, leakage-safe `Pipeline`) regenerates every number and answer.

---

### Avaliação Prática 1 — CIFAR-10: Five Deep-Learning Strategies

Full write-up: [`activities/avaliacao-pratica-1.md`](activities/avaliacao-pratica-1.md) · Code: [`avaliacao-pratica-1/`](activities/avaliacao-pratica-1/) · Runner: [`colab.ipynb`](activities/avaliacao-pratica-1/colab.ipynb)

One CNN trained from scratch, one frozen ImageNet backbone with a shallow classifier, one
fine-tuned CNN, the same with data augmentation, and a fine-tuned ViT — all on an **equal
budget**: the same stratified 10,000-image subsample, the same validation set, and a single
scoring pass on the full official 10,000-image test set.

The equal budget is what makes the comparison legal. The five strategies differ ~50× in cost,
and a free-tier T4 cannot afford all of them on the full 50,000 images. Given that, the choice
is between *all strategies on 10,000 images* and *some strategies on more data than others* —
and only the first is an experiment. The consequence is declared rather than buried: 10,000
images is a low-data regime, which is exactly where transfer learning is strongest.

Four **controlled ablations** answer the open questions — one variable moves, everything else
is pinned: backbone swap (MobileNetV2 / ResNet50 / InceptionV3), `Flatten()` vs
`GlobalMaxPooling2D()` vs `GlobalAveragePooling2D()`, optimiser, and augmentation policy
(including the lecture notebook's own, which omits horizontal flip — the single most effective
label-preserving transform on CIFAR-10).

> **Status:** core strategies complete; ablations running. Every claim is repeated over 3 seeds
> and the top pairs are tested with **exact McNemar** on paired test predictions. Results land
> in [`avaliacao-pratica-1/results/`](activities/avaliacao-pratica-1/results/) as JSON carrying
> their own 10,000 test predictions, so the tables, the confusion matrix and the significance
> tests regenerate on CPU with no retraining.

---

### Avaliação Prática 2 — Text Classification: LSTM vs Transformer

Full write-up: [`activities/avaliacao-pratica-2.md`](activities/avaliacao-pratica-2.md) · Code: [`avaliacao-pratica-2/`](activities/avaliacao-pratica-2/) · Runners: [`colab.ipynb`](activities/avaliacao-pratica-2/colab.ipynb) · [`kaggle.ipynb`](activities/avaliacao-pratica-2/kaggle.ipynb)

Two tasks over 2,436 Portuguese news headlines annotated with emotions — binary valence and
7-class emotion — comparing an **LSTM**, a fine-tuned **BERTimbau**, and a classical
**TF-IDF + linear SVM** baseline. Protocol: stratified 70/30 holdout, 3 seeds, exact McNemar
on paired predictions.

**BERT wins both tasks with significance** (0.829 binary, 0.574 multiclass; $p < 10^{-5}$
against the LSTM). But the classical baseline — **36k parameters, one second of CPU** —
*significantly beats the 912k-parameter LSTM on both tasks* and covers **46%** of the
LSTM→BERT gap. A model with no attention and no pretraining cannot do that if architecture is
what decides. The result suggests **prior linguistic knowledge**, not attention, carries the
substantial share of the advantage — stated as the indirect inference it is, since the control
that would prove it (a Transformer trained *without* pretraining) is not part of the design.

Three corpus findings preceded any training, and each changed the numbers:

- **292 duplicates (10.7%)** across the two files. De-duplication happens *before* the split —
  otherwise a text and its copy straddle train and test, and the model is scored on what it
  memorised.
- **Four texts carry conflicting labels, and every conflict is negative-vs-negative.**
  Annotators never disagree about *valence*, only about *which* negative emotion. The binary
  task is therefore clean, while the multiclass task carries **irreducible label noise** — and
  the confusion matrix bleeds exactly there, as predicted in writing before the models ran.
- The **majority-class baseline** (57.6% / 24.8%) is quoted beside every accuracy. Without it,
  a 62% binary classifier reads as a result rather than as arithmetic.

---

## Image Pattern Reference

All adaptive images in this repository use the following pattern, supporting both GitHub's native `prefers-color-scheme` and Obsidian CSS snippet hooks:

```html
<p align="center">
  <picture class="github-mode-only">
    <source media="(prefers-color-scheme: dark)" srcset="assets/img/example-dark.png">
    <img src="assets/img/example-light.png" alt="Description">
  </picture>
  <img class="obsidian-light-only" width="0" height="0" src="assets/img/example-light.png" alt="Description">
  <img class="obsidian-dark-only" width="0" height="0" src="assets/img/example-dark.png" alt="Description">
</p>
```

- `class="github-mode-only"` — rendered by GitHub, hidden in Obsidian via CSS snippet
- `obsidian-light-only` / `obsidian-dark-only` — rendered by Obsidian snippet, hidden on GitHub
- The `width="0" height="0"` trick makes the fallback images invisible on GitHub while remaining accessible to Obsidian's renderer

---

## Rendering & Tooling

| Context | How it renders |
|---|---|
| **GitHub** (primary) | Native Markdown + HTML subset; `<picture>` adaptive images; LaTeX via MathJax (enabled in repo settings) |
| **Obsidian** (secondary) | Local vault mirror; CSS snippet required for `obsidian-light-only` / `obsidian-dark-only` class hooks |
| **VS Code** | Markdown Preview with standard GFM rendering |

> **LaTeX in GitHub Markdown:** GitHub renders LaTeX math when delimited with `$...$` (inline) or `$$...$$` (block) in `.md` files, provided the repository has math rendering enabled. This repository uses that syntax for all formal notation.

---

## Experimental Discipline

The three deep-learning activities share a protocol, and it is deliberate rather than
incidental. Each has a `common.py` that owns the splits, the metrics and the statistics, and
each persists **the test predictions themselves** into its result JSONs — which is what makes
the tables, confusion matrices and significance tests regenerable on a CPU, by anyone, with no
GPU and no retraining. It is also what lets a PyTorch model and a Keras model stay comparable:
the protocol lives in the data and the evaluation, not in the framework.

Four rules are applied without exception:

| Rule | Why |
|---|---|
| **The test set is read once per configuration.** | Model selection happens on validation. A number tuned on test is not a measurement. |
| **No claim rests on a single run.** | Every trained model runs at 3 seeds; a gap smaller than the seed-to-seed spread is not a result. |
| **Paired comparisons use exact McNemar.** | Models are scored on *identical* samples, so their errors are paired; only the discordant predictions carry information. Differences that fail the test are reported as **technical ties**, not wins. |
| **Cost is a reported variable.** | Accuracy without training time and parameter count invites celebrating a 0.3 pp win that cost 50× the compute. |

Where a lecture notebook's methodology is departed from, the departure is stated and its effect
quantified — not silently corrected.

---

## Roadmap

- [x] Implement Atividade 1 as a reproducible script and report full results + analysis
- [x] Add exam-prep layer: worked examples, intuition boxes, self-checks, study guide, glossary, flashcards
- [x] Convert mind map from static PNG to an interactive SVG/HTML view
- [x] Avaliação Prática 2 — LSTM vs Transformer, with a classical baseline that reframes the result
- [ ] Avaliação Prática 1 — CIFAR-10; core strategies done, ablations running
- [ ] Unify the two `common.py` protocol modules into one shared package (duplicated on purpose while both activities were in flight)
- [ ] Open a dedicated SDN/Smart Grid research repo and link from the Research Context section
- [ ] Add future activity write-ups to `/activities/` as the course progresses

---

## Reference Sites & Inspirations

| Resource | Design principle borrowed |
|---|---|
| [roadmap.sh/machine-learning](https://roadmap.sh/machine-learning) | Step-by-step concept ordering; community-curated dependency structure |
| [Metacademy](https://metacademy.org) | Prerequisites-first navigation; "learn X to understand Y" framing |
| [learney.me](https://app.learney.me/) | Skill-tree visual layout; concept clustering |
| [mrdbourke/machine-learning-roadmap](https://github.com/mrdbourke/machine-learning-roadmap) | Connecting concepts ↔ tools ↔ math ↔ resources in one view |
| [bishwaghimire/ai-learning-roadmaps](https://github.com/bishwaghimire/ai-learning-roadmaps) | Modular research-grade progression: foundations → specialisation → research |
| [prathyvsh/knowledge-atlases](https://github.com/prathyvsh/knowledge-atlases) | Catalogue framing; knowledge maps as navigable atlases |
| [imteekay/machine-learning-research](https://github.com/imteekay/machine-learning-research) | Research-oriented README structure; annotated reading lists |

---

*Last updated: July 2026.*  
*Course: Aprendizagem de Máquina · PPGIA/PUC-PR · MSc 2026*  
*Instructor: Prof. Alceu de Souza Britto Jr. (alceu@ppgia.pucpr.br)*
