# Avaliação Prática 1 — Deep Models: CNN, Transfer Learning & Vision Transformers

> **Course:** Aprendizagem de Máquina · PPGIA / PUC-PR · MSc 2026
> **Instructor:** Prof. Alceu de Souza Britto Jr. (alceu@ppgia.pucpr.br)
> **Related module:** [Module 5 — Deep Techniques](../modules/05-deep.md)
> **Code:** [`activities/avaliacao-pratica-1/`](avaliacao-pratica-1/) · **Runner:** [`colab.ipynb`](avaliacao-pratica-1/colab.ipynb)

---

## Overview

Five strategies are compared for classifying CIFAR-10, spanning the full range from
"learn everything from the data at hand" to "inherit almost everything from a model
pretrained on 14 million images":

| # | Strategy | What is learned | What is inherited | Script |
|---|---|---|---|---|
| 1 | CNN from scratch | every weight | nothing | [`s1_cnn_scratch.py`](avaliacao-pratica-1/s1_cnn_scratch.py) |
| 2 | Pretrained CNN as feature extractor + shallow classifier | a shallow classifier | all convolutional filters | [`s2_feature_extraction.py`](avaliacao-pratica-1/s2_feature_extraction.py) |
| 3 | Fine-tuning a pretrained CNN | head + top block | early and mid-level filters | [`s3_finetuning.py`](avaliacao-pratica-1/s3_finetuning.py) |
| 4 | Fine-tuning + data augmentation | head + top block | same, plus an enlarged effective dataset | [`s4_augmentation.py`](avaliacao-pratica-1/s4_augmentation.py) |
| 5 | Fine-tuning a Vision Transformer | head + encoder | ImageNet-21k representations | [`s5_vit.py`](avaliacao-pratica-1/s5_vit.py) |

The interesting question is not which one wins — with 10,000 training images the answer
is nearly foreordained — but **what each increment of transfer actually buys, and at
what cost**. That framing is why every result below is reported alongside its input
resolution, trainable parameter count, and wall-clock training time.

### Dataset: CIFAR-10

| Property | Value |
|---|---|
| Source | `tf.keras.datasets.cifar10` (Krizhevsky, 2009) |
| Images | 60,000 colour images, 32×32×3 |
| Classes | 10, perfectly balanced (6,000 images each) |
| Official split | 50,000 train / 10,000 test |

Classes: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`.
Note that `cat`/`dog` and `automobile`/`truck` are semantically adjacent pairs — the
confusion matrix of any model on this dataset is largely a story about those two pairs.

---

## Experimental Protocol

The protocol is defined once, in [`common.py`](avaliacao-pratica-1/common.py), and every strategy
imports it. Three decisions carry the comparison:

### 1. Equal budget

Every strategy trains on the **same stratified 10,000-image subsample** (1,000 per class),
validates on the **same 2,000 images**, and is scored **once** on the **full official
10,000-image test set**.

The subsample is not a shortcut — it is what makes the comparison legal. The five
strategies differ in cost by a factor of ~50 (a from-scratch CNN at 32×32 trains in
minutes; ViT-B/16 at 224×224 does not), and a free-tier T4 cannot afford the full 50,000
images across five strategies, three seeds and four ablations. Given that constraint, the
choice is between *all strategies on 10,000 images* and *some strategies on more data than
others*. The first is a fair experiment; the second is not an experiment at all.

The consequence must be stated openly rather than buried: **10,000 images is a low-data
regime, and low-data regimes are exactly where transfer learning is strongest.** A
from-scratch CNN would close much of the gap at 50,000 images. The ranking reported here
is a ranking *at this budget*, and the report says so.

### 2. Input resolution: 128×128, not 224×224

CIFAR-10 images are 32×32. Upsampling them to an ImageNet-native 224×224 does not add a
single bit of information — it fabricates pixels by interpolation — while costing roughly
**3× the compute**. Strategies 2, 3 and 4 therefore run at **128×128**, still a 4× upsample
of the source and an officially supported MobileNetV2 input size. The choice is applied
identically across those three strategies, so it is a controlled constant of the
comparison, not a variable. Strategy 5 stays at 224×224: a ViT's patch grid and position
embeddings are fixed by its pretrained checkpoint.

### 3. The test set is touched once

Architecture, epochs, and early stopping are all selected on the validation split. The
test set produces exactly one number per configuration. There is no "best epoch on test",
no threshold tuned on test, and no configuration chosen because it looked good on test.

### 4. No claim rests on a single run

Every trained network is run at three seeds (42, 7, 2024) and reported as mean ± standard
deviation. The seed changes weight initialisation and augmentation sampling — never the
data, which is fixed by a separate RNG. A difference between two configurations that is
smaller than the seed-to-seed spread is not a result.

For the two best models, a **McNemar exact test** on paired test predictions decides
significance. This — not a Wilcoxon over folds — is the right test: both models are
evaluated on identical samples, so their errors are paired, and only the discordant
predictions carry information.

### Evaluation

Accuracy (primary), macro-F1, macro-precision, macro-recall, Wilson 95% confidence
interval, per-class accuracy, and the full confusion matrix. On a 10,000-image test set
the Wilson half-width at ~90% accuracy is about ±0.6 pp — **that is the resolution limit
of every claim in this report**, and differences below it are reported as ties.

---

## Departures from the lecture notebooks

The course examples ([`FeatureExtraction_v_new.ipynb`](../course-materials/notebooks/),
[`Aula10_FineTuning_CNN-1.ipynb`](../course-materials/notebooks/),
[`CNN_FineTuning_DataAugmentation.ipynb`](../course-materials/notebooks/)) are written for a
1,000-image dataset. Four departures were necessary, and each is a result in itself:

| Lecture notebook | This activity | Why |
|---|---|---|
| `np.save` of all images at 224×224 float32 | uint8 32×32 in RAM, resize on the fly in `tf.data` | 60,000 × 224² × 3 float32 ≈ **36 GB**. It does not fit, and it does not need to. |
| `backbone.trainable = False`, never unfrozen | Phase 1 frozen head, then **Phase 2** unfreezes the top block at 1e-5 | `trainable=False` throughout is feature extraction with a deep head — *not* fine-tuning. Strategy 3 is only strategy 3 if Phase 2 exists. |
| `ImageDataGenerator` (CPU) | Keras preprocessing layers inside the `tf.data` graph (GPU) | The CPU generator becomes the bottleneck once images are upsampled to 224×224; it is also deprecated in Keras 3. |
| Single 70/30 holdout, one run | Fixed splits, 3 seeds, McNemar on the top-2 pair | A single holdout gives a point estimate with no dispersion — nothing to test a difference against. |

A reference run reproducing the lecture setup exactly (VGG16 + `Flatten`, frozen backbone,
no Phase 2) is included so the report can quantify what the departures bought.

---

## Open Questions

The assignment attaches four open questions to the strategies. Each is answered as a
**controlled ablation**: one variable moves, everything else is pinned to the same data,
schedule, and seeds.

**Q2(a) — Does swapping the backbone for something simpler, like MobileNet, significantly
change the result?**
MobileNetV2 (3.5M params, 0.6 GFLOPs) vs ResNet50 (25.6M, 8.2) vs InceptionV3 (23.9M, 11.5),
with the classifier and the data held fixed. A ~14× parameter gap and ~19× compute gap.

**Q4(a) — Does replacing `Flatten()` with `GlobalMaxPooling2D()` significantly change the
result?**
Not a cosmetic choice. On MobileNetV2 at 128px the final feature map is 4×4×1280, so
`Flatten → Dense(512)` is **10.5M** head parameters against `GlobalMaxPool → Dense(512)`'s
**0.66M** — a 16× difference in head capacity, fitted on 10,000 images. Global average
pooling is included as a third arm.

**Q4(b) — Can changing the optimiser (`Adam()`) improve the result?**
Adam vs AdamW vs SGD+Nesterov vs RMSprop. SGD is given a 10× larger learning rate, because
comparing optimisers at a learning rate tuned for Adam is a rigged fight, not an experiment.

**Q4(c) — Can other data-augmentation strategies improve the result?**
Four policies, including the lecture notebook's own (`rotation 10°, zoom 0.15, shift 0.1`).
Note what that policy omits: **horizontal flip** — on CIFAR-10 the single most effective and
most obviously label-preserving transform available.

---

## Design Limitations

Three factors confound the comparison. All are stated *before* the results, because none
is fixable within the budget — and a report that omitted them would attribute the effects
it observes to the wrong causes.

**1. The ViT is not facing a new problem.** ViT-B/16 is pretrained on **ImageNet-21k**,
whose label set substantially overlaps CIFAR-10's: airplane, automobile, bird, cat, deer,
dog, frog, horse, ship and truck are all well-represented ImageNet categories. The model
is therefore **not generalising to an unseen domain — it is recognising categories it has
already been trained on**, at far higher resolution and with orders of magnitude more
examples. Its accuracy is an upper bound on transfer under exceptionally favourable
conditions, **not** evidence that attention beats convolution. The pretrained CNNs
(strategies 2–4) share this advantage, but to a lesser degree: ImageNet-1k is ~14× smaller.

**2. Input resolution is not constant across strategies.** The from-scratch CNN runs at
32×32 — the images' native size — while the others run at 128×128 (224×224 for the ViT),
because pretrained networks need inputs compatible with the statistics they were trained
on. The difference is **inherent to what the strategies are**, not an oversight:
upsampling to train a network from scratch would interpolate pixels without adding
information, at a higher compute cost. It nonetheless prevents attributing the performance
gap to pretraining alone.

**3. The ablations ran on a reduced epoch budget.** Open questions 4(b) and 4(c) required
sixteen additional strategy-4 trainings, which did not fit the GPU budget under the
delivery schedule. Epochs were cut from 15 + 12 to 8 + 6, **identically across every arm
of each ablation**. Internal validity is preserved — the question is which arm beats
which, and all arms run on the same budget — but the ablations' **absolute accuracies are
not comparable to the main table**, only to each other.

---

## Results

> Generated by [`report.py`](avaliacao-pratica-1/report.py) from the run artefacts in
> [`avaliacao-pratica-1/results/`](avaliacao-pratica-1/results/). Every JSON carries its own 10,000 test
> predictions, so the tables, the confusion matrix and the significance test can all be
> regenerated on a CPU without retraining anything.

### Strategy comparison

<!-- BEGIN GENERATED: main-table -->
| Strategy | Configuration | Accuracy (test) | 95% CI | Macro-F1 | Resolution | Trainable params | Training |
|---|---|---|---|---|---|---|---|
| 5 — ViT fine-tuning | `vit_base_patch16_224_in21k` | **0.9818 ± 0.0006** | 0.9785–0.9849 | 0.9818 ± 0.0007 | 224px | 85,806,346 | 6.8 min |
| 4 — Fine-tuning + augmentation | `mobilenetv2_gap_flip_crop` | **0.8650** | 0.8582–0.8716 | 0.8650 | 128px | 2,171,722 | 8.6 min |
| 3 — Fine-tuning | `mobilenetv2_gap` | **0.8576 ± 0.0025** | 0.8488–0.8660 | 0.8575 ± 0.0024 | 128px | 2,171,722 | 3.3 min |
| 2 — Feature extraction | `mobilenetv2_svm` | **0.8522** | 0.8451–0.8590 | 0.8524 | 128px | 0 | 0.6 min |
| 1 — CNN from scratch | `cnn_scratch` | **0.7636** | 0.7552–0.7718 | 0.7617 | 32px | 305,258 | 1.7 min |
<!-- END GENERATED: main-table -->

### Statistical significance of the top-2 gap

<!-- BEGIN GENERATED: significance -->
- **1º** `s5_vit / vit_base_patch16_224_in21k` — 0.9825
- **2º** `s4_augment / mobilenetv2_gap_flip_crop` — 0.8650

| Discordant | 1st right / 2nd wrong | 1st wrong / 2nd right | p (exact McNemar) |
|---|---|---|---|
| 1311 | 1243 | 68 | 3.234e-280 |

The 11.75 pp gap is **significant** (α = 0.05). The top-ranked model is therefore the best model in this comparison.
<!-- END GENERATED: significance -->

### Every strategy against every other (paired McNemar, primary seed)

<!-- BEGIN GENERATED: pairwise -->
| Comparison | Δ | p (McNemar) | Significant? |
|---|---|---|---|
| Strategy 2 vs. 1 | +8.86 pp | 5.7e-83 | **yes** |
| Strategy 3 vs. 1 | +9.22 pp | 8.16e-89 | **yes** |
| Strategy 3 vs. 2 | +0.36 pp | 0.159 | no — technical tie |
| Strategy 4 vs. 1 | +10.14 pp | 2.86e-109 | **yes** |
| Strategy 4 vs. 2 | +1.28 pp | 5.96e-06 | **yes** |
| Strategy 4 vs. 3 | +0.92 pp | 0.000397 | **yes** |
| Strategy 5 vs. 1 | +21.89 pp | 0 | **yes** |
| Strategy 5 vs. 2 | +13.03 pp | 4.74e-322 | **yes** |
| Strategy 5 vs. 3 | +12.67 pp | 6.83e-312 | **yes** |
| Strategy 5 vs. 4 | +11.75 pp | 3.23e-280 | **yes** |
<!-- END GENERATED: pairwise -->

*Each row pits one increment of transfer against the previous one. A gap that fails
McNemar is a gap that was **paid for and not received**: the compute was spent, the
accuracy did not arrive. An accuracy ranking cannot show this.*

### Accuracy per GPU-minute

<!-- BEGIN GENERATED: cost -->
| Strategy | Accuracy | Training | pp over from-scratch CNN | pp per minute |
|---|---|---|---|---|
| 5 — ViT fine-tuning | 0.9818 | 6.8 min | +21.82 pp | +3.20 |
| 4 — Fine-tuning + augmentation | 0.8650 | 8.6 min | +10.14 pp | +1.18 |
| 3 — Fine-tuning | 0.8576 | 3.3 min | +9.40 pp | +2.86 |
| 2 — Feature extraction | 0.8522 | 0.6 min | +8.86 pp | +14.45 |
| 1 — CNN from scratch | 0.7636 | 1.7 min | +0.00 pp | +0.00 |
<!-- END GENERATED: cost -->


### Best model — confusion matrix

<p align="center">
  <picture class="github-mode-only">
    <source media="(prefers-color-scheme: dark)" srcset="../assets/img/avaliacao-pratica-1-confusion-dark.png">
    <img src="../assets/img/avaliacao-pratica-1-confusion-light.png" alt="Confusion matrix of the best model">
  </picture>
  <img class="obsidian-light-only" width="0" height="0" src="../assets/img/avaliacao-pratica-1-confusion-light.png" alt="Confusion matrix of the best model">
  <img class="obsidian-dark-only" width="0" height="0" src="../assets/img/avaliacao-pratica-1-confusion-dark.png" alt="Confusion matrix of the best model">
</p>

<!-- BEGIN GENERATED: hardest-classes -->
| Confusion | Rate | Reading |
|---|---|---|
| dog → cat | 3.6% | |
| cat → dog | 2.3% | |
| truck → automobile | 1.5% | |
| automobile → truck | 1.0% | |
| bird → cat | 0.7% | |
<!-- END GENERATED: hardest-classes -->

### Q2(a) — Backbone swap

<!-- BEGIN GENERATED: ablation-backbone -->
| Backbone + classifier | Accuracy (mean ± sd, 1 seeds) | Δ vs. best | Macro-F1 | Training |
|---|---|---|---|---|
| `resnet50_svm` | 0.8761 | +0.00 pp | 0.8767 | 1.0 min |
| `resnet50_mlp` | 0.8651 | -1.10 pp | 0.8653 | 0.4 min |
| `mobilenetv2_svm` | 0.8522 | -2.39 pp | 0.8524 | 0.6 min |
| `mobilenetv2_mlp` | 0.8392 | -3.69 pp | 0.8389 | 0.7 min |
| `inceptionv3_svm` | 0.7867 | -8.94 pp | 0.7867 | 1.6 min |
| `inceptionv3_mlp` | 0.7796 | -9.65 pp | 0.7793 | 0.6 min |
<!-- END GENERATED: ablation-backbone -->

### Q4(a) — `Flatten()` vs `GlobalMaxPooling2D()`

<!-- BEGIN GENERATED: ablation-head -->
| Head (pooling) | Accuracy (mean ± sd, 2 seeds) | Δ vs. best | Macro-F1 | Training |
|---|---|---|---|---|
| `mobilenetv2_gap` | 0.8576 ± 0.0025 | +0.00 pp | 0.8575 | 3.3 min |
| `mobilenetv2_gmp` | 0.8558 ± 0.0020 | -0.18 pp | 0.8555 | 3.5 min |
| `mobilenetv2_flatten` | 0.8470 ± 0.0063 | -1.06 pp | 0.8465 | 4.0 min |
<!-- END GENERATED: ablation-head -->

### Q4(b) — Optimiser

<!-- BEGIN GENERATED: ablation-optimizer -->
| Optimiser | Accuracy (mean ± sd, 2 seeds) | Δ vs. best | Macro-F1 | Training |
|---|---|---|---|---|
| `adamw` | 0.8660 ± 0.0013 | +0.00 pp | 0.8659 | 12.7 min |
| `adam` | 0.8652 ± 0.0034 | -0.09 pp | 0.8648 | 13.4 min |
| `rmsprop` | 0.8638 ± 0.0027 | -0.22 pp | 0.8637 | 12.7 min |
| `sgd` | 0.8349 ± 0.0380 | -3.12 pp | 0.8351 | 11.1 min |
<!-- END GENERATED: ablation-optimizer -->

### Q4(c) — Augmentation policy

<!-- BEGIN GENERATED: ablation-policy -->
| Augmentation policy | Accuracy (mean ± sd, 2 seeds) | Δ vs. best | Macro-F1 | Training |
|---|---|---|---|---|
| `strong` | 0.8678 ± 0.0010 | +0.00 pp | 0.8672 | 20.7 min |
| `lecture` | 0.8676 ± 0.0014 | -0.02 pp | 0.8672 | 21.0 min |
| `flip_crop` | 0.8653 ± 0.0004 | -0.25 pp | 0.8649 | 11.5 min |
| `flip` | 0.8590 ± 0.0028 | -0.88 pp | 0.8585 | 6.3 min |
<!-- END GENERATED: ablation-policy -->

---

## Reproducing

```bash
cd activities/avaliacao-pratica-1
pip install -r requirements.txt

python run_all.py --stage core        # 5 strategies, primary seed   (~35 min, T4)
python run_all.py --stage ablations   # the four open questions      (~70 min)
python run_all.py --stage seeds       # remaining seeds, main table  (~50 min)
python run_all.py --stage baseline    # the lecture notebook's setup (~10 min)
python report.py                      # tables + confusion matrix    (~10 s, CPU)
```

The stages are ordered by value per GPU-minute: the report is already deliverable after
`core`, and each stage after it strengthens rather than completes the submission. Run
`report.py` after any stage — configurations with a single seed report no standard
deviation rather than a fabricated zero.

On Colab, open [`colab.ipynb`](avaliacao-pratica-1/colab.ipynb), select a T4 runtime, and run the
cells: it clones this repository and calls the same scripts. The notebook is the execution
environment; the scripts are the artefact.

---

*[← README](../README.md) · [Module 5 — Deep Techniques](../modules/05-deep.md) · [Answers (pt-BR)](avaliacao-pratica-1-respostas.md)*
