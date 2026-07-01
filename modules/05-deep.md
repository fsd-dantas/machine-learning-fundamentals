# Module 5 — Deep Techniques

> **Colour:** ![#58a6ff](https://placehold.co/12x12/58a6ff/58a6ff.png) Blue  
> **Prerequisites:** [Module 1 — Foundations](01-foundations.md) · [Module 3 — Shallow Techniques](03-shallow.md) (MLP baseline)  
> **Connects to:** [Module 2 — Experimental Protocols](02-protocols.md) (evaluation applies unchanged)

---

## Overview

Deep techniques extend the parametric learning paradigm of the MLP to architectures capable of learning hierarchical feature representations directly from raw data. The central insight is that useful features — edges, textures, syntax, temporal patterns — need not be hand-crafted; they can be learned jointly with the classifier or regressor through backpropagation across many layers.

This module covers the biological motivation and the Perceptron as the atomic unit of neural computation, the theoretical foundations common to all deep architectures, and five canonical families: Multilayer Perceptrons (fully connected), Convolutional Neural Networks (spatial/local structure), Recurrent Neural Networks (sequential structure), Siamese Networks (similarity learning), and the Transformer with its role as a generative model.

---

## How to Use This Module (Exam Prep)

**After this module you should be able to:**
- Explain the **perceptron**, its update rule, and **why XOR forced the move to multiple layers**.
- Describe **backpropagation** in words (forward pass → loss → chain-rule gradients backward).
- Apply one **gradient-descent** weight update from numbers.
- State what **CNNs** (spatial locality, parameter sharing) and **RNN/LSTM** (sequence memory, gates) are *for*.
- Explain **self-attention** at a high level and why transformers beat RNNs on long sequences.
- Match each architecture to a data type (images→CNN, sequences→RNN/Transformer, tabular→MLP).

**⭐ High-yield for exams:** perceptron + XOR limitation · what backprop does · vanishing-gradient problem (and how LSTM gates fix it) · CNN parameter sharing · attention vs. recurrence · activation functions (ReLU, sigmoid, softmax).

**If you only read one thing:** §1 Perceptron and §3 Backpropagation — they explain *how every neural network learns*.

**Suggested time:** ~2 h reading + ~45 min on the [Worked Examples](#worked-examples) and [Self-Check](#self-check). Architectures (§5–8) are often awareness-level; confirm depth with your instructor.

---

## 1. Biological Motivation and the Perceptron

### From Biological Neuron to Computational Unit

A biological neuron integrates incoming signals at its dendrites, and fires an output spike along its axon once the aggregated input crosses a threshold. The **McCulloch–Pitts neuron** (1943) abstracts this into a binary thresholded sum of inputs — establishing that networks of simple thresholding units can, in principle, compute logical functions. This is the conceptual seed of all neural computation: a unit that performs a weighted aggregation followed by a non-linearity.

### The Perceptron

Rosenblatt's **Perceptron** (1958) makes the McCulloch–Pitts unit *learnable* by attaching adjustable weights and a learning rule. Given input $\mathbf{x} \in \mathbb{R}^d$, weights $\mathbf{w}$, and bias $b$, it computes:

$$\hat{y} = \text{sign}(\mathbf{w}^\top \mathbf{x} + b)$$

The decision surface $\mathbf{w}^\top \mathbf{x} + b = 0$ is exactly the hyperplane from [Module 1 — Analytical Geometry](01-foundations.md#2-analytical-geometry). The perceptron is therefore a **linear classifier**; what is new relative to Module 3 is not the model but the *online, error-driven* way it is fit.

### Perceptron Learning Rule

For each misclassified example $(\mathbf{x}_i, y_i)$ with labels $y_i \in \{-1, +1\}$, the weights are nudged toward correcting that example:

$$\mathbf{w} \leftarrow \mathbf{w} + \eta \, y_i \, \mathbf{x}_i, \qquad b \leftarrow b + \eta \, y_i$$

where $\eta > 0$ is the learning rate. The **Perceptron Convergence Theorem** (Novikoff, 1962) guarantees that this procedure terminates in a finite number of updates **if and only if the training data is linearly separable**.

> **📝 Worked Example — one perceptron update**
> Weights $\mathbf{w} = [0, 0]$, $b = 0$, learning rate $\eta = 1$. Training point $\mathbf{x} = [2, 1]$ with true label $y = +1$.
> 1. **Predict:** $\mathbf{w}^\top\mathbf{x} + b = 0 \Rightarrow \text{sign}(0)$ — treat as wrong (not $+1$).
> 2. **Update** (misclassified): $\mathbf{w} \leftarrow [0,0] + (1)(+1)[2,1] = [2, 1]$; $b \leftarrow 0 + 1 = 1$.
> 3. **Re-check:** $\mathbf{w}^\top\mathbf{x} + b = (2)(2)+(1)(1)+1 = 6 > 0 \Rightarrow +1$ ✓ now correct.
>
> The rule simply **nudges the boundary toward getting that point right**. Repeat over all points until none are misclassified (guaranteed only if the data is linearly separable).

### The XOR Limitation — Why Depth Is Necessary

A single perceptron can represent AND, OR, and NOT, but **not XOR**: no single hyperplane separates $\{(0,0),(1,1)\}$ from $\{(0,1),(1,0)\}$. Minsky and Papert's *Perceptrons* (1969) formalised this limitation and is commonly cited as a trigger for the first "AI winter."

```
   XOR is not linearly separable — no single straight line splits ● from ○:

     x2
      1 │  ●(0,1)        ○(1,1)
        │
        │        (no line works)
      0 │  ○(0,0)        ●(1,1→1,0)
        └────────────────────────  x1
           0              1
     ● = class 1 (output 1):  (0,1),(1,0)
     ○ = class 0 (output 0):  (0,0),(1,1)
   Fix: stack perceptrons — a hidden layer carves TWO lines and combines them.
```

The resolution is **composition**: stacking perceptrons into hidden layers lets the network carve the input space with multiple hyperplanes and recombine them, representing non-linearly-separable functions such as XOR. This is the direct motivation for the Multilayer Perceptron (§4) and for the depth argument that follows. Two ingredients were required to make the stack trainable: a **differentiable** activation replacing the non-differentiable $\text{sign}(\cdot)$, and **backpropagation** (§3) to assign credit across layers.

**Common pitfall:** Treating the perceptron as obsolete. Its update rule is the conceptual ancestor of stochastic gradient descent, and the linear-separability lens remains the right first question to ask of any classification problem before reaching for capacity.

---

## 2. Universal Approximation and the Case for Depth

A single hidden-layer MLP with a non-polynomial activation function can approximate any continuous function on a compact set to arbitrary precision — the **Universal Approximation Theorem** (Cybenko, 1989; Hornik, 1991). This establishes theoretical expressiveness but does not address efficiency.

**Depth vs. width:** Representing certain functions requires exponentially wide shallow networks but only polynomially deep networks. Empirically, depth confers better generalisation on structured data (images, text, time-series) by composing progressively abstract representations.

---

## 3. Backpropagation and Gradient-Based Optimisation

### Forward Pass

For a network with $L$ layers, the forward pass computes activations sequentially:

$$\mathbf{a}^{(l)} = \sigma\left(\mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}\right), \quad l = 1, \ldots, L$$

### Backpropagation

> **💡 Intuition:** Training a net = "turn each knob (weight) the direction that lowers the error." Backprop is just the **chain rule** computing, efficiently, *how much each weight contributed to the error* — by propagating the error signal **backwards** from the output layer to the input layer. Forward pass makes a prediction; backward pass assigns blame; the optimiser nudges every weight a little against its gradient. Repeat.

Computes the gradient of the loss $\mathcal{L}$ with respect to all parameters via the chain rule. The gradient at layer $l$:

$$\boldsymbol{\delta}^{(l)} = \left(\mathbf{W}^{(l+1)\top} \boldsymbol{\delta}^{(l+1)}\right) \odot \sigma'\left(\mathbf{z}^{(l)}\right)$$

where $\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$ and $\odot$ denotes element-wise multiplication.

### Optimisers

| Optimiser | Update rule (sketch) | Notes |
|---|---|---|
| SGD | $\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla \mathcal{L}$ | Baseline; needs tuned $\eta$ and momentum |
| Momentum | Accumulates gradient history | Dampens oscillations |
| RMSprop | Adapts learning rate per parameter | Effective for non-stationary objectives |
| Adam | Combines momentum + RMSprop | Default for most tasks; $\beta_1=0.9$, $\beta_2=0.999$ |

> **📝 Worked Example — one gradient-descent step**
> Minimise $\mathcal{L}(w) = w^2$ (gradient $\nabla\mathcal{L} = 2w$). Start $w = 4$, learning rate $\eta = 0.1$.
> - Step 1: $w \leftarrow 4 - 0.1(2\cdot4) = 4 - 0.8 = \mathbf{3.2}$
> - Step 2: $w \leftarrow 3.2 - 0.1(2\cdot3.2) = 3.2 - 0.64 = \mathbf{2.56}$
> - … each step moves $w$ **downhill** toward the minimum at $w=0$.
>
> **Learning-rate lesson:** too small ($\eta=0.001$) → crawls; too large ($\eta=1$) → $w \leftarrow 4 - 1(8) = -4$, then $+4$… it **oscillates and diverges**. Picking $\eta$ is the single most important hyperparameter.

### Activation Functions

| Function | Formula | Range | Notes |
|---|---|---|---|
| Sigmoid | $\frac{1}{1+e^{-z}}$ | $(0,1)$ | Saturates; vanishing gradient in deep nets |
| Tanh | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $(-1,1)$ | Zero-centred; still saturates |
| ReLU | $\max(0, z)$ | $[0, \infty)$ | Default hidden activation; dead neuron problem |
| Leaky ReLU | $\max(\alpha z, z)$, $\alpha \ll 1$ | $(-\infty, \infty)$ | Mitigates dead neurons |
| GELU | $z \cdot \Phi(z)$ | $(-\infty, \infty)$ | Used in transformers |

### Regularisation

- **L2 (weight decay):** Adds $\frac{\lambda}{2}\|\mathbf{w}\|^2$ to the loss — penalises large weights.
- **Dropout:** Randomly zeroes activations with probability $p$ during training; approximates an ensemble of $2^n$ subnetworks.
- **Batch Normalisation:** Normalises layer inputs to zero mean and unit variance per mini-batch; accelerates training and acts as implicit regularisation.
- **Early stopping:** Monitors validation loss; halts training when it stops decreasing.

---

## 4. Multilayer Perceptron (MLP)

A fully connected feedforward network. Each neuron in layer $l$ is connected to every neuron in layer $l-1$.

**Design decisions:** Number of layers, units per layer, activation function, output activation (softmax for multiclass, sigmoid for binary, linear for regression), loss function.

**Common pitfall:** More capacity is not always better — without regularisation, deeper/wider networks overfit on small datasets. The bias–variance trade-off from Module 1 applies directly.

---

## 5. Convolutional Neural Networks (CNN)

CNNs exploit the spatial locality and translational invariance of image-like data through **convolutional layers**:

$$(\mathbf{X} * \mathbf{K})_{i,j} = \sum_m \sum_n \mathbf{X}_{i+m, j+n} \cdot \mathbf{K}_{m,n}$$

**Key components:**

- **Convolutional layer:** Learns local feature detectors (kernels/filters). Parameters: kernel size, number of filters, stride, padding.
- **Activation (ReLU):** Applied element-wise after convolution.
- **Pooling layer:** Downsamples spatial dimensions — max pooling retains the most salient activation in each region.
- **Fully connected layer:** Aggregates learned features for the final prediction.

**Parameter sharing** (same kernel applied across all spatial positions) drastically reduces parameter count compared to a fully connected layer on raw pixels.

**Landmark architectures:** LeNet-5, AlexNet, VGG, ResNet (residual connections to mitigate vanishing gradients), EfficientNet.

---

## 6. Recurrent Neural Networks (RNN)

RNNs process sequential data by maintaining a hidden state $\mathbf{h}_t$ that summarises history:

$$\mathbf{h}_t = \sigma(\mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{W}_x \mathbf{x}_t + \mathbf{b})$$

**Vanishing / exploding gradient problem:** Gradients propagated over long sequences either vanish (exponentially small) or explode (exponentially large), making plain RNNs impractical for long-range dependencies.

### Long Short-Term Memory (LSTM)

Introduces a cell state $\mathbf{c}_t$ and three gating mechanisms:

$$\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \quad \text{(forget gate)}$$
$$\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \quad \text{(input gate)}$$
$$\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) \quad \text{(output gate)}$$
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$$

The cell state acts as a long-term memory that can be selectively read, written, and erased — resolving the vanishing gradient problem for sequences of practical length.

> **💡 Intuition:** A plain RNN forgets the start of a long sentence by the time it reaches the end (gradients vanish). An LSTM adds a **conveyor-belt memory** (the cell state) plus three gates that decide what to **forget**, what to **add**, and what to **output**. Because information can ride the belt unchanged, gradients survive over long sequences. GRU is the slimmed-down version (two gates, no separate cell state).

### Gated Recurrent Unit (GRU)

A simplified variant of LSTM with two gates (reset and update) and no separate cell state. Competitive performance with fewer parameters.

### Research Application (SDN / Smart Grid)

LSTMs and GRUs are directly applicable to the time-series prediction tasks in the author's MSc research:
- **VNF auto-scaling:** Predict future traffic load from historical SDN flow statistics → LSTM regressor.
- **Smart Grid forecasting:** Predict power demand from historical consumption and weather data → seq2seq LSTM.
- **Anomaly detection:** Model normal traffic sequences; flag deviations — unsupervised LSTM autoencoder.

---

## 7. Siamese Networks

A Siamese network is an architecture designed for **similarity learning**: rather than classifying an input into a fixed set of classes, it learns a function that measures how similar two inputs are. The architecture consists of two (or more) identical subnetworks — sharing the same weights — that each encode one input into an embedding space:

$$f_\theta(\mathbf{x}_1), \quad f_\theta(\mathbf{x}_2) \quad \text{(shared encoder } f_\theta \text{)}$$

The similarity (or distance) between the two embeddings drives the final decision. Common distance functions: Euclidean distance $\|f_\theta(\mathbf{x}_1) - f_\theta(\mathbf{x}_2)\|_2$, cosine similarity.

### Loss Functions

**Contrastive loss** (Hadsell et al., 2006) — pulls similar pairs together, pushes dissimilar pairs apart:

$$\mathcal{L} = (1 - y) \cdot \frac{1}{2} D^2 + y \cdot \frac{1}{2} \max(0,\, m - D)^2$$

where $D = \|f_\theta(\mathbf{x}_1) - f_\theta(\mathbf{x}_2)\|_2$, $y = 1$ for dissimilar pairs, and $m$ is a margin.

**Triplet loss** (Schroff et al., 2015) — anchor $\mathbf{x}_a$, positive $\mathbf{x}_p$ (same class), negative $\mathbf{x}_n$ (different class):

$$\mathcal{L} = \max\left(0,\; \|f_\theta(\mathbf{x}_a) - f_\theta(\mathbf{x}_p)\|_2^2 - \|f_\theta(\mathbf{x}_a) - f_\theta(\mathbf{x}_n)\|_2^2 + \alpha\right)$$

where $\alpha > 0$ enforces that positives are strictly closer than negatives by a margin.

### Applications

Siamese networks excel in **few-shot learning** and **metric learning** scenarios where class distributions shift or new classes appear at test time. The model generalises by similarity, not by memorised class boundaries:

- **Face verification** (same person / different person) — LFW benchmark.
- **Signature verification** and handwritten character recognition — central to the research group of Prof. Alceu (cf. Tannugi, Britto & Koerich, SMC 2019 on cross-dataset facial expression recognition).
- **Cross-dataset transfer** — the shared encoder can be pre-trained on one domain and fine-tuned for another.

**Key design decision:** The backbone encoder $f_\theta$ can be any architecture — CNN for images, LSTM for sequences, MLP for tabular data. The Siamese structure is a training paradigm, not a fixed architecture.

---

## 8. Transformer and Generative Models

The transformer (Vaswani et al., 2017) represents a fundamental shift from recurrent and convolutional inductive biases to a fully attention-based mechanism. It is the dominant architecture for sequence modelling, NLP, and increasingly for vision and multimodal tasks.

### Self-Attention Mechanism

Given an input sequence $\mathbf{X} \in \mathbb{R}^{n \times d}$, self-attention projects it into query, key, and value matrices via learned projections $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V$:

$$\mathbf{Q} = \mathbf{X}\mathbf{W}^Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}^K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}^V$$

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

> **💡 Intuition:** Self-attention lets every word **look at every other word** and decide how much each one matters for understanding it ("it" → which noun?). Think database lookup: each token issues a **query**, every token offers a **key**; the match scores (softmax) say how much of each token's **value** to mix in. Unlike an RNN, this happens for all positions **at once** (parallel) and connects distant words in one step — at the cost of $O(n^2)$ comparisons.

The $\frac{1}{\sqrt{d_k}}$ scaling prevents dot products from saturating the softmax as $d_k$ grows. **Multi-head attention** runs $h$ independent attention heads in parallel, concatenates their outputs, and projects the result — allowing simultaneous attention to different representation subspaces.

**Key advantage over RNNs:** All positions are processed in parallel; long-range dependencies are modelled in $O(1)$ sequential operations (vs. $O(n)$ for RNNs). Trade-off: $O(n^2)$ memory and compute in sequence length $n$.

### Architecture

The original encoder–decoder transformer:

- **Encoder:** $N$ identical layers, each with multi-head self-attention + position-wise feed-forward + residual connection + layer normalisation.
- **Decoder:** Same structure plus a cross-attention sub-layer attending to encoder output.
- **Positional encoding:** Self-attention is permutation-invariant, so positional information is injected via sinusoidal encodings added to input embeddings.

### Transformer as a Generative Model

Generative transformers model the conditional distribution $p(x_t \mid x_1, \ldots, x_{t-1})$ autoregressively — the probability of the next token given all previous tokens. Generation proceeds by iterative sampling.

**Decoder-only (GPT family):** Trained on language modelling; generates sequences token by token.

**Encoder–decoder (T5, original seq2seq):** Encoder builds a context representation; decoder generates conditioned on it.

Broader generative model landscape:

| Model | Generative mechanism |
|---|---|
| Autoregressive transformer | $p(x_t \mid x_{<t})$ — sequential sampling |
| Variational Autoencoder (VAE) | Encoder maps to latent distribution $q(\mathbf{z}\mid\mathbf{x})$; decoder samples |
| GAN | Generator vs. discriminator adversarial training |
| Diffusion model | Iterative denoising of Gaussian noise toward data distribution |

**Graph Neural Networks (GNNs)** extend deep learning to graph-structured data — directly relevant to SDN topology modelling. They are surveyed in Extended Reading below.

---

## Worked Examples

Collected for revision: **one perceptron update** (§1), the **XOR non-separability picture** (§1), and **one gradient-descent step** (§3). The perceptron update and the GD step are the two hand-computations most likely to appear; the XOR argument is the most likely *conceptual* question.

---

## Self-Check

<details>
<summary><strong>Q1.</strong> Why can't a single perceptron solve XOR, and what fixes it? (§1)</summary>

XOR isn't **linearly separable** — no single straight line splits its two classes. Adding a **hidden layer** (stacking perceptrons) lets the network combine multiple lines to carve a non-linear boundary.
</details>

<details>
<summary><strong>Q2.</strong> In one sentence each: what do the forward pass and the backward pass do? (§3)</summary>

**Forward:** feed input through the layers to produce a prediction and compute the loss. **Backward:** use the chain rule to compute each weight's gradient (its share of the blame), so the optimiser can update it.
</details>

<details>
<summary><strong>Q3.</strong> $\mathcal{L}(w)=w^2$, $w=2$, $\eta=0.1$. What is $w$ after one GD step? (§3)</summary>

$\nabla = 2w = 4$; $w \leftarrow 2 - 0.1(4) = \mathbf{1.6}$. (Moving downhill toward 0.)
</details>

<details>
<summary><strong>Q4.</strong> What problem do LSTM gates solve that plain RNNs suffer from? (§6)</summary>

The **vanishing/exploding gradient** problem over long sequences. The cell state + gates let information (and gradients) flow across many time steps without decaying.
</details>

<details>
<summary><strong>Q5.</strong> Why do CNNs use far fewer parameters than a fully connected net on images? (§5)</summary>

**Parameter sharing** — the same small kernel slides across the whole image, so one set of weights detects a feature everywhere, instead of separate weights per pixel.
</details>

<details>
<summary><strong>Q6.</strong> One advantage and one cost of self-attention vs. an RNN. (§8)</summary>

**Advantage:** processes all positions in parallel and links distant tokens in one step (great for long-range dependencies). **Cost:** $O(n^2)$ memory/compute in sequence length.
</details>

<details>
<summary><strong>Q7.</strong> Which activation for a hidden layer, for a binary output, for a multiclass output? (§3)</summary>

Hidden: **ReLU**. Binary output: **sigmoid**. Multiclass output: **softmax**.
</details>

---

## 🔑 Quick Revision

| Concept | One-line takeaway |
|---|---|
| Perceptron | linear classifier + error-driven update; can't do XOR |
| Depth | stacking layers represents non-linear functions (XOR fix) |
| Backprop | chain rule assigns error-blame to every weight, output→input |
| Gradient descent | $w \leftarrow w - \eta\nabla\mathcal{L}$; $\eta$ too big diverges, too small crawls |
| Activations | ReLU (hidden), sigmoid (binary), softmax (multiclass) |
| Regularisation | dropout, L2/weight decay, batch-norm, early stopping |
| CNN | local filters + parameter sharing → images |
| RNN → LSTM/GRU | sequence memory; gates fix vanishing gradients |
| Siamese | shared encoder; similarity/few-shot learning |
| Transformer | self-attention; parallel; long-range; $O(n^2)$; powers GPT |

**Two mantras:** *Forward predicts, backward assigns blame, optimiser updates.* *XOR broke the single perceptron — depth fixed it.*

---

## Professor's References

> Full entries in [`../references.md`](../references.md).

- GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. MIT Press, 2016. — Part II: Modern Practical Deep Networks (Chapters 6–12). Available at: <https://www.deeplearningbook.org>.
- BISHOP, C. M. *Pattern Recognition and Machine Learning*. Springer, 2006. — Chapter 5 (neural networks).
- LECUN, Y.; BENGIO, Y.; HINTON, G. Deep learning. *Nature*, v. 521, p. 436–444, 2015.

---

## Extended Reading

- HOCHREITER, S.; SCHMIDHUBER, J. Long short-term memory. *Neural Computation*, v. 9, n. 8, p. 1735–1780, 1997. — Original LSTM paper.
- VASWANI, A. et al. Attention is all you need. In: *NeurIPS*, 2017. Available at: <https://arxiv.org/abs/1706.03762>. — Transformer architecture.
- HE, K. et al. Deep residual learning for image recognition. In: *CVPR*, 2016. Available at: <https://arxiv.org/abs/1512.03385>. — ResNet; residual connections.
- SCARSELLI, F. et al. The graph neural network model. *IEEE Transactions on Neural Networks*, v. 20, n. 1, p. 61–80, 2009. — GNN foundations; relevant to SDN topology.
- ZHANG, C. et al. A deep neural network for traffic flow prediction. *IEEE Transactions on Intelligent Transportation Systems*, 2019. — Time-series deep learning applied to network traffic; methodological parallel to VNF load prediction.

---

## Connected Activities

No course activity currently targets Module 5 exclusively. The MLP appears in [Atividade 1](../activities/atividade-1.md) as a shallow-regime baseline. Deep architectures will be exercised in future activities.

---

*[← Module 4 — Descriptive Models](04-descriptive.md) · [README](../README.md)*
