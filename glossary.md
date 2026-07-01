# Glossary

> Every acronym and term used across the modules, one line each. When a symbol or abbreviation in a module is unfamiliar, look it up here. Grouped by theme; module links point to where the concept is developed.

---

## Acronyms (quick lookup)

| Acronym | Expansion | One-line meaning |
|---|---|---|
| **AUC** | Area Under the (ROC) Curve | Threshold-independent measure of classifier discrimination ([M2](modules/02-protocols.md#3-evaluation-metrics)) |
| **CART** | Classification And Regression Trees | Tree algorithm using Gini impurity, binary splits ([M3](modules/03-shallow.md#5-tree-based-methods)) |
| **CNN** | Convolutional Neural Network | Deep net for images; local filters + parameter sharing ([M5](modules/05-deep.md#5-convolutional-neural-networks-cnn)) |
| **CV** | Cross-Validation | Resampling to estimate generalisation error ([M2](modules/02-protocols.md#2-data-partitioning-strategies)) |
| **DBSCAN** | Density-Based Spatial Clustering of Applications with Noise | Density clustering; arbitrary shapes + outliers ([M4](modules/04-descriptive.md#dbscan)) |
| **F1** | F1 Score | Harmonic mean of precision and recall ([M2](modules/02-protocols.md#3-evaluation-metrics)) |
| **FN / FP** | False Negative / False Positive | A miss / a false alarm ([M2](modules/02-protocols.md#3-evaluation-metrics)) |
| **GELU** | Gaussian Error Linear Unit | Smooth activation used in transformers ([M5](modules/05-deep.md#3-backpropagation-and-gradient-based-optimisation)) |
| **GNN** | Graph Neural Network | Deep learning on graph-structured data ([M5](modules/05-deep.md#8-transformer-and-generative-models)) |
| **GRU** | Gated Recurrent Unit | Simplified LSTM, two gates ([M5](modules/05-deep.md#6-recurrent-neural-networks-rnn)) |
| **ID3** | Iterative Dichotomiser 3 | Original decision tree, uses information gain ([M3](modules/03-shallow.md#5-tree-based-methods)) |
| **k-NN** | k-Nearest Neighbours | Predict from the $k$ closest training points ([M3](modules/03-shallow.md#2-instance-based-methods)) |
| **LDA** | Linear Discriminant Analysis | Supervised dimensionality reduction; max class separation ([M4](modules/04-descriptive.md#linear-discriminant-analysis-lda)) |
| **LSTM** | Long Short-Term Memory | Gated RNN that handles long sequences ([M5](modules/05-deep.md#6-recurrent-neural-networks-rnn)) |
| **MAE** | Mean Absolute Error | Average absolute regression error; robust to outliers ([M2](modules/02-protocols.md#3-evaluation-metrics)) |
| **MLE** | Maximum Likelihood Estimation | Pick parameters that make the data most probable ([M1](modules/01-foundations.md#3-probability--statistics)) |
| **MLP** | Multilayer Perceptron | Fully connected feedforward neural network ([M5](modules/05-deep.md#4-multilayer-perceptron-mlp)) |
| **MSE / RMSE** | (Root) Mean Squared Error | Squared regression error; penalises large errors ([M2](modules/02-protocols.md#3-evaluation-metrics)) |
| **PCA** | Principal Component Analysis | Unsupervised max-variance dimensionality reduction ([M4](modules/04-descriptive.md#principal-component-analysis-pca)) |
| **QDA** | Quadratic Discriminant Analysis | Like LDA but per-class covariance ([M1](modules/01-foundations.md#3-probability--statistics)) |
| **RBF** | Radial Basis Function | Gaussian kernel for SVM ([M3](modules/03-shallow.md#4-geometric-methods)) |
| **ReLU** | Rectified Linear Unit | $\max(0,z)$; default hidden activation ([M5](modules/05-deep.md#3-backpropagation-and-gradient-based-optimisation)) |
| **RNN** | Recurrent Neural Network | Net with a hidden state for sequences ([M5](modules/05-deep.md#6-recurrent-neural-networks-rnn)) |
| **ROC** | Receiver Operating Characteristic | TPR-vs-FPR curve across thresholds ([M2](modules/02-protocols.md#3-evaluation-metrics)) |
| **R²** | Coefficient of Determination | Proportion of variance explained (can be < 0) ([M2](modules/02-protocols.md#3-evaluation-metrics)) |
| **SGD** | Stochastic Gradient Descent | Gradient update on mini-batches ([M5](modules/05-deep.md#3-backpropagation-and-gradient-based-optimisation)) |
| **SVD** | Singular Value Decomposition | rotate→stretch→rotate matrix factorisation; powers PCA ([M1](modules/01-foundations.md#1-linear-algebra)) |
| **SVM / SVR** | Support Vector Machine / Regression | Maximum-margin classifier / $\varepsilon$-tube regressor ([M3](modules/03-shallow.md#4-geometric-methods)) |
| **t-SNE** | t-distributed Stochastic Neighbour Embedding | Non-linear 2-D visualisation; not reusable ([M4](modules/04-descriptive.md#t-sne)) |
| **TP / TN** | True Positive / True Negative | Correct positive / negative prediction ([M2](modules/02-protocols.md#3-evaluation-metrics)) |
| **UMAP** | Uniform Manifold Approximation and Projection | Faster, reusable non-linear reduction ([M4](modules/04-descriptive.md#umap)) |
| **XGBoost** | Extreme Gradient Boosting | Regularised, second-order gradient boosting ([M3](modules/03-shallow.md#xgboost-extreme-gradient-boosting)) |

---

## Core terms

- **Bias (statistical):** Error from a model being too simple to capture the true pattern → underfitting. ([M1](modules/01-foundations.md#3-probability--statistics))
- **Variance (model):** Error from a model being too sensitive to the specific training sample → overfitting. ([M1](modules/01-foundations.md#3-probability--statistics))
- **Inductive bias:** The set of assumptions a method makes to generalise beyond the training data (e.g. k-NN assumes nearby points share labels). ([M3](modules/03-shallow.md))
- **Generalisation error:** Expected loss on unseen data from the same distribution — what we actually care about. ([M2](modules/02-protocols.md#1-the-generalisation-problem))
- **Overfitting / Underfitting:** Memorising noise (low train, high test error) / too simple to fit (high train error). ([M1](modules/01-foundations.md#3-probability--statistics))
- **Hyperparameter:** A setting chosen before training (e.g. $k$, $C$, tree depth), tuned on validation data — not learned. ([M3](modules/03-shallow.md))
- **Regularisation:** Penalising complexity (L1/L2, dropout) to reduce variance. ([M3](modules/03-shallow.md#1-linear-models), [M5](modules/05-deep.md#3-backpropagation-and-gradient-based-optimisation))
- **Supervised / Unsupervised:** Learning with labels (classification/regression) / without labels (clustering, dimensionality reduction). ([M3](modules/03-shallow.md), [M4](modules/04-descriptive.md))
- **Bagging:** Train many models in parallel on bootstrap samples and average → reduces variance. ([M3](modules/03-shallow.md#6-ensemble-methods))
- **Boosting:** Train models sequentially, each fixing the last one's errors → reduces bias. ([M3](modules/03-shallow.md#6-ensemble-methods))
- **Bootstrap sample:** A sample of the data drawn with replacement (same size, some rows repeated). ([M3](modules/03-shallow.md#6-ensemble-methods))
- **Entropy:** Measure of class impurity in a node; 0 = pure, $\log_2 K$ = maximally mixed. ([M3](modules/03-shallow.md#5-tree-based-methods))
- **Information gain:** Reduction in entropy from a split; ID3 picks the highest-gain attribute. ([M3](modules/03-shallow.md#5-tree-based-methods))
- **Margin:** Distance from an SVM's boundary to the nearest points (support vectors); SVM maximises it. ([M3](modules/03-shallow.md#4-geometric-methods))
- **Kernel trick:** Compute dot products in a high-dimensional space implicitly, enabling non-linear SVM boundaries. ([M3](modules/03-shallow.md#4-geometric-methods))
- **Centroid:** The mean point of a cluster; k-means iteratively moves centroids. ([M4](modules/04-descriptive.md#1-clustering))
- **Stratification:** Splitting data so each fold keeps the full dataset's class proportions. ([M2](modules/02-protocols.md#2-data-partitioning-strategies))
- **Confusion matrix:** Table of TP/FP/FN/TN; the source of all classification metrics. ([M2](modules/02-protocols.md#3-evaluation-metrics))
- **Backpropagation:** Chain-rule algorithm computing each weight's gradient, propagating error output→input. ([M5](modules/05-deep.md#3-backpropagation-and-gradient-based-optimisation))
- **Vanishing/exploding gradient:** Gradients shrinking/growing exponentially over long sequences; LSTM gates mitigate it. ([M5](modules/05-deep.md#6-recurrent-neural-networks-rnn))
- **Self-attention:** Mechanism letting every sequence position weigh every other; core of the transformer. ([M5](modules/05-deep.md#8-transformer-and-generative-models))
- **Softmax:** Turns a vector of scores into a probability distribution; multiclass output activation. ([M5](modules/05-deep.md#8-transformer-and-generative-models))

---

*[← README](README.md) · [Study Guide](study-guide.md) · [Flashcards](flashcards.md)*
