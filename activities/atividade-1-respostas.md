# Atividade 1 — Modelos Rasos: Respostas

> **Disciplina:** Aprendizagem de Máquina · PPGIA / PUC-PR · Mestrado 2026
> **Aluno:** Fernando Dantas
> **Reprodutibilidade:** todos os valores foram gerados por [`atividade-1.py`](atividade-1.py) (semente 42) — ver a seção *Protocolo Experimental* abaixo.

---

## Protocolo Experimental

Os experimentos seguiram o protocolo especificado no enunciado: **validação cruzada com 5 folds**, estratificada na Parte A (preservando a proporção 212/357 entre classes em cada dobra) e com embaralhamento na Parte B. Todos os componentes estocásticos utilizaram semente fixa (42) para garantir reprodutibilidade. Para os indutores sensíveis à escala dos atributos (k-NN, SVM/SVR e MLP), a padronização (*z-score*) foi ajustada exclusivamente nas dobras de treinamento, por meio de *pipeline*, evitando vazamento de dados. As métricas reportadas correspondem à média das cinco dobras. Ambiente: Python 3.13, scikit-learn 1.9.0, XGBoost 3.3.0.

---

## Parte A — Classificação (Breast Cancer Wisconsin)

### Melhor Resultado Observado

| Indutor | Taxa de Acerto (%) | F1_score | Precisão | Recall |
|---|---|---|---|---|
| **MLP (Perceptron Multicamadas)** | **97,72** | 0,9753 | 0,9797 | 0,9723 |

*F1, precisão e recall em média macro. O MLP e a SVM empataram em taxa de acerto (97,72%); o teste de postos sinalizados de Wilcoxon sobre as acurácias por dobra (W = 4,0; p = 0,875) indica que os dois modelos são estatisticamente indistinguíveis neste conjunto de dados. O MLP foi selecionado por apresentar média marginalmente superior em precisão.*

Ordenação completa por taxa de acerto: MLP ≈ SVM (0,9772) > k-NN ≈ XGBoost (0,9631) > Random Forest (0,9561) > Bagging (0,9543) > AdaBoost (0,9525) > Naive Bayes (0,9385) > Árvore de Decisão (0,9104).

### A.1) Qual a taxa de acerto de cada classe?

**R:** Considerando as predições *out-of-fold* agregadas das cinco dobras do melhor modelo (MLP):

- **Classe maligna (0): 95,28%** (202 acertos em 212 instâncias);
- **Classe benigna (1): 99,16%** (354 acertos em 357 instâncias).

Observa-se que a classe maligna — a minoritária e clinicamente crítica — concentra a maior parte dos erros do modelo.

### A.2) Informe a matriz de confusão.

**R:** Matriz de confusão agregada das predições *out-of-fold* do MLP (linhas = classe real; colunas = classe prevista):

| | Previsto: maligno (0) | Previsto: benigno (1) |
|---|---|---|
| **Real: maligno (0)** — 212 | 202 | 10 |
| **Real: benigno (1)** — 357 | 3 | 354 |

Os 10 falsos negativos (tumores malignos classificados como benignos) constituem o erro de maior custo no contexto clínico, pois implicam adiamento do tratamento; os 3 falsos positivos implicam apenas exames confirmatórios adicionais.

### A.3) Informe o valor dos parâmetros utilizados no treinamento.

**R:** O modelo vencedor foi o `MLPClassifier` do scikit-learn, precedido de padronização dos atributos (`StandardScaler` ajustado apenas nas dobras de treinamento), com os seguintes parâmetros:

| Parâmetro | Valor |
|---|---|
| Arquitetura (camadas ocultas) | 1 camada com 100 neurônios — `hidden_layer_sizes=(100,)` |
| Função de ativação | ReLU |
| Otimizador | Adam (β₁ = 0,9; β₂ = 0,999; ε = 10⁻⁸) |
| Taxa de aprendizagem inicial | 0,001 (constante) |
| Regularização L2 (α) | 0,0001 |
| Tamanho do *batch* | `auto` = min(200, n amostras) |
| Número máximo de épocas | 1000 |
| Critério de parada | tolerância 10⁻⁴ por 10 iterações sem melhora |
| Semente aleatória | 42 |

### A.4) Há diferença significativa entre acurácia e F1_score?

**R:** **Não.** A acurácia média (0,9772) supera o F1_score macro médio (0,9753) em apenas 0,19 ponto percentual, e o teste de Wilcoxon pareado sobre os valores por dobra não rejeita a hipótese nula ao nível de 5% (W = 0,0; p = 0,0625). Cabe registrar, contudo, que a diferença, embora pequena, é sistemática — a acurácia excede o F1 em todas as cinco dobras. Isso decorre do desbalanceamento moderado da base (37,3% malignos vs. 62,7% benignos): a acurácia é levemente inflada pelo bom desempenho na classe majoritária (benigna, 99,16% de acerto), ao passo que o F1 macro pondera igualmente as classes e, portanto, penaliza a maior taxa de erro na classe maligna (4,72%). Em síntese, não há diferença estatisticamente significativa, mas o F1 macro é a métrica mais fidedigna para este problema, por refletir melhor o desempenho na classe minoritária e clinicamente relevante.

---

## Parte B — Regressão (Diabetes)

### Tabela de Resultados (Melhor Regressor)

| Indutor | Coeficiente de Determinação (R²) | Erro Médio Absoluto (MAE) |
|---|---|---|
| **MLP (`MLPRegressor`)** | **0,4686** | **44,05** |

O MLP obteve, simultaneamente, o maior R² e o menor MAE entre os regressores avaliados. Ordenação completa por R²: MLP (0,4686) > Random Forest (0,4294) > k-NN (0,3912) > Bagging (0,3761) > XGBoost (0,3290) > SVR (0,1492) > Árvore de Regressão (−0,1325). Nota-se que a árvore de regressão isolada apresentou R² negativo — desempenho inferior ao de simplesmente predizer a média —, evidenciando a alta variância desse indutor sem regularização. O MAE de 44,05 corresponde a aproximadamente 13,7% da amplitude do alvo (25–346), o que indica que a base possui sinal preditivo limitado para todos os indutores avaliados.

### B.1) Informe o valor dos parâmetros utilizados no treinamento do modelo.

**R:** O modelo vencedor foi o `MLPRegressor` do scikit-learn, precedido de padronização dos atributos (`StandardScaler` ajustado apenas nas dobras de treinamento), com os seguintes parâmetros:

| Parâmetro | Valor |
|---|---|
| Arquitetura (camadas ocultas) | 1 camada com 100 neurônios — `hidden_layer_sizes=(100,)` |
| Função de ativação | ReLU |
| Função de perda | Erro quadrático (`squared_error`) |
| Otimizador | Adam (β₁ = 0,9; β₂ = 0,999; ε = 10⁻⁸) |
| Taxa de aprendizagem inicial | 0,001 (constante) |
| Regularização L2 (α) | 0,0001 |
| Tamanho do *batch* | `auto` = min(200, n amostras) |
| Número máximo de épocas | 2000 |
| Critério de parada | tolerância 10⁻⁴ por 10 iterações sem melhora |
| Semente aleatória | 42 |

---

## Reprodutibilidade

Todos os resultados podem ser regenerados com:

```bash
python activities/atividade-1.py
```

A seção final da saída (`Respostas — enunciado original`) imprime exatamente os valores reportados neste documento: ranking dos indutores, taxa de acerto por classe, matriz de confusão, parâmetros dos modelos vencedores e o teste de significância entre acurácia e F1.

---

*[← Atividade 1 (versão estendida, em inglês)](atividade-1.md) · [README](../README.md)*
