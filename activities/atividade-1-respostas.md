# Atividade 1 — Modelos Rasos: Respostas

> **Disciplina:** Aprendizagem de Máquina · PPGIA / PUC-PR · Mestrado 2026
> **Aluno:** Fernando Dantas
> **Reprodutibilidade:** todos os valores foram gerados por [`atividade-1.py`](atividade-1.py) (semente 42) — ver a seção *Protocolo Experimental* abaixo.

---

## Protocolo Experimental

Os experimentos seguiram o protocolo especificado no enunciado: **validação cruzada com 5 folds**, estratificada na Parte A (preservando a proporção 212/357 entre classes em cada dobra) e com embaralhamento na Parte B. Todos os componentes estocásticos utilizaram semente fixa (42) para garantir reprodutibilidade. Para os indutores sensíveis à escala dos atributos (k-NN, SVM/SVR e MLP), a padronização (*z-score*) foi ajustada exclusivamente nas dobras de treinamento, por meio de *pipeline*, evitando vazamento de dados. As métricas são reportadas como média das cinco dobras, acompanhadas do desvio-padrão entre dobras. Ambiente: Python 3.13, scikit-learn 1.9.0, XGBoost 3.3.0.

**Escopo da comparação.** Todos os indutores foram avaliados com as configurações padrão das respectivas bibliotecas (apenas `max_iter` foi elevado, para garantir convergência do MLP); **não houve busca sistemática de hiperparâmetros**. As conclusões referem-se, portanto, à comparação de *configurações padrão*, e não à superioridade intrínseca dos algoritmos. Um eventual ajuste de hiperparâmetros exigiria validação cruzada aninhada (busca no laço interno, avaliação no laço externo); ajustar e avaliar nas mesmas dobras introduziria viés de seleção.

**Nota sobre inferência estatística.** As dobras de validação cruzada não são independentes (os conjuntos de treinamento se sobrepõem), o que compromete intervalos de confiança convencionais; os desvios-padrão reportados devem ser lidos como medida descritiva de estabilidade, não como base para inferência exata.

---

## Parte A — Classificação (Breast Cancer Wisconsin)

### Melhor Resultado Observado

**Dois indutores são reportados, em empate técnico:**

| Indutor | Taxa de Acerto (%) | F1_score | Precisão | Recall |
|---|---|---|---|---|
| **MLP (Perceptron Multicamadas)** | **97,72 ± 1,33** | 0,9753 ± 0,0145 | 0,9797 ± 0,0103 | 0,9723 ± 0,0189 |
| **SVM (`SVC`, kernel RBF)** | **97,72 ± 1,82** | 0,9754 ± 0,0196 | 0,9783 ± 0,0172 | 0,9732 ± 0,0224 |

*F1, precisão e recall em média macro; dispersões são desvios-padrão entre as cinco dobras.*

**Justificativa do empate técnico.** Os dois indutores são reportados conjuntamente porque nenhum critério defensável os separa:

1. **Taxa de acerto média idêntica** (97,72% em ambos, coincidindo até a quarta casa decimal), e o teste de postos sinalizados de Wilcoxon sobre as acurácias por dobra não distingue os modelos (W = 4,0; p = 0,875) — as diferenças por dobra alternam de sinal (MLP vence nas dobras 2 e 5, SVM nas dobras 1 e 3, com empate exato na dobra 4; ver tabela de estabilidade), comportamento típico de ruído amostral, não de superioridade sistemática.
2. **A diferença entre eles é uma ordem de grandeza menor que a variação entre dobras** de cada um (DP de 1,33 e 1,82 pontos percentuais, respectivamente).
3. **As demais métricas se dividem**: a SVM é marginalmente superior em F1 macro (0,9754 vs. 0,9753) e recall macro (0,9732 vs. 0,9723) — incluindo o recall da classe maligna (9 contra 10 falsos negativos; ver A.2) —, enquanto o MLP é marginalmente superior em precisão macro (0,9797 vs. 0,9783). Nenhum critério de desempate foi pré-declarado no protocolo, e qualquer escolha *post hoc* entre essas margens seria arbitrária.

Declarar um vencedor único, portanto, sobreinterpretaria diferenças menores que o ruído entre dobras.

### Estabilidade dos resultados (acurácia por dobra)

| Indutor | Dobra 1 | Dobra 2 | Dobra 3 | Dobra 4 | Dobra 5 | Média | DP |
|---|---|---|---|---|---|---|---|
| MLP | 0,9737 | 0,9649 | 0,9649 | 0,9912 | 0,9912 | 0,9772 | 0,0133 |
| SVM | 0,9912 | 0,9474 | 0,9737 | 0,9912 | 0,9823 | 0,9772 | 0,0182 |
| k-NN | 0,9825 | 0,9474 | 0,9386 | 0,9825 | 0,9646 | 0,9631 | 0,0200 |
| XGBoost | 0,9737 | 0,9474 | 0,9737 | 0,9561 | 0,9646 | 0,9631 | 0,0114 |
| Random Forest | 0,9649 | 0,9386 | 0,9561 | 0,9474 | 0,9735 | 0,9561 | 0,0138 |
| Bagging | 0,9474 | 0,9386 | 0,9561 | 0,9737 | 0,9558 | 0,9543 | 0,0130 |
| AdaBoost | 0,9649 | 0,9123 | 0,9649 | 0,9737 | 0,9469 | 0,9525 | 0,0245 |
| Naive Bayes | 0,9649 | 0,9035 | 0,9298 | 0,9298 | 0,9646 | 0,9385 | 0,0262 |
| Árvore de Decisão | 0,9298 | 0,8684 | 0,8860 | 0,9386 | 0,9292 | 0,9104 | 0,0312 |

Os valores por dobra alimentam os testes estatísticos reportados neste documento e permitem avaliar a estabilidade de cada indutor: a diferença entre MLP e SVM (e entre estes e o k-NN/XGBoost) é da mesma ordem de grandeza da variação entre dobras.

### A.1) Qual a taxa de acerto de cada classe?

**R:** Considerando as predições *out-of-fold* agregadas das cinco dobras, para os dois indutores reportados:

| Classe | MLP | SVM |
|---|---|---|
| **Maligna (0)** | 95,28% (202/212) | 95,75% (203/212) |
| **Benigna (1)** | 99,16% (354/357) | 98,88% (353/357) |

Em ambos os modelos, a classe maligna — minoritária e crítica no contexto da tarefa — concentra a maior parte dos erros; os dois indutores diferem em apenas uma instância maligna e uma benigna.

### A.2) Informe a matriz de confusão.

**R:** Matrizes de confusão agregadas das predições *out-of-fold* (linhas = classe real; colunas = classe prevista):

**MLP:**

| | Previsto: maligno (0) | Previsto: benigno (1) |
|---|---|---|
| **Real: maligno (0)** — 212 | 202 | 10 |
| **Real: benigno (1)** — 357 | 3 | 354 |

**SVM (grupo de empate):**

| | Previsto: maligno (0) | Previsto: benigno (1) |
|---|---|---|
| **Real: maligno (0)** — 212 | 203 | 9 |
| **Real: benigno (1)** — 357 | 4 | 353 |

No contexto abstrato da tarefa, os falsos negativos (tumores malignos classificados como benignos) tendem a apresentar custo potencialmente maior, por implicarem adiamento de investigação; os falsos positivos também têm custo não trivial — exames confirmatórios adicionais envolvem procedimento invasivo (biópsia), com ônus clínico e psicológico. Os custos efetivos, contudo, dependem do fluxo assistencial em que o modelo se inseriria, e nenhuma conclusão sobre uso clínico pode ser extraída de uma base pequena, curada e sem validação externa.

### A.3) Informe o valor dos parâmetros utilizados no treinamento.

**R:** Ambos os indutores reportados (`MLPClassifier` e `SVC` do scikit-learn) foram precedidos de padronização dos atributos (`StandardScaler` ajustado apenas nas dobras de treinamento). Os parâmetros correspondem aos **valores padrão das bibliotecas, sem busca sistemática de hiperparâmetros** (no MLP, apenas o número máximo de épocas foi elevado para assegurar convergência).

**MLP (`MLPClassifier`):**

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

**SVM (`SVC`):**

| Parâmetro | Valor |
|---|---|
| Kernel | RBF (base radial) |
| Regularização (C) | 1,0 |
| Coeficiente do kernel (γ) | `scale` = 1 / (n_atributos · var(X)) |
| Tolerância de parada | 10⁻³ |
| *Shrinking* | ativado |
| Limite de iterações | ilimitado (`max_iter=-1`) |
| Semente aleatória | 42 |

### A.4) Há diferença significativa entre acurácia e F1_score?

**R:** A diferença descritiva é pequena: acurácia média de 0,9772 (DP 0,0133) contra F1_score macro médio de 0,9753 (DP 0,0145) — **0,19 ponto percentual**, com o mesmo sinal nas cinco dobras (diferenças por dobra: 0,0016; 0,0030; 0,0034; 0,0006; 0,0007). Os valores referem-se ao MLP; o padrão na SVM é análogo (0,9772 vs. 0,9754).

Duas ressalvas impedem tratar essa comparação como teste de hipótese formal. Primeira: acurácia e F1 macro **não são estimativas intercambiáveis da mesma grandeza** — medem propriedades diferentes do desempenho —, de modo que perguntar se "diferem significativamente" não tem a mesma interpretação de comparar dois classificadores pela mesma métrica. Segunda: com apenas cinco pares, um teste pareado bilateral tem potência baixíssima — o menor p alcançável pelo teste de Wilcoxon nessa configuração é 0,0625, valor atingido aqui; um resultado "não significativo" seria, portanto, um piso de potência, e **não constitui evidência de equivalência**.

A leitura adequada é definicional: com o desbalanceamento moderado da base (37,3% malignos vs. 62,7% benignos), a acurácia é levemente inflada pelo bom desempenho na classe majoritária (benigna, 99,16% de acerto), ao passo que o F1 macro pondera as classes igualmente e penaliza a maior taxa de erro na classe maligna (4,72%). As métricas devem ser interpretadas segundo suas definições e sua sensibilidade ao desbalanceamento; para este problema, o F1 macro é a métrica mais informativa, por refletir melhor o desempenho na classe minoritária e mais relevante.

---

## Parte B — Regressão (Diabetes)

### Tabela de Resultados (Melhor Regressor entre os solicitados no enunciado)

| Indutor | Coeficiente de Determinação (R²) | Erro Médio Absoluto (MAE) |
|---|---|---|
| **MLP (`MLPRegressor`)** | **0,4686 ± 0,0598** | **44,05 ± 2,01** |

*Dispersões são desvios-padrão entre as cinco dobras. R² por dobra: 0,4895; 0,5226; 0,3686; 0,4991; 0,4631. MAE por dobra: 41,45; 43,02; 45,87; 46,27; 43,64.*

**Distinção necessária para leitura correta:**

- **Melhor entre os regressores solicitados no enunciado: MLP** — maior R² e menor MAE da lista (árvore de regressão, k-NN, SVR, MLP, Random Forest, Bagging, XGBoost).
- **Incluindo a baseline linear adicional (fora da lista do enunciado): a regressão Ridge obtém o maior R²** — 0,4791 (DP 0,0933) —, com MAE ligeiramente superior ao do MLP (44,24; DP 2,83).

Ordenação completa por R² entre os solicitados: MLP (0,4686) > Random Forest (0,4294) > k-NN (0,3912) > Bagging (0,3761) > XGBoost (0,3290) > SVR (0,1492) > Árvore de Regressão (−0,1325). A árvore isolada apresentou R² negativo — desempenho inferior ao de simplesmente predizer a média —, evidenciando sua alta variância sem regularização. O fato de uma baseline linear regularizada alcançar o maior R² reforça que a base possui sinal aproximadamente linear e limitado: o MAE de 44,05 corresponde a cerca de 13,7% da amplitude do alvo (25–346).

**Nota metodológica sobre o R².** Os valores reportados são **médias de R² calculados dobra a dobra**: em cada dobra, o R² usa como referência a média e a variância do alvo do próprio conjunto de teste. Não se trata, portanto, de uma fração de "variância explicada" calculada sobre a base completa, e a comparação qualitativa com o MAE médio (sensível a erros quadráticos vs. lineares) deve ser lida sob essa estrutura por dobra.

### B.1) Informe o valor dos parâmetros utilizados no treinamento do modelo.

**R:** O indutor reportado foi o `MLPRegressor` do scikit-learn, precedido de padronização dos atributos (`StandardScaler` ajustado apenas nas dobras de treinamento). Os parâmetros correspondem aos **valores padrão da biblioteca, sem busca sistemática de hiperparâmetros** (apenas o número máximo de épocas foi elevado para assegurar convergência):

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

A seção final da saída (`Respostas — enunciado original`) imprime exatamente os valores reportados neste documento: acurácia por dobra com desvios-padrão, o teste de empate entre MLP e SVM, as duas matrizes de confusão agregadas, os parâmetros dos modelos reportados, a comparação descritiva entre acurácia e F1, e os valores por dobra de R²/MAE — incluindo a referência Ridge fora da lista do enunciado.

---

*[← Atividade 1 (versão estendida, em inglês)](atividade-1.md) · [README](../README.md)*
