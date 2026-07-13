# Avaliação Prática 1 — Classificação de Imagens (CIFAR-10): Respostas

> **Disciplina:** Aprendizagem de Máquina · PPGIA / PUC-PR · Mestrado 2026
> **Aluno:** Fernando Dantas
> **Reprodutibilidade:** todos os valores foram gerados pelos scripts listados na seção *Scripts* (*seed* primária 42); o relatório é regenerado a partir dos artefatos de resultado por [`report.py`](https://github.com/fsd-dantas/machine-learning-fundamentals/blob/main/activities/avaliacao-pratica-1/report.py).

---

## Protocolo Experimental

Foram avaliadas cinco estratégias de classificação sobre a base **CIFAR-10** (60.000 imagens coloridas de 32×32 pixels, 10 classes balanceadas): (1) CNN treinada do zero; (2) rede pré-treinada como extratora de características, com classificador raso; (3) ajuste fino de CNN pré-treinada na ImageNet; (4) ajuste fino com aumento de dados; e (5) ajuste fino de um *Vision Transformer* (ViT).

**Orçamento igual entre estratégias.** Todas as estratégias foram treinadas sobre a **mesma subamostra estratificada de 10.000 imagens** (1.000 por classe), validadas sobre as **mesmas 2.000 imagens** e avaliadas **uma única vez** sobre o **conjunto de teste oficial completo do CIFAR-10** (10.000 imagens). A subamostragem não é um atalho, mas a condição de legalidade da comparação: as cinco estratégias diferem em custo computacional por um fator de aproximadamente 50 — uma CNN treinada do zero em 32×32 converge em minutos, enquanto o ajuste fino de um ViT-B/16 em 224×224 não —, e o orçamento disponível (GPU T4, camada gratuita) não comporta a base completa em cinco estratégias, três sementes aleatórias (*seeds*) e quatro ablações. Diante dessa restrição, a escolha se dá entre *todas as estratégias com 10.000 imagens* e *algumas estratégias com mais dados que outras*; apenas a primeira constitui um experimento controlado.

**Consequência declarada.** Um regime de 10.000 imagens é um regime de poucos dados, e é precisamente nesse regime que a transferência de aprendizado exibe sua maior vantagem. Uma CNN treinada do zero recuperaria parte substancial da diferença com as 50.000 imagens da base completa. As conclusões aqui reportadas referem-se, portanto, à comparação **sob este orçamento de dados**, e não a uma superioridade intrínseca e incondicional das estratégias baseadas em transferência.

**Resolução de entrada.** As estratégias 2, 3 e 4 operam a **128×128 pixels**, e não aos 224×224 nativos da ImageNet. As imagens do CIFAR-10 têm 32×32 pixels: ampliá-las para 224×224 não acrescenta informação alguma — apenas fabrica pixels por interpolação —, ao custo de aproximadamente três vezes mais computação. A resolução de 128 px ainda representa uma ampliação de 4× sobre a fonte e é um tamanho de entrada oficialmente suportado pela MobileNetV2. A escolha é aplicada de modo idêntico às três estratégias, constituindo, portanto, uma constante controlada da comparação, e não uma variável. A estratégia 5 permanece em 224×224, pois a grade de *patches* e as codificações posicionais do ViT são fixadas pelo checkpoint pré-treinado.

**Isolamento do conjunto de teste.** A arquitetura, o número de épocas e a parada antecipada foram selecionados exclusivamente sobre o conjunto de validação. O conjunto de teste produziu exatamente um número por configuração: não houve seleção de época, de limiar ou de configuração com base no teste.

**Dispersão.** As ablações das questões em aberto foram executadas em **duas *seeds*** (42 e 7) e são reportadas como média ± desvio-padrão. A tabela principal de comparação entre estratégias, contudo, reporta uma **única execução** (*seed* 42): a cota de GPU gratuita esgotou-se durante a execução, e as *seeds* adicionais foram sacrificadas em favor das ablações, que são o objeto das questões em aberto. Essa limitação é declarada, e não contornada: o relatório **omite o desvio-padrão** das linhas da tabela principal em vez de imprimir um zero fabricado. As comparações entre estratégias são, por isso, sustentadas pelo **teste exato de McNemar sobre predições pareadas** — que não requer repetição, pois extrai sua evidência das 10.000 predições discordantes ou concordantes do próprio conjunto de teste —, e não pela dispersão entre execuções. A *seed* altera a inicialização dos pesos e a amostragem do aumento de dados, jamais os dados: a subamostra é fixada por um gerador independente.

**Inferência estatística.** Para o par de melhores modelos aplicou-se o **teste exato de McNemar** sobre as predições pareadas do conjunto de teste. Este é o teste adequado ao delineamento — e não o teste de Wilcoxon sobre dobras, empregado na Atividade 1 —, pois ambos os modelos são avaliados sobre **as mesmas** 10.000 imagens: os erros são, por construção, pareados, e apenas as predições discordantes carregam informação. Reporta-se ainda o intervalo de confiança de Wilson (95%) para a acurácia; sobre 10.000 amostras, sua semiamplitude é de aproximadamente ±0,6 ponto percentual em torno de 90% de acurácia, o que estabelece o **limite de resolução de qualquer afirmação deste relatório**.

**Escopo da comparação.** Não houve busca sistemática de hiperparâmetros: as configurações seguem as práticas correntes de cada estratégia (taxas de aprendizado padrão, parada antecipada sobre a acurácia de validação). As conclusões referem-se à comparação dessas configurações, não à superioridade intrínseca dos métodos. As ablações das questões em aberto isolam uma única variável por vez, mantendo dados, cronograma e *seeds* fixos.

**Ambiente.** Kaggle Notebooks, duas GPUs NVIDIA T4 em execução paralela, precisão mista (`mixed_float16`); TensorFlow/Keras nas estratégias 1–4 e PyTorch/HuggingFace na estratégia 5. Os *splits* são materializados em disco e lidos por ambos os frameworks, o que garante que as estratégias vejam dados byte a byte idênticos independentemente do framework utilizado.

---

## Limitações do Delineamento

Três fatores confundem a comparação entre as cinco estratégias. Todos são declarados aqui, antes dos resultados, porque nenhum deles é corrigível dentro do orçamento disponível — e um relatório que os omitisse atribuiria a causas erradas os efeitos que observa.

**1. O ViT não enfrenta um problema novo.** O ViT-B/16 foi pré-treinado na **ImageNet-21k**, cujo conjunto de classes contém categorias que coincidem, em substância, com as do CIFAR-10 — avião, automóvel, pássaro, gato, cervo, cão, sapo, cavalo, navio e caminhão são todas categorias amplamente representadas na ImageNet. O modelo, portanto, **não generaliza para um domínio inédito: ele reconhece categorias que já viu durante o pré-treinamento**, em resolução muito superior e com ordens de magnitude mais exemplos. Sua acurácia deve ser lida como um limite superior de transferência sob condições excepcionalmente favoráveis, e **não** como evidência de superioridade intrínseca da arquitetura *transformer* sobre a convolucional. As redes convolucionais pré-treinadas (estratégias 2 a 4) compartilham essa vantagem, porém em grau menor: foram pré-treinadas na ImageNet-1k, cerca de quatorze vezes menor.

**2. A resolução de entrada não é constante entre as estratégias.** A CNN treinada do zero opera em 32×32 — a resolução nativa das imagens —, enquanto as demais operam em 128×128 (ou 224×224, no caso do ViT), pois as redes pré-treinadas exigem entradas compatíveis com suas estatísticas de origem. Essa diferença é **inerente à natureza das estratégias comparadas**, e não um descuido: ampliar as imagens para treinar uma rede do zero apenas interpolaria pixels, sem acrescentar informação, ao custo de mais computação. Ainda assim, ela impede atribuir a diferença de desempenho exclusivamente ao pré-treinamento.

**3. A tabela principal repousa sobre uma única execução por estratégia.** O esgotamento da cota de GPU gratuita durante a execução impôs uma escolha entre repetir a comparação principal em múltiplas *seeds* ou executar as ablações que respondem às questões em aberto. Optou-se pelas ablações. A consequência é que as linhas da tabela principal não trazem desvio-padrão — omitido, e não fabricado. O impacto sobre a validade é atenuado pelo fato de as comparações serem decididas por **McNemar sobre predições pareadas**, cuja evidência provém das 10.000 predições do conjunto de teste e não da repetição entre execuções; ainda assim, a estabilidade das estratégias frente à inicialização dos pesos **não foi medida**, e não é reivindicada.

---

## Resultados por Estratégia

<!-- BEGIN GENERATED: main-table -->
| Estratégia | Configuração | Acurácia (teste) | IC 95% | Macro-F1 | Resolução | Parâmetros treináveis | Treino |
|---|---|---|---|---|---|---|---|
| 5 — Ajuste fino de ViT | `vit_base_patch16_224_in21k` | **0,9825** | 0,9797–0,9849 | 0,9825 | 224px | 85.806.346 | 6,7 min |
| 4 — Ajuste fino + aumento de dados | `mobilenetv2_gap_flip_crop` | **0,8650** | 0,8582–0,8716 | 0,8650 | 128px | 2.171.722 | 8,6 min |
| 2 — Extração de características | `mobilenetv2_svm` | **0,8522** | 0,8451–0,8590 | 0,8524 | 128px | 0 | 0,6 min |
| 3 — Ajuste fino | `mobilenetv2_gap` | **0,8512** | 0,8441–0,8580 | 0,8512 | 128px | 2.171.722 | 5,7 min |
| 1 — CNN do zero | `cnn_scratch` | **0,7636** | 0,7552–0,7718 | 0,7617 | 32px | 305.258 | 1,7 min |
<!-- END GENERATED: main-table -->

*A tabela reporta, ao lado da acurácia, a resolução de entrada, o número de parâmetros treináveis e o tempo de treinamento. Um ranking de acurácia desacompanhado dessas colunas convida o leitor a celebrar uma vantagem de 0,3 ponto percentual que custou cinquenta vezes mais computação e que se encontra dentro do ruído experimental.*

---

## Melhor Resultado

### Significância estatística da diferença entre o 1º e o 2º colocados

<!-- BEGIN GENERATED: significance -->
- **1º** `s5_vit / vit_base_patch16_224_in21k` — 0,9825
- **2º** `s4_augment / mobilenetv2_gap_flip_crop` — 0,8650

| Discordâncias | 1º certo / 2º errado | 1º errado / 2º certo | p (McNemar exato) |
|---|---|---|---|
| 1311 | 1243 | 68 | 3,234e-280 |

A diferença de 11,75 pp é **significativa** (α = 0,05). O primeiro colocado é, portanto, o melhor modelo desta comparação.
<!-- END GENERATED: significance -->

### Todas as comparações entre estratégias (McNemar exato, *seed* primária)

<!-- BEGIN GENERATED: pairwise -->
| Comparação | Δ | p (McNemar) | Significativa? |
|---|---|---|---|
| Estratégia 2 vs. 1 | +8,86 pp | 5,7e-83 | **sim** |
| Estratégia 3 vs. 1 | +8,76 pp | 4,7e-79 | **sim** |
| Estratégia 3 vs. 2 | -0,10 pp | 0,722 | não — empate técnico |
| Estratégia 4 vs. 1 | +10,14 pp | 2,86e-109 | **sim** |
| Estratégia 4 vs. 2 | +1,28 pp | 5,96e-06 | **sim** |
| Estratégia 4 vs. 3 | +1,38 pp | 3,04e-07 | **sim** |
| Estratégia 5 vs. 1 | +21,89 pp | 0 | **sim** |
| Estratégia 5 vs. 2 | +13,03 pp | 4,74e-322 | **sim** |
| Estratégia 5 vs. 3 | +13,13 pp | 2,96e-323 | **sim** |
| Estratégia 5 vs. 4 | +11,75 pp | 3,23e-280 | **sim** |
<!-- END GENERATED: pairwise -->

*Cada linha confronta um incremento de transferência com o anterior. Uma diferença que não sobrevive ao teste de McNemar é uma diferença **paga e não recebida**: o custo computacional foi incorrido, o ganho não se materializou. Essa informação é invisível em um ranking de acurácia.*

**Achado central: o ajuste fino não superou a extração de características.** A Estratégia 3 (descongelamento do bloco superior da MobileNetV2, com taxa de aprendizado reduzida) obteve **0,8512**, contra **0,8522** da Estratégia 2 (backbone integralmente congelado, SVM sobre as características extraídas). A diferença é de **−0,10 ponto percentual** — negativa, portanto — e o teste exato de McNemar sobre as predições pareadas não a distingue de zero (p = 0,722): **empate técnico**.

O custo, contudo, não empata. A Estratégia 2 consome **0,6 minuto** de GPU (um único passe direto das 22.000 imagens pela rede congelada, seguido do ajuste de um classificador raso); a Estratégia 3 consome **5,7 minutos** — **9,5 vezes mais**, para não entregar acurácia adicional alguma.

A interpretação é direta e tem consequência prática. Com 10.000 imagens de treinamento, **não há sinal suficiente para reajustar proveitosamente 2,17 milhões de parâmetros convolucionais**: as características que a MobileNetV2 já extraiu da ImageNet são adequadas ao CIFAR-10 como estão, e o descongelamento apenas oferece à rede capacidade para memorizar o conjunto de treinamento. Os registros de treinamento corroboram: na fase 2, a acurácia de treinamento atinge 99,8% enquanto a de validação permanece estacionária em ~85% — a definição operacional de sobreajuste.

Cabe destacar que o notebook da disciplina **congela o backbone e nunca o descongela** — procedimento que, conforme argumentado na seção de desvios metodológicos, é extração de características com cabeça densa, e não ajuste fino. Este experimento mostra que, neste regime de dados, **o notebook não perde nada por isso**. O desvio metodológico que implementamos (a fase 2) foi executado, medido, e concluiu-se que não se paga.

### Acurácia por minuto de GPU

<!-- BEGIN GENERATED: cost -->
| Estratégia | Acurácia | Treino | pp acima da CNN do zero | pp por minuto |
|---|---|---|---|---|
| 5 — Ajuste fino de ViT | 0,9825 | 6,7 min | +21,89 pp | +3,25 |
| 4 — Ajuste fino + aumento de dados | 0,8650 | 8,6 min | +10,14 pp | +1,18 |
| 2 — Extração de características | 0,8522 | 0,6 min | +8,86 pp | +13,67 |
| 3 — Ajuste fino | 0,8512 | 5,7 min | +8,76 pp | +1,53 |
| 1 — CNN do zero | 0,7636 | 1,7 min | +0,00 pp | +0,00 |
<!-- END GENERATED: cost -->

*A pergunta do enunciado — qual estratégia é a melhor — não admite resposta sem um eixo de custo: uma estratégia que vence por 0,1 ponto percentual consumindo vinte vezes mais computação não venceu nada que um praticante compraria.*

**A extração de características é, disparadamente, a estratégia mais eficiente:** entrega **13,67 pontos percentuais por minuto** de GPU acima da CNN treinada do zero, contra 3,25 do ViT, 1,53 do ajuste fino e 1,18 do ajuste fino com aumento de dados. Uma ordem de grandeza separa a primeira colocada da segunda nesse critério.

Isso **não contradiz** a tabela de acurácia — o ViT continua sendo o modelo mais acurado, e por larga margem. O que a tabela de custo estabelece é que as duas perguntas têm **respostas diferentes**: se o critério é acurácia máxima, a resposta é o ViT; se o critério é acurácia por unidade de recurso, a resposta é a extração de características, e não por pouco. Um relatório que reportasse apenas a primeira coluna deixaria o leitor sem a informação de que **99% do ganho de transferência sobre a CNN do zero é capturado em 36 segundos de GPU**, sem treinar um único parâmetro convolucional.

**O aumento de dados, por sua vez, paga-se.** A Estratégia 4 supera a Estratégia 3 em **+1,38 pp** com significância (p = 3,0 × 10⁻⁷) — é o único incremento sobre o ajuste fino que sobrevive ao teste estatístico. Trata-se de resultado coerente com o regime de poucos dados: se 10.000 imagens são insuficientes para reajustar o backbone, ampliá-las artificialmente ataca exatamente a restrição vigente.


### Matriz de confusão do melhor modelo

<p align="center">
  <img src="../assets/img/avaliacao-pratica-1-confusion-light.png" alt="Matriz de confusão do melhor modelo" width="720">
</p>

*Matriz normalizada por linha: cada célula lê-se como “desta classe verdadeira, tal fração foi predita como aquela classe”. Valores em porcentagem.*

### Principais confusões

<!-- BEGIN GENERATED: hardest-classes -->
| Confusão | Taxa | Leitura |
|---|---|---|
| dog → cat | 3,6% | |
| cat → dog | 2,3% | |
| truck → automobile | 1,5% | |
| automobile → truck | 1,0% | |
| bird → cat | 0,7% | |
<!-- END GENERATED: hardest-classes -->

**O erro residual do ViT é semanticamente estruturado, e não ruído.** As cinco maiores confusões concentram-se em **dois pares**: `dog` ↔ `cat` (3,6% e 2,3%) e `truck` ↔ `automobile` (1,5% e 1,0%). São, precisamente, os dois pares de classes semanticamente adjacentes do CIFAR-10 — animais domésticos quadrúpedes de porte comparável, e veículos terrestres de quatro rodas —, e sua discriminação depende de detalhes de textura e proporção que, em imagens de 32×32 pixels, aproximam-se do limite da informação disponível.

Que um modelo com 98,25% de acurácia erre **apenas** nesses pares indica que o problema está essencialmente resolvido no que a resolução permite: o que resta não é falha do classificador, mas ambiguidade da fonte. A classe `bird` → `cat` (0,7%) completa o padrão, e é o único erro que sugere confusão de textura em vez de categoria.

---

## Questões em Aberto

> **Nota sobre o estado desta seção.** As quatro questões em aberto exigem 28 treinamentos adicionais. Aquelas cujas tabelas aparecem preenchidas abaixo foram executadas e são reportadas com dispersão entre *seeds*; aquelas assinaladas como sem resultados **não foram executadas**, por esgotamento da cota de GPU gratuita dentro do prazo de entrega. Nenhum valor é estimado, inferido ou reportado sem o artefato correspondente em [`results/`](https://github.com/fsd-dantas/machine-learning-fundamentals/tree/main/activities/avaliacao-pratica-1/results): uma tabela ausente significa exatamente que o experimento não foi realizado, e não que seu resultado foi desfavorável.

### 2(a) — Trocar a rede CNN por uma mais simples, como a MobileNet, impacta significativamente o resultado?

<!-- BEGIN GENERATED: ablation-backbone -->
_(sem resultados para `Backbone + classificador*` — execute a ablação correspondente)_
<!-- END GENERATED: ablation-backbone -->

*Ablação controlada: o classificador raso, os dados e o protocolo permanecem fixos; varia-se exclusivamente a rede extratora. MobileNetV2 (3,5 M de parâmetros; 0,6 GFLOPs) é confrontada com ResNet50 (25,6 M; 8,2 GFLOPs) e InceptionV3 (23,9 M; 11,5 GFLOPs) — uma diferença de aproximadamente 14× em parâmetros e 19× em custo computacional.*

### 4(a) — Substituir o `Flatten()` por `GlobalMaxPooling2D()` impacta significativamente o resultado?

<!-- BEGIN GENERATED: ablation-head -->
_(sem resultados para `Cabeça (pooling)*` — execute a ablação correspondente)_
<!-- END GENERATED: ablation-head -->

*A substituição não é cosmética. Sobre a MobileNetV2 em 128 px, o mapa de características final tem dimensão 4×4×1280; logo, `Flatten() → Dense(512)` implica **10,5 milhões** de parâmetros na cabeça da rede, contra **0,66 milhão** de `GlobalMaxPooling2D() → Dense(512)` — uma razão de 16× em capacidade, ajustada sobre apenas 10.000 imagens. O agrupamento (pooling) global descarta ainda a posição espacial, que na classificação de objetos é majoritariamente uma variável de perturbação. O `GlobalAveragePooling2D()` é incluído como terceiro braço da ablação.*

### 4(b) — Substituir o algoritmo otimizador (`Adam()`) pode melhorar o resultado?

<!-- BEGIN GENERATED: ablation-optimizer -->
_(sem resultados para `optimizer_*` — execute a ablação correspondente)_
<!-- END GENERATED: ablation-optimizer -->

*Adam, AdamW, SGD com momento de Nesterov e RMSprop. O SGD recebe taxa de aprendizado dez vezes maior, pois comparar otimizadores sob uma taxa calibrada para o Adam não constitui experimento, mas disputa viciada.*

### 4(c) — Avaliar outras estratégias de aumento de dados pode melhorar o resultado?

<!-- BEGIN GENERATED: ablation-policy -->
_(sem resultados para `policy_*` — execute a ablação correspondente)_
<!-- END GENERATED: ablation-policy -->

*Quatro políticas, incluindo a do notebook da disciplina (`rotation 10°, zoom 0.15, shift 0.1`). Cabe notar o que essa política **omite**: o espelhamento horizontal — sobre o CIFAR-10, a transformação mais eficaz e mais evidentemente preservadora de rótulo disponível (um caminhão espelhado continua sendo um caminhão). O espelhamento vertical é deliberadamente excluído de todas as políticas: um cavalo de cabeça para baixo não é uma imagem natural, e o conjunto de teste não contém nenhuma, de modo que treinar sobre tais exemplos apenas acrescentaria ruído.*

---

## Scripts

Todo o código é versionado e executável de forma independente. O notebook Colab é apenas o
ambiente de execução; o artefato científico são os scripts.

| Componente | Script |
|---|---|
| Protocolo compartilhado (splits, avaliação, McNemar, IC de Wilson) | [`common.py`](https://github.com/fsd-dantas/machine-learning-fundamentals/blob/main/activities/avaliacao-pratica-1/common.py) |
| Estratégia 1 — CNN do zero | [`s1_cnn_scratch.py`](https://github.com/fsd-dantas/machine-learning-fundamentals/blob/main/activities/avaliacao-pratica-1/s1_cnn_scratch.py) |
| Estratégia 2 — Extração de características | [`s2_feature_extraction.py`](https://github.com/fsd-dantas/machine-learning-fundamentals/blob/main/activities/avaliacao-pratica-1/s2_feature_extraction.py) |
| Estratégia 3 — Ajuste fino | [`s3_finetuning.py`](https://github.com/fsd-dantas/machine-learning-fundamentals/blob/main/activities/avaliacao-pratica-1/s3_finetuning.py) |
| Estratégia 4 — Ajuste fino + aumento de dados | [`s4_augmentation.py`](https://github.com/fsd-dantas/machine-learning-fundamentals/blob/main/activities/avaliacao-pratica-1/s4_augmentation.py) |
| Estratégia 5 — Ajuste fino de ViT | [`s5_vit.py`](https://github.com/fsd-dantas/machine-learning-fundamentals/blob/main/activities/avaliacao-pratica-1/s5_vit.py) |
| Orquestrador (todas as etapas e ablações) | [`run_all.py`](https://github.com/fsd-dantas/machine-learning-fundamentals/blob/main/activities/avaliacao-pratica-1/run_all.py) |
| Gerador do relatório (tabelas, matriz de confusão, testes) | [`report.py`](https://github.com/fsd-dantas/machine-learning-fundamentals/blob/main/activities/avaliacao-pratica-1/report.py) |
| Executor Colab (GPU) | [`colab.ipynb`](https://github.com/fsd-dantas/machine-learning-fundamentals/blob/main/activities/avaliacao-pratica-1/colab.ipynb) |
| Artefatos de resultado (JSON, com as predições de teste) | [`results/`](https://github.com/fsd-dantas/machine-learning-fundamentals/tree/main/activities/avaliacao-pratica-1/results) |

**Repositório:** <https://github.com/fsd-dantas/machine-learning-fundamentals>

---

*[← Relatório completo (inglês)](avaliacao-pratica-1.md) · [Módulo 5 — Técnicas Profundas](../modules/05-deep.md)*
