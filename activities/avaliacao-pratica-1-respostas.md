# Avaliação Prática 1 — Classificação de Imagens (CIFAR-10): Respostas

> **Disciplina:** Aprendizagem de Máquina · PPGIA / PUC-PR · Mestrado 2026
> **Aluno:** Fernando Dantas
> **Reprodutibilidade:** todos os valores foram gerados pelos scripts listados na seção *Scripts* (semente primária 42); o relatório é regenerado a partir dos artefatos de resultado por [`report.py`](https://github.com/fsd-dantas/machine-learning-fundamentals/blob/main/activities/avaliacao-pratica-1/report.py).

---

## Protocolo Experimental

Foram avaliadas cinco estratégias de classificação sobre a base **CIFAR-10** (60.000 imagens coloridas de 32×32 pixels, 10 classes balanceadas): (1) CNN treinada do zero; (2) rede pré-treinada como extratora de características, com classificador raso; (3) ajuste fino de CNN pré-treinada na ImageNet; (4) ajuste fino com aumento de dados; e (5) ajuste fino de um *Vision Transformer* (ViT).

**Orçamento igual entre estratégias.** Todas as estratégias foram treinadas sobre a **mesma subamostra estratificada de 10.000 imagens** (1.000 por classe), validadas sobre as **mesmas 2.000 imagens** e avaliadas **uma única vez** sobre o **conjunto de teste oficial completo do CIFAR-10** (10.000 imagens). A subamostragem não é um atalho, mas a condição de legalidade da comparação: as cinco estratégias diferem em custo computacional por um fator de aproximadamente 50 — uma CNN treinada do zero em 32×32 converge em minutos, enquanto o ajuste fino de um ViT-B/16 em 224×224 não —, e o orçamento disponível (GPU T4, camada gratuita) não comporta a base completa em cinco estratégias, três sementes e quatro ablações. Diante dessa restrição, a escolha se dá entre *todas as estratégias com 10.000 imagens* e *algumas estratégias com mais dados que outras*; apenas a primeira constitui um experimento controlado.

**Consequência declarada.** Um regime de 10.000 imagens é um regime de poucos dados, e é precisamente nesse regime que a transferência de aprendizado exibe sua maior vantagem. Uma CNN treinada do zero recuperaria parte substancial da diferença com as 50.000 imagens da base completa. As conclusões aqui reportadas referem-se, portanto, à comparação **sob este orçamento de dados**, e não a uma superioridade intrínseca e incondicional das estratégias baseadas em transferência.

**Isolamento do conjunto de teste.** A arquitetura, o número de épocas e a parada antecipada foram selecionados exclusivamente sobre o conjunto de validação. O conjunto de teste produziu exatamente um número por configuração: não houve seleção de época, de limiar ou de configuração com base no teste.

**Dispersão.** Toda rede treinada foi executada em **três sementes** (42, 7 e 2024) e é reportada como média ± desvio-padrão. A semente altera a inicialização dos pesos e a amostragem do aumento de dados, jamais os dados: a subamostra é fixada por um gerador independente. Uma diferença entre configurações menor que a dispersão entre sementes não é reportada como resultado.

**Inferência estatística.** Para o par de melhores modelos aplicou-se o **teste exato de McNemar** sobre as predições pareadas do conjunto de teste. Este é o teste adequado ao delineamento — e não o teste de Wilcoxon sobre dobras, empregado na Atividade 1 —, pois ambos os modelos são avaliados sobre **as mesmas** 10.000 imagens: os erros são, por construção, pareados, e apenas as predições discordantes carregam informação. Reporta-se ainda o intervalo de confiança de Wilson (95%) para a acurácia; sobre 10.000 amostras, sua semiamplitude é de aproximadamente ±0,6 ponto percentual em torno de 90% de acurácia, o que estabelece o **limite de resolução de qualquer afirmação deste relatório**.

**Escopo da comparação.** Não houve busca sistemática de hiperparâmetros: as configurações seguem as práticas correntes de cada estratégia (taxas de aprendizado padrão, parada antecipada sobre a acurácia de validação). As conclusões referem-se à comparação dessas configurações, não à superioridade intrínseca dos métodos. As ablações das questões em aberto isolam uma única variável por vez, mantendo dados, cronograma e sementes fixos.

**Ambiente.** Google Colab, GPU NVIDIA T4, precisão mista (`mixed_float16`); TensorFlow/Keras nas estratégias 1–4 e PyTorch/HuggingFace na estratégia 5. Os *splits* são materializados em disco e lidos por ambos os arcabouços, o que garante que as estratégias vejam dados byte a byte idênticos independentemente do arcabouço utilizado.

---

## Resultados por Estratégia

<!-- BEGIN GENERATED: main-table -->
_(pendente — executar `python run_all.py --stage core --stage seeds`)_
<!-- END GENERATED: main-table -->

*A tabela reporta, ao lado da acurácia, a resolução de entrada, o número de parâmetros treináveis e o tempo de treinamento. Um ranking de acurácia desacompanhado dessas colunas convida o leitor a celebrar uma vantagem de 0,3 ponto percentual que custou cinquenta vezes mais computação e que se encontra dentro do ruído experimental.*

---

## Melhor Resultado

### Significância estatística da diferença entre o 1º e o 2º colocados

<!-- BEGIN GENERATED: significance -->
_(pendente)_
<!-- END GENERATED: significance -->

### Matriz de confusão do melhor modelo

<p align="center">
  <img src="../assets/img/avaliacao-pratica-1-confusion-light.png" alt="Matriz de confusão do melhor modelo" width="720">
</p>

*Matriz normalizada por linha: cada célula lê-se como “desta classe verdadeira, tal fração foi predita como aquela classe”. Valores em porcentagem.*

### Principais confusões

<!-- BEGIN GENERATED: hardest-classes -->
_(pendente)_
<!-- END GENERATED: hardest-classes -->

---

## Questões em Aberto

### 2(a) — Trocar a rede CNN por uma mais simples, como a MobileNet, impacta significativamente o resultado?

<!-- BEGIN GENERATED: ablation-backbone -->
_(pendente — executar `python s2_feature_extraction.py --ablation`)_
<!-- END GENERATED: ablation-backbone -->

*Ablação controlada: o classificador raso, os dados e o protocolo permanecem fixos; varia-se exclusivamente a rede extratora. MobileNetV2 (3,5 M de parâmetros; 0,6 GFLOPs) é confrontada com ResNet50 (25,6 M; 8,2 GFLOPs) e InceptionV3 (23,9 M; 11,5 GFLOPs) — uma diferença de aproximadamente 14× em parâmetros e 19× em custo computacional.*

### 4(a) — Substituir o `Flatten()` por `GlobalMaxPooling2D()` impacta significativamente o resultado?

<!-- BEGIN GENERATED: ablation-head -->
_(pendente — executar `python s3_finetuning.py --ablation-head`)_
<!-- END GENERATED: ablation-head -->

*A substituição não é cosmética. Sobre a MobileNetV2 em 224 px, o mapa de características final tem dimensão 7×7×1280; logo, `Flatten() → Dense(512)` implica **32,1 milhões** de parâmetros na cabeça da rede, contra **0,66 milhão** de `GlobalMaxPooling2D() → Dense(512)` — uma razão de 49× em capacidade, ajustada sobre apenas 10.000 imagens. O agrupamento (pooling) global descarta ainda a posição espacial, que na classificação de objetos é majoritariamente uma variável de perturbação. O `GlobalAveragePooling2D()` é incluído como terceiro braço da ablação.*

### 4(b) — Substituir o algoritmo otimizador (`Adam()`) pode melhorar o resultado?

<!-- BEGIN GENERATED: ablation-optimizer -->
_(pendente — executar `python s4_augmentation.py --ablation-optimizer`)_
<!-- END GENERATED: ablation-optimizer -->

*Adam, AdamW, SGD com momento de Nesterov e RMSprop. O SGD recebe taxa de aprendizado dez vezes maior, pois comparar otimizadores sob uma taxa calibrada para o Adam não constitui experimento, mas disputa viciada.*

### 4(c) — Avaliar outras estratégias de aumento de dados pode melhorar o resultado?

<!-- BEGIN GENERATED: ablation-policy -->
_(pendente — executar `python s4_augmentation.py --ablation-policy`)_
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
