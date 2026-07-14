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

**Cobertura experimental.** A cobertura entre estratégias é desigual, e é declarada tal como é. A Estratégia 5 (ViT) dispõe de **três execuções** (*seeds* 42, 7 e 2024) e a Estratégia 3, de **duas** (42 e 7, herdadas da ablação de cabeça); as Estratégias 1, 2 e 4 repousam sobre **execução única** (*seed* 42). As ablações dispõem de duas *seeds* por braço, à exceção da ablação de backbone (uma). O relatório **omite o desvio-padrão onde ele não foi medido**, em vez de imprimir um zero fabricado. A origem dessa assimetria é a restrição computacional documentada em *Limitações do Delineamento*.

A validade das comparações não repousa, contudo, sobre a repetição. Ela repousa sobre o **teste exato de McNemar aplicado a predições pareadas**: como todos os modelos são avaliados sobre as **mesmas** 10.000 imagens, seus erros são pareados por construção, e a evidência é extraída das concordâncias e discordâncias entre eles — não da variância entre execuções. A *seed* altera a inicialização dos pesos e a amostragem do aumento de dados, jamais os dados: a subamostra é fixada por um gerador independente.

**Estabilidade do modelo vencedor.** As três execuções do ViT produziram 0,9825, 0,9815 e 0,9813 — **desvio-padrão de 0,0006**, ou 0,06 ponto percentual. Trata-se do modelo mais estável do experimento: sua dispersão é **oito vezes inferior** ao ruído de treinamento medido nas redes convolucionais (0,46 pp, adiante) e **sessenta vezes inferior** à do SGD (investigação 4b). O ajuste fino de um *transformer* pré-treinado, neste regime, é notavelmente insensível à inicialização — o que é coerente com a hipótese de que o pré-treinamento, e não a otimização, responde pela maior parte de seu desempenho.

**Variabilidade entre execuções idênticas.** A Estratégia 3, na *seed* 42, foi treinada duas vezes sob configuração rigorosamente idêntica, obtendo **0,8512** e **0,8558**. A diferença de **0,46 ponto percentual** decorre do não-determinismo dos núcleos cuDNN em GPU e fornece uma **estimativa empírica do ruído de treinamento** neste experimento. Trata-se de valor da mesma ordem de grandeza das diferenças que separam as Estratégias 2, 3 e 4, o que estabelece um piso de resolução para qualquer comparação entre elas — e justifica que os vereditos sejam emitidos por teste estatístico pareado, e não por leitura direta das acurácias.

**Inferência estatística.** Para o par de melhores modelos aplicou-se o **teste exato de McNemar** sobre as predições pareadas do conjunto de teste. Este é o teste adequado ao delineamento — e não o teste de Wilcoxon sobre dobras, empregado na Atividade 1 —, pois ambos os modelos são avaliados sobre **as mesmas** 10.000 imagens: os erros são, por construção, pareados, e apenas as predições discordantes carregam informação. Reporta-se ainda o intervalo de confiança de Wilson (95%) para a acurácia; sobre 10.000 amostras, sua semiamplitude é de aproximadamente ±0,6 ponto percentual em torno de 90% de acurácia, o que estabelece o **limite de resolução de qualquer afirmação deste relatório**.

**Escopo da comparação.** Não houve busca sistemática de hiperparâmetros: as configurações seguem as práticas correntes de cada estratégia (taxas de aprendizado padrão, parada antecipada sobre a acurácia de validação). As conclusões referem-se à comparação dessas configurações, não à superioridade intrínseca dos métodos. As ablações das investigações complementares isolam uma única variável por vez, mantendo dados, cronograma e *seeds* fixos.

**Ambiente.** Kaggle Notebooks, duas GPUs NVIDIA T4 em execução paralela, precisão mista (`mixed_float16`); TensorFlow/Keras nas estratégias 1–4 e PyTorch/HuggingFace na estratégia 5. Os *splits* são materializados em disco e lidos por ambos os frameworks, o que garante que as estratégias vejam dados byte a byte idênticos independentemente do framework utilizado.

---

## Limitações do Delineamento

Quatro fatores limitam a interpretação dos resultados. Todos são declarados aqui, antes dos resultados, porque nenhum deles é corrigível dentro do orçamento disponível — e um relatório que os omitisse atribuiria a causas erradas os efeitos que observa.

**1. O ViT não enfrenta um problema novo.** O ViT-B/16 foi pré-treinado na **ImageNet-21k**, cujo conjunto de classes contém categorias que coincidem, em substância, com as do CIFAR-10 — avião, automóvel, pássaro, gato, cervo, cão, sapo, cavalo, navio e caminhão são todas categorias amplamente representadas na ImageNet. O modelo, portanto, **não generaliza para um domínio inédito: ele reconhece categorias que já viu durante o pré-treinamento**, em resolução muito superior e com ordens de magnitude mais exemplos. Sua acurácia deve ser lida como um limite superior de transferência sob condições excepcionalmente favoráveis, e **não** como evidência de superioridade intrínseca da arquitetura *transformer* sobre a convolucional. As redes convolucionais pré-treinadas (estratégias 2 a 4) compartilham essa vantagem, porém em grau menor: foram pré-treinadas na ImageNet-1k, cerca de quatorze vezes menor.

**2. A resolução de entrada não é constante entre as estratégias.** A CNN treinada do zero opera em 32×32 — a resolução nativa das imagens —, enquanto as demais operam em 128×128 (ou 224×224, no caso do ViT), pois as redes pré-treinadas exigem entradas compatíveis com suas estatísticas de origem. Essa diferença é **inerente à natureza das estratégias comparadas**, e não um descuido: ampliar as imagens para treinar uma rede do zero apenas interpolaria pixels, sem acrescentar informação, ao custo de mais computação. Ainda assim, ela impede atribuir a diferença de desempenho exclusivamente ao pré-treinamento.

**3. Restrição computacional e cobertura desigual.** O experimento foi conduzido em infraestrutura de GPU de acesso gratuito, com cota limitada. Em consequência, a cobertura entre estratégias é desigual (três execuções para a Estratégia 5, duas para a Estratégia 3, uma para as demais). **A estabilidade das Estratégias 1, 2 e 4 frente à inicialização dos pesos não foi medida, e não é reivindicada**; a inferência é sustentada pelo teste pareado de McNemar, cuja evidência provém das 10.000 predições do conjunto de teste, e pela estimativa de ruído de treinamento apresentada no protocolo.

**4. Seleção de modelo interna a cada fase.** A parada antecipada seleciona o melhor modelo **dentro** de cada uma das duas fases de treinamento (cabeça congelada; descongelamento do bloco superior), e não **globalmente entre elas**. O delineamento pressupõe, portanto, que a fase 2 aprimore o resultado da fase 1 — pressuposto válido em vinte e seis das vinte e sete execuções de ajuste fino conduzidas, e violado em uma (SGD, *seed* 7), conforme documentado na investigação 4(b). A correção é direta e está registrada como trabalho futuro; sua ausência afeta apenas execuções cuja fase 2 degrada o modelo, e o faz na direção conservadora.

---

## Resultados por Estratégia

<!-- BEGIN GENERATED: main-table -->
| Estratégia | Configuração | Acurácia (teste) | IC 95% | Macro-F1 | Resolução | Parâmetros treináveis | Treino |
|---|---|---|---|---|---|---|---|
| 5 — Ajuste fino de ViT | `vit_base_patch16_224_in21k` | **0,9818 ± 0,0006** | 0,9785–0,9849 | 0,9818 ± 0,0007 | 224px | 85.806.346 | 6,8 min |
| 4 — Ajuste fino + aumento de dados | `mobilenetv2_gap_flip_crop` | **0,8650** | 0,8582–0,8716 | 0,8650 | 128px | 2.171.722 | 8,6 min |
| 3 — Ajuste fino | `mobilenetv2_gap` | **0,8576 ± 0,0025** | 0,8488–0,8660 | 0,8575 ± 0,0024 | 128px | 2.171.722 | 3,3 min |
| 2 — Extração de características | `mobilenetv2_svm` | **0,8522** | 0,8451–0,8590 | 0,8524 | 128px | 0 | 0,6 min |
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
| Estratégia 3 vs. 1 | +9,22 pp | 8,16e-89 | **sim** |
| Estratégia 3 vs. 2 | +0,36 pp | 0,159 | não — empate técnico |
| Estratégia 4 vs. 1 | +10,14 pp | 2,86e-109 | **sim** |
| Estratégia 4 vs. 2 | +1,28 pp | 5,96e-06 | **sim** |
| Estratégia 4 vs. 3 | +0,92 pp | 0,000397 | **sim** |
| Estratégia 5 vs. 1 | +21,89 pp | 0 | **sim** |
| Estratégia 5 vs. 2 | +13,03 pp | 4,74e-322 | **sim** |
| Estratégia 5 vs. 3 | +12,67 pp | 6,83e-312 | **sim** |
| Estratégia 5 vs. 4 | +11,75 pp | 3,23e-280 | **sim** |
<!-- END GENERATED: pairwise -->

*Cada linha confronta um incremento de transferência com o anterior. Uma diferença que não sobrevive ao teste de McNemar é uma diferença **paga e não recebida**: o custo computacional foi incorrido, o ganho não se materializou. Essa informação é invisível em um ranking de acurácia.*

**Achado central: o ajuste fino não supera significativamente a extração de características.** Ambas as estratégias compartilham o **mesmo backbone** (MobileNetV2) e a **mesma resolução** (128 px) — o que muda entre elas é exclusivamente o descongelamento do bloco superior. Na *seed* 42, a Estratégia 3 obtém **0,8558** contra **0,8522** da Estratégia 2: uma diferença de **+0,36 ponto percentual** que o teste exato de McNemar **não distingue de zero** (p = 0,159; 327 discordâncias a favor contra 291 contra). **Empate técnico.**

O custo, contudo, não empata: a Estratégia 2 consome **0,6 minuto** de GPU — um único passe direto das 22.000 imagens pela rede congelada, seguido do ajuste de uma SVM —, contra **3,3 minutos** da Estratégia 3. **Cinco vezes mais computação, para um ganho que não sobrevive ao teste estatístico.**

A interpretação tem consequência prática. Com 10.000 imagens de treinamento, **não há sinal suficiente para reajustar proveitosamente 2,17 milhões de parâmetros convolucionais**: as características que a MobileNetV2 já extraiu da ImageNet servem ao CIFAR-10 como estão, e o descongelamento oferece à rede sobretudo capacidade de memorizar o conjunto de treinamento. Os registros de treinamento corroboram: na fase 2, a acurácia de treinamento atinge 99,8% enquanto a de validação permanece estacionária em torno de 85% — a definição operacional de sobreajuste.

Cabe destacar que o notebook da disciplina **congela o backbone e nunca o descongela** — procedimento que, conforme a seção de desvios metodológicos, é extração de características com cabeça densa, e não ajuste fino. Este experimento indica que, **neste regime de dados, tal procedimento não acarreta perda de desempenho**. O desvio metodológico aqui implementado — a fase 2 de descongelamento — foi executado, mensurado, e não produziu ganho estatisticamente detectável.

### Acurácia por minuto de GPU

<!-- BEGIN GENERATED: cost -->
| Estratégia | Acurácia | Treino | pp acima da CNN do zero | pp por minuto |
|---|---|---|---|---|
| 5 — Ajuste fino de ViT | 0,9818 | 6,8 min | +21,82 pp | +3,20 |
| 4 — Ajuste fino + aumento de dados | 0,8650 | 8,6 min | +10,14 pp | +1,18 |
| 3 — Ajuste fino | 0,8576 | 3,3 min | +9,40 pp | +2,86 |
| 2 — Extração de características | 0,8522 | 0,6 min | +8,86 pp | +14,45 |
| 1 — CNN do zero | 0,7636 | 1,7 min | +0,00 pp | +0,00 |
<!-- END GENERATED: cost -->

*A pergunta do enunciado — qual estratégia é a melhor — não admite resposta sem um eixo de custo. Uma vantagem de 0,1 ponto percentual obtida ao preço de vinte vezes mais computação não constitui superioridade sob qualquer critério de engenharia aplicável.*

**A extração de características é, por larga margem, a estratégia mais eficiente:** entrega **14,45 pontos percentuais por minuto** de GPU acima da CNN treinada do zero, contra 3,25 do ViT, 2,86 do ajuste fino e 1,18 do ajuste fino com aumento de dados. Uma ordem de grandeza separa a primeira colocada da segunda nesse critério.

Isso **não contradiz** a tabela de acurácia — o ViT permanece o modelo mais acurado, e por larga margem. O que o eixo de custo estabelece é que as duas perguntas têm **respostas diferentes**: se o critério é acurácia máxima, a resposta é o ViT; se é acurácia por unidade de recurso, a resposta é a extração de características. Um relatório que reportasse apenas a primeira coluna omitiria do leitor o fato de que **81% do ganho de transferência sobre a CNN do zero (8,86 dos 10,94 pp atingidos pela melhor estratégia convolucional) é capturado em 36 segundos de GPU**, sem treinar um único parâmetro convolucional.

**O aumento de dados, por sua vez, justifica seu custo.** A Estratégia 4 supera a Estratégia 3 em **+0,92 pp** com significância (p = 4,0 × 10⁻⁴) — é o único incremento sobre o ajuste fino que sobrevive ao teste. Resultado coerente com o regime de poucos dados: se 10.000 imagens são insuficientes para reajustar o backbone, ampliá-las artificialmente ataca precisamente a restrição vigente.

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

**O erro residual do ViT é semanticamente estruturado, e não disperso.** As cinco maiores confusões concentram-se em **dois pares**: `dog` ↔ `cat` (3,6% e 2,3%) e `truck` ↔ `automobile` (1,5% e 1,0%). São os dois pares de classes semanticamente adjacentes do CIFAR-10 — animais domésticos quadrúpedes de porte comparável, e veículos terrestres de quatro rodas.

A estrutura do erro é, portanto, **compatível com a hipótese** de que o que resta a discriminar depende de detalhes de textura e proporção pouco preservados a 32×32 pixels. Cabe, porém, distinguir o que a evidência sustenta do que ela não sustenta: **a matriz de confusão exibe um padrão, e não demonstra um limite intrínseco da base.** Estabelecer que esses erros são irredutíveis exigiria evidência que este experimento não produz — por exemplo, a taxa de concordância entre anotadores humanos sobre as imagens confundidas, ou o desempenho de um modelo de capacidade muito superior sobre o mesmo par de classes. A afirmação defensável é mais modesta: **o erro remanescente não é aleatório, e concentra-se onde a semelhança semântica é maior**.

---

## Investigações Complementares

O enunciado formula quatro questões de investigação. **Todas foram conduzidas e respondidas.** As conclusões sintéticas são apresentadas a seguir; cada seção subsequente detalha o delineamento, os resultados e os testes de significância.

| Questão | Evidência | Conclusão |
|---|---|---|
| **2(a)** Trocar a CNN por uma mais simples (MobileNet) impacta significativamente o resultado? | 6 configurações (3 backbones × 2 classificadores), 1 *seed* | **Sim.** A adoção da MobileNetV2 custa **2,39 pp** de acurácia frente à ResNet50 (p = 4×10⁻¹⁰). A InceptionV3, contudo, a mais custosa das três, é a de pior desempenho: **capacidade não prediz qualidade de transferência**. |
| **4(a)** Substituir `Flatten()` por `GlobalMaxPooling2D()` impacta significativamente o resultado? | 3 configurações, 2 *seeds* por braço | **Sim, e favoravelmente:** +1,19 pp (p = 9×10⁻⁵). O melhor braço, porém, é o `GlobalAveragePooling2D()`, em empate técnico com o máximo — o mérito é do **agrupamento global** como classe de operação. |
| **4(b)** Substituir o otimizador (`Adam()`) pode melhorar o resultado? | 4 otimizadores × 2 *seeds* | **Não.** Nenhum supera o Adam significativamente (AdamW: +0,08 pp; p = 0,087). O achado está na dispersão: o SGD varia **5,4 pp** entre *seeds* — 29× o desvio do AdamW. |
| **4(c)** Outras estratégias de aumento de dados podem melhorar o resultado? | 4 políticas × 2 *seeds* | **Sim, mas não como previsto.** A política da disciplina empata com a melhor; o espelhamento horizontal isolado é a **pior**. A hipótese registrada por este relatório foi **refutada**. |

Nenhum valor é estimado, inferido ou reportado sem o artefato correspondente em [`results/`](https://github.com/fsd-dantas/machine-learning-fundamentals/tree/main/activities/avaliacao-pratica-1/results). **A ausência de uma tabela indica que o experimento não foi conduzido — e não que seu resultado tenha sido desfavorável.**

### 2(a) — Trocar a rede CNN por uma mais simples, como a MobileNet, impacta significativamente o resultado?

*Investigação conduzida. Seis configurações, uma* seed.

<!-- BEGIN GENERATED: ablation-backbone -->
| Backbone + classificador | Acurácia (média ± dp, 1 seeds) | Δ vs. melhor | Macro-F1 | Treino |
|---|---|---|---|---|
| `resnet50_svm` | 0,8761 | +0,00 pp | 0,8767 | 1,0 min |
| `resnet50_mlp` | 0,8651 | -1,10 pp | 0,8653 | 0,4 min |
| `mobilenetv2_svm` | 0,8522 | -2,39 pp | 0,8524 | 0,6 min |
| `mobilenetv2_mlp` | 0,8392 | -3,69 pp | 0,8389 | 0,7 min |
| `inceptionv3_svm` | 0,7867 | -8,94 pp | 0,7867 | 1,6 min |
| `inceptionv3_mlp` | 0,7796 | -9,65 pp | 0,7793 | 0,6 min |
<!-- END GENERATED: ablation-backbone -->

*Ablação controlada: o classificador raso, os dados e o protocolo permanecem fixos; varia-se exclusivamente a rede extratora.*

**Resposta: sim, a troca impacta significativamente — mas não na direção que a intuição sugere.** A ResNet50 (0,8761) supera a MobileNetV2 (0,8522) em **+2,39 pontos percentuais**, e o teste exato de McNemar confirma a diferença (p = 4,1 × 10⁻¹⁰). Adotar a rede mais simples, portanto, **custa** acurácia — o que responde afirmativamente à questão do enunciado.

**Contudo, o resultado mais informativo é a InceptionV3.** Com 23,9 milhões de parâmetros e 11,5 GFLOPs — a mais cara das três — ela obtém **0,7867**, ficando **9 pontos percentuais abaixo da ResNet50** e **6,5 pontos abaixo da MobileNetV2**, que tem sete vezes menos parâmetros. **Capacidade não prediz qualidade de transferência.** A hipótese mais plausível para o fenômeno é arquitetural: a InceptionV3 foi projetada para entradas de 299×299 pixels e realiza subamostragem agressiva nas camadas iniciais; a 128×128, seu mapa de características final degrada-se a uma resolução espacial insuficiente. Registre-se que isso constitui uma **limitação da ablação**, e não uma propriedade intrínseca da rede: as três operam a 128 px por exigência do orçamento computacional, e essa resolução é nativa para a MobileNetV2, tolerável para a ResNet50 e adversa para a InceptionV3. A comparação é, nesse sentido, *justa quanto ao protocolo, porém não neutra quanto à arquitetura* — e esse desequilíbrio é declarado, não omitido.

**Limitação desta ablação: uma única *seed*.** As seis configurações foram executadas apenas na *seed* 42. O teste de McNemar sustenta que, **para aquelas predições**, a ResNet50 supera a MobileNetV2 de modo não atribuível ao acaso amostral do conjunto de teste — mas ele **não mede a estabilidade da conclusão frente à aleatoriedade do treinamento**. A estimativa de ruído de treinamento apresentada no protocolo (0,46 pp entre execuções idênticas) situa a margem aqui observada (2,39 pp) em cerca de cinco vezes essa variação, o que torna improvável — porém **não medida** — a inversão do ordenamento. A ordenação entre ResNet50 e MobileNetV2 é reportada como **provável**, não como estabelecida.

**A SVM supera o MLP em todos os três backbones** (+2,4 pp, +1,3 pp e +0,7 pp), sem exceção. Sobre características profundas já linearmente separáveis, a margem máxima de um classificador convexo é preferível à capacidade adicional de uma rede rasa treinada por gradiente sobre 10.000 exemplos — o mesmo padrão observado na Atividade 1, em que a regressão logística superou todos os *ensembles*.

### 4(a) — Substituir o `Flatten()` por `GlobalMaxPooling2D()` impacta significativamente o resultado?

*Investigação conduzida. Três configurações, duas* seeds.

<!-- BEGIN GENERATED: ablation-head -->
| Cabeça (pooling) | Acurácia (média ± dp, 2 seeds) | Δ vs. melhor | Macro-F1 | Treino |
|---|---|---|---|---|
| `mobilenetv2_gap` | 0,8576 ± 0,0025 | +0,00 pp | 0,8575 | 3,3 min |
| `mobilenetv2_gmp` | 0,8558 ± 0,0020 | -0,18 pp | 0,8555 | 3,5 min |
| `mobilenetv2_flatten` | 0,8470 ± 0,0063 | -1,06 pp | 0,8465 | 4,0 min |
<!-- END GENERATED: ablation-head -->

**Resposta: sim — o `GlobalMaxPooling2D()` melhora sobre o `Flatten()`, em +1,19 pontos percentuais, com significância** (0,8558 contra 0,8470; p = 9,4 × 10⁻⁵).

Duas precisões são necessárias, contudo, para não superinterpretar o resultado. Primeira: **o melhor desempenho observado não é do `GlobalMaxPooling2D()`, e sim do `GlobalAveragePooling2D()`** (0,8576), incluído como terceiro braço da ablação. Segunda: os dois **empatam tecnicamente** entre si (Δ = +0,14 pp; p = 0,598), de modo que o experimento **não autoriza preferir um ao outro**.

A conclusão que a evidência sustenta é, portanto: **qualquer agrupamento global supera o achatamento; entre média e máximo, este protocolo não detecta diferença.** A substituição sugerida pelo enunciado é benéfica, mas o mérito pertence ao *agrupamento global* enquanto classe de operação — não à escolha do máximo em particular.

O mecanismo é o número de parâmetros. Sobre a MobileNetV2 em 128 px, o mapa de características final tem dimensão 4×4×1280; logo:

| Cabeça | Parâmetros treináveis | Acurácia (2 *seeds*) |
|---|---|---|
| `Flatten() → Dense(512)` | **12.002.122** | 0,8470 ± 0,0063 |
| `GlobalMaxPooling2D() → Dense(512)` | 2.171.722 | 0,8558 ± 0,0020 |
| `GlobalAveragePooling2D() → Dense(512)` | 2.171.722 | **0,8576 ± 0,0025** |

O `Flatten()` exige **5,5 vezes mais parâmetros** para entregar acurácia **inferior** — e com **dispersão 2,5 vezes maior** entre *seeds* (± 0,0063 contra ± 0,0025), sinal de instabilidade. O `Flatten()` preserva a posição espacial de cada ativação; contudo, na classificação de objetos, a posição constitui majoritariamente **variável de perturbação** — a identidade da classe independe da localização do objeto no enquadramento. Exigir que a rede induza tal invariância a partir de 10.000 exemplos consome capacidade que o agrupamento global fornece por construção arquitetural.

Cabe registrar que **o notebook da disciplina utiliza `Flatten()`** — e que a alternativa comentada no próprio código (`#model.add(GlobalMaxPooling2D())`) é, conforme esta ablação demonstra, a melhor escolha.

### 4(b) — Substituir o algoritmo otimizador (`Adam()`) pode melhorar o resultado?

*Investigação conduzida. Quatro otimizadores, duas* seeds *por braço.*

<!-- BEGIN GENERATED: ablation-optimizer -->
| Otimizador | Acurácia (média ± dp, 2 seeds) | Δ vs. melhor | Macro-F1 | Treino |
|---|---|---|---|---|
| `adamw` | 0,8660 ± 0,0013 | +0,00 pp | 0,8659 | 12,7 min |
| `adam` | 0,8652 ± 0,0034 | -0,09 pp | 0,8648 | 13,4 min |
| `rmsprop` | 0,8638 ± 0,0027 | -0,22 pp | 0,8637 | 12,7 min |
| `sgd` | 0,8349 ± 0,0380 | -3,12 pp | 0,8351 | 11,1 min |
<!-- END GENERATED: ablation-optimizer -->

**Resposta: não. Nenhum otimizador supera o Adam de modo estatisticamente significativo.** O AdamW obtém a maior média (0,8660 contra 0,8652 do Adam), mas a diferença de **0,08 ponto percentual** é inferior a um quinto do ruído de treinamento medido (0,46 pp), e o teste exato de McNemar não a distingue de zero (p = 0,087). O RMSprop fica 0,14 pp abaixo do Adam. **A configuração padrão do enunciado já é adequada**, e a substituição do otimizador não constitui via de melhoria neste delineamento.

**O resultado informativo desta ablação não está nas médias, e sim na dispersão.**

| Otimizador | Média (2 *seeds*) | Desvio-padrão | Razão vs. AdamW |
|---|---|---|---|
| AdamW | 0,8660 | **± 0,0013** | 1× |
| Adam | 0,8652 | ± 0,0034 | 2,6× |
| RMSprop | 0,8638 | ± 0,0027 | 2,1× |
| **SGD + Nesterov** | 0,8349 | **± 0,0380** | **29×** |

O SGD produziu **0,8617** em uma *seed* e **0,8080** na outra — **5,4 pontos percentuais de diferença entre execuções que variam apenas na inicialização dos pesos**. Sua dispersão é vinte e nove vezes a do AdamW. A resposta útil à questão não é, portanto, que o SGD seja pior *em média*, mas que ele é **inconfiável neste regime**: uma execução única do SGD tem probabilidade não desprezível de produzir um resultado catastroficamente inferior, e nenhum protocolo baseado em execução única seria capaz de detectá-lo.

**Uma limitação do delineamento, exposta por esta ablação.** A investigação da execução anômala (`sgd`, *seed* 7) revela que sua **melhor acurácia de validação ocorreu na fase 1** (época 10 de 27), isto é, **antes do descongelamento**. Todas as demais execuções desta ablação têm sua melhor época na fase 2 (Adam: 22 e 25; AdamW: 25 e 24; RMSprop: 21 e 24). Como a parada antecipada seleciona o melhor modelo **dentro de cada fase**, e não globalmente entre as duas, o protocolo reportou o modelo da fase 2 — inferior, neste caso, ao que a fase 1 havia produzido.

Trata-se de um pressuposto implícito do delineamento em duas fases: **o de que a fase 2 aprimora o resultado da fase 1**. O pressuposto é válido para os três otimizadores estáveis, e falha exatamente para o instável. A correção — seleção global de modelo entre as duas fases — é direta, e está registrada como trabalho futuro; sua ausência afeta apenas a execução anômala aqui identificada, e a afeta na direção conservadora (subestimando o SGD, não superestimando-o).

### 4(c) — Avaliar outras estratégias de aumento de dados pode melhorar o resultado?

*Investigação conduzida. Quatro políticas, duas* seeds *por braço.*

<!-- BEGIN GENERATED: ablation-policy -->
| Política de aumento de dados | Acurácia (média ± dp, 2 seeds) | Δ vs. melhor | Macro-F1 | Treino |
|---|---|---|---|---|
| `strong` | 0,8678 ± 0,0010 | +0,00 pp | 0,8672 | 20,7 min |
| `lecture` | 0,8676 ± 0,0014 | -0,02 pp | 0,8672 | 21,0 min |
| `flip_crop` | 0,8653 ± 0,0004 | -0,25 pp | 0,8649 | 11,5 min |
| `flip` | 0,8590 ± 0,0028 | -0,88 pp | 0,8585 | 6,3 min |
<!-- END GENERATED: ablation-policy -->

**Resposta: sim, a política importa — porém não na direção que este relatório havia previsto.**

| Política | Média (2 *seeds*) | Desvio-padrão | Treino |
|---|---|---|---|
| `strong` (flip + rotação + zoom + deslocamento + contraste) | **0,8678** | ± 0,0010 | 20,7 min |
| `lecture` (**a do notebook da disciplina**) | 0,8676 | ± 0,0014 | 21,0 min |
| `flip_crop` (espelhamento + deslocamento 12,5%) | 0,8653 | ± 0,0004 | 11,5 min |
| `flip` (apenas espelhamento horizontal) | 0,8590 | ± 0,0028 | 6,3 min |

Os vereditos pareados (McNemar exato, *seed* 42) são: `lecture` supera `flip` em **+0,76 pp** com significância (p = 0,0037); `lecture` **empata tecnicamente** com `flip_crop` (p = 0,159) e com `strong` (p = 0,547).

**A hipótese registrada por este relatório antes da execução foi refutada.** Na seção de desvios metodológicos afirmou-se que a política da disciplina *"deixa desempenho na mesa"* por omitir o espelhamento horizontal — transformação reportada na literatura como das mais eficazes sobre o CIFAR-10. Os dados contradizem a expectativa em dois pontos:

1. **A política da disciplina figura entre as melhores**, em empate técnico com a política mais elaborada do delineamento (`strong`), que aplica cinco transformações — inclusive o espelhamento — e custa o mesmo tempo.
2. **O espelhamento horizontal isolado é a pior política testada**, significativamente inferior à da disciplina.

A explicação mais plausível reside na **diferença de regime**. A eficácia do espelhamento horizontal é estabelecida na literatura para redes convolucionais **treinadas do zero**, sobre as 50.000 imagens completas, ao longo de centenas de épocas. O regime aqui é outro: *backbone* **pré-treinado na ImageNet**, 10.000 imagens, 27 épocas, maioria dos pesos congelada. Nesse contexto, o *backbone* já incorpora invariância a espelhamento — adquirida sobre milhões de imagens que contêm objetos em ambas as orientações —, e reensiná-la não acrescenta informação, apenas consome épocas. As transformações que a política da disciplina aplica (rotação, zoom, deslocamento) atacam variações **não** cobertas pelo pré-treinamento, e é isso que as torna produtivas.

**A conclusão metodológica supera, em interesse, a conclusão empírica: uma heurística validada em um regime não transfere automaticamente para outro.** É, aliás, a mesma lição que a Estratégia 3 já havia produzido — o ajuste fino, prática padrão em regimes de dados abundantes, não se justifica com 10.000 imagens. O valor de registrar a hipótese *antes* do experimento está precisamente aqui: uma previsão refutada pelos próprios dados é resultado; uma previsão jamais formulada não é coisa alguma.

**Custo.** Cabe registrar que `flip_crop` atinge 99,7% da acurácia da política vencedora consumindo **metade do tempo de treinamento** (11,5 contra 20,7 minutos), e empata tecnicamente com ela. Sob restrição computacional, é a escolha racional.


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
