# Aprendendo equilíbrios de leilão por autoplay: visão geral do projeto

Este documento explica o projeto do zero, para quem não acompanhou o dia a
dia: a pergunta de pesquisa, o método (ator + crítico), como cada peça é
treinada, e um resumo cronológico dos experimentos e resultados. Os
detalhes técnicos de cada experimento específico estão nos documentos
`docs/findings_*.md` (em inglês); este aqui é o mapa para se orientar entre
eles.

## 1. A pergunta de pesquisa

Leilões têm equilíbrios de Bayes-Nash conhecidos em alguns casos clássicos
(leilão de primeiro preço, leilão *all-pay*, variantes com aversão a
risco), derivados analiticamente por teoria dos jogos. A pergunta aqui é:

> **Dá para dois ou mais agentes "descobrirem" esse equilíbrio sozinhos,
> só jogando uns contra os outros repetidamente e ajustando sua estratégia
> por gradiente — sem que ninguém programe a fórmula do equilíbrio?**

Se sim, isso importa porque o mesmo método poderia, em princípio, ser usado
em leilões (ou outros jogos) para os quais **não existe** fórmula fechada
— e de fato, um dos casos estudados aqui (aversão a risco assimétrica) só
tem solução numérica, não fórmula fechada, então o método precisa
funcionar "às cegas" também.

A validação é sempre a mesma: nos casos em que a teoria dá uma resposta
fechada, comparamos a estratégia aprendida com a fórmula teórica (erro
quadrático médio, desvio máximo). Nos casos sem fórmula fechada, comparamos
contra uma solução numérica de referência (resolvida à parte, não pelo
próprio treino) ou observamos se o jogo "assenta" (payoffs param de mudar).

## 2. O método: ator e crítico

### 2.1 Setup do jogo

$N$ agentes, cada um com um valor privado $v_i \sim U[0,1]$ (não observa o
valor dos outros). Cada agente escolhe um lance $b_i(v_i)$. Regras de
pagamento estudadas:

- **Primeiro preço**: só quem vence paga, paga o próprio lance.
- **All-pay**: todo mundo paga o próprio lance, só o maior lance vence.
- **Contínuo entre os dois** (`payment_share` $\lambda\in[0,1]$): vencedor
  paga $b$, perdedores pagam $\lambda b$ ($\lambda=0$ é primeiro preço,
  $\lambda=1$ é all-pay).
- **Primeiro preço com aversão a risco heterogênea**: cada agente tem
  utilidade côncava $u_i(x)=x^{\rho_i}$ sobre o ganho se vencer
  ($\rho_i=1$ neutro ao risco; $\rho_i<1$ avesso, quanto menor mais avesso).

### 2.2 O ator (*actor*)

Cada agente é uma rede pequena, `fictitious_play.actor.Actor`, que mapeia
valor $\to$ lance:
$$b(v) = v \cdot \sigma(\text{raw}(v)),$$
ou seja, o lance é sempre uma **fração** do valor (nunca lança mais do que
vale, o que é estritamente dominante em primeiro preço e nunca compensa em
all-pay). $\text{raw}(v)$ é um viés mais blocos de unidades ocultas que
podem ser adicionados aos poucos (**crescimento function-preserving**: um
bloco novo começa valendo exatamente zero, então crescer a rede nunca muda
a função já aprendida — só abre espaço para refiná-la). Isso permite, por
exemplo, treinar primeiro um ator "burro" (sem camada oculta — um fator de
redução constante, exatamente certo para o caso simétrico simples) e
crescê-lo depois se o problema exigir uma forma mais curva.

O ator é treinado por **ascensão de gradiente** na utilidade esperada: ele
precisa saber, para cada lance que considera dar, qual a probabilidade de
vencer com aquele lance — e é exatamente isso que o crítico fornece.

### 2.3 O crítico: o papel central

$P(\text{vencer}\mid b)$ não é conhecida de antemão — depende de como os
**outros** agentes estão lançando, e isso muda a cada rodada de treino. O
crítico é o componente que **estima essa função a partir de amostras**
(lances recentes dos oponentes), para que o ator tenha um gradiente para
subir.

Cada agente guarda um **buffer de crença** (*belief buffer*): um conjunto
de tamanho fixo dos seus próprios lances recentes. A cada rodada,
`belief_decay` (ex.: 0.95) da amostra antiga é mantida e o resto é trocado
por lances novos — ou seja, memória decrescente do próprio histórico, não
só o último lance. Isso é o que impede o fenômeno clássico de
"leapfrogging": sem memória (`belief_decay=0`), dois agentes
best-respondendo um ao último lance exato do outro entram num ciclo que
nunca assenta.

A cada rodada, **um único agente** por vez:
1. Amostra o lance máximo dos outros $N-1$ agentes (via seus buffers de
   crença) — vencer significa bater todos eles.
2. Ajusta seu crítico a essa amostra (como, depende do tipo de crítico —
   seção 3).
3. Treina seu próprio ator por várias centenas/milhares de passos de
   gradiente contra esse crítico, até estabilizar.
4. Atualiza seu próprio buffer de crença com os novos lances.

Os agentes se revezam (round-robin). Convergência é medida por: (a) desvio
máximo absoluto contra a curva teórica, quando existe fórmula fechada; ou
(b) estabilidade do payoff esperado, quando não existe.

## 3. As famílias de crítico (o "quê" e o "porquê" de cada uma)

Esse é o ponto onde mais tempo foi investido, porque a escolha do crítico
é o que mais afeta estabilidade e velocidade do treino.

| Crítico | Como ajusta | Forma assumida | Rápido? | Quando usar |
|---|---|---|---|---|
| `SplineCritic` | interpola quantis empíricos com uma spline monótona | nenhuma (não-paramétrico) | sim, fechado | baseline seguro, mais variância |
| `BetaCritic` | MLE fechado de 2 parâmetros ($\alpha$, escala $c$) | $\mathrm{Beta}(\alpha,1)$ escalada: $F(b)=(b/c)^\alpha$ | sim, fechado | quando o equilíbrio é uma lei de potência (primeiro preço, all-pay, o contínuo $\lambda$) |
| `MonotonicNNCritic` | rede neural pequena, monótona por construção, reajustada por gradiente a cada rodada | qualquer forma monótona | **não** — milhares de passos de Adam por rodada | quando a forma não é lei de potência (aversão a risco) e se aceita o custo |
| `MixtureBetaCritic` | mistura de $K$ CDFs $\mathrm{Beta}(\alpha_k,1)$ (mesma escala), ajustada por EM fechado | mais flexível que 1 componente, menos que uma rede livre | sim, quase tão rápido quanto Beta (~50 iterações de EM, não milhares de passos de Adam) | tentativa mais recente de ter o melhor dos dois mundos |

**Por que existe mais de um?** Porque não existe crítico "certo"
universal — existe um trade-off entre viés (o crítico assume uma forma que
pode não ser a verdadeira) e variância/custo (um crítico sem forma
nenhuma precisa de mais dados/tempo para não ser ruidoso). O fio condutor
descoberto ao longo do projeto:

> **Sempre que o equilíbrio verdadeiro é de fato uma lei de potência
> $b^*(v)=c\,v^p$ — o que é exatamente o caso do primeiro preço, do
> all-pay, e de tudo no meio (`payment_share`) — o `BetaCritic` é exato,
> não aproximado, e o treino converge rápido e limpo sem precisar de mais
> nenhuma firula.** Isso foi confirmado até com N=5 agentes, sem buffer
> reforçado nem continuação alguma (ver seção 5). Toda a maquinaria extra
> (redes neurais de crítico, mistura de Betas, continuação homotópica,
> buffer de memória mais forte) existe **só** para os casos em que essa
> suposição de lei de potência não vale — o caso motivador é a aversão a
> risco heterogênea, cujo equilíbrio não tem essa forma (nem tem fórmula
> fechada: precisou ser resolvido numericamente por um sistema de EDOs, ver
> `docs/findings_risk_aversion.md`).

### Uma nota sobre a "calibração" do crítico de mistura

O `MixtureBetaCritic` tem uma propriedade bonita: uma $\mathrm{Beta}(\alpha,1)$
vira exatamente uma distribuição Exponencial($\alpha$) sob a transformação
$Y=-\log X$. Isso significa que ajustar uma mistura de Betas aos lances dos
oponentes é **matematicamente idêntico** a ajustar uma mistura de
exponenciais ao log-negativo desses lances — e isso tem uma atualização de
EM (*expectation-maximization*) totalmente fechada, sem precisar de
otimizador de gradiente algum. Por isso ele é ~100-150x mais rápido que o
crítico-rede-neural: ambos são "flexíveis", mas um se resolve em poucas
dezenas de contas fechadas por rodada, o outro precisa de milhares de
passos de gradiente por rodada.

## 4. Continuação homotópica: a estratégia para configurações difíceis

Treinar direto numa configuração "dura" (ex.: aversão a risco bem
assimétrica, ou all-pay puro com muitos agentes) do zero costuma ficar
preso longe do equilíbrio, ou demorar demais. A ideia da **continuação
homotópica** é: começar numa configuração fácil (que tem solução conhecida
e treina rápido) e **avançar aos poucos** para a configuração difícil,
deixando que cada passo comece já "aquecido" (pesos herdados) do passo
anterior, em vez de do zero. Três eixos de continuação usados:

- **Em $\lambda$ (payment_share)**: de primeiro preço ($\lambda=0$) até
  all-pay ($\lambda=1$), em passos pequenos.
- **Em $\rho$ (aversão a risco)**: de neutro ao risco ($\rho=1$) até o
  alvo assimétrico desejado, em passos pequenos.
- **Em $N$ (número de agentes)**: treina com $N$ agentes até convergir,
  depois clona um agente existente para criar um novo (começa idêntico ao
  original, não do zero) e treina mais um pouco com $N+1$.

Cada "degrau" da escada precisa de **rounds suficientes** para reequilibrar
antes de avançar para o próximo — se o orçamento de rounds por estágio for
pequeno demais, o treino pode "pular" degraus sem realmente assentar, e o
resultado final sai pior do que seria sem homotopia nenhuma (isso
realmente aconteceu nesta sessão: ver seção 5).

(Uma tentativa anterior e mais elaborada de estabilizar o caso all-pay —
um termo de regularização "informado pela física" que penaliza a violação
da condição de primeira ordem do leilão, usando a derivada do crítico ou
invertendo diretamente a rede do oponente — está documentada em
`docs/findings_pinn_regularizer.md`. Foi largamente superada pela descoberta
mais simples do `BetaCritic`: o problema que a regularização tentava
consertar por fora era, em grande parte, viés do próprio crítico, resolvido
escolhendo um crítico com a forma certa.)

## 5. Linha do tempo dos experimentos e principais resultados

1. **Primeiro preço, `SplineCritic`**: funciona bem, baseline estabelecido.

2. **All-pay revela um problema estrutural**: ao contrário do primeiro
   preço, a condição de ótimo do all-pay depende **só da derivada** do
   crítico, não do seu nível — então qualquer ruído na estimativa de
   densidade (mesmo pequeno na CDF) quebra a convergência. Resultado: um
   ciclo quase-periódico que nunca assenta, mesmo com buffer de memória
   forte (`docs/findings_all_pay.md`).

3. **Descoberta do `BetaCritic`**: percebendo que o equilíbrio do all-pay
   (e do primeiro preço, e de tudo no meio) é uma lei de potência exata,
   construiu-se um crítico de 2 parâmetros que é **exato** nessa forma.
   Resultado: all-pay para de ciclar e converge limpo — testado e
   confirmado em N=2 (convergência em 55 rounds) **e em N=5** (convergência
   estável, payoffs batendo com o teórico, **sem** nenhum buffer reforçado,
   regularização ou continuação) (`docs/findings_beta_critic.md`).

4. **Aversão a risco expõe o limite do `BetaCritic`**: aqui o equilíbrio
   não é lei de potência (e nem tem fórmula fechada — precisou de um
   sistema de EDOs resolvido numericamente). `BetaCritic` fica
   estruturalmente enviesado (plateau a ~9% de erro), `SplineCritic`
   converge corretamente mas devagar (`docs/findings_risk_aversion.md`).

5. **`MonotonicNNCritic`**: tentativa de ter a flexibilidade de forma do
   Spline com menos ruído, usando uma rede pequena monótona por construção.
   Funciona, mas cada rodada precisa refazer um ajuste por gradiente
   (lento — ordem de minutos por rodada) e ainda mostra oscilação residual,
   motivando acrescentar memória mais forte (`belief_decay` maior) e
   continuação homotópica.

6. **`MixtureBetaCritic` (mais recente)**: mesma ideia do Beta, mas com
   $K=2$ componentes, ajuste fechado por EM. Validado isoladamente (recupera
   a CDF teórica quase exatamente) e em autoplay direto (qualidade
   comparável ao NN, ~100-150x mais rápido). Isso tornou viável rodar o
   cronograma de continuação completo em minutos em vez de dias.

7. **Primeira tentativa de homotopia com a mistura, usando os orçamentos de
   rounds herdados da era do NN-crítico** (poucos rounds por estágio, já
   que antes cada rodada era cara): terminou rápido, mas **a qualidade
   piorou** — erro subindo a cada estágio na aversão a risco, e colapso
   assimétrico grave ao crescer o all-pay para N=4 (agentes convergindo
   para curvas de lance bem diferentes entre si, longe da teórica).

8. **Refinando a granularidade** (passos de $\lambda$ menores, muito mais
   rounds por estágio — agora barato, já que a mistura é rápida): o
   problema qualitativo (colapso assimétrico) **foi corrigido** — os 4
   agentes convergem para curvas simétricas e suaves — mas ainda sobra um
   resíduo quantitativo de ~25-30% contra a curva teórica no topo, não
   fechado nem com 400 rounds por estágio de crescimento. (A aversão a
   risco ainda não foi re-testada com orçamento maior — item em aberto.)

## 6. Onde estamos agora

- `MixtureBetaCritic` é a aposta mais promissora atual para equilíbrios
  que não são lei de potência, pela combinação de velocidade e qualidade.
- O tema central do projeto, reconfirmado continuamente: **toda a
  maquinaria extra (buffers reforçados, homotopia, críticos flexíveis)
  existe para compensar viés de modelo em casos não-lei-de-potência, não
  para consertar uma instabilidade inerente ao autoplay** — nos casos onde
  o `BetaCritic` simples é exato, ele sozinho já basta e já funciona bem.
- Em aberto: fechar o resíduo do all-pay em N=4 (mais rounds? mais
  componentes na mistura?), e repetir o refinamento de orçamento na
  aversão a risco.

Para os números exatos, comandos usados e tabelas de cada comparação
citada acima, ver os respectivos `docs/findings_*.md`.
