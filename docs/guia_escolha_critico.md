# Guia de escolha do crítico (documento vivo, v1)

**Status: rascunho inicial**, baseado em duas famílias de leilão testadas
a fundo (primeiro-preço/all-pay/contínuo $\lambda$, e aversão a risco
heterogênea) com N variando até 5. Três testes já planejados (segundo
preço, valores comuns, aversão a risco com N=3 — ver seção final) devem
confirmar, refinar ou derrubar partes deste guia. Trate como hipótese de
trabalho, não como receita validada.

A pergunta que este documento tenta responder: **dado o desenho de um
leilão novo, qual "cara" de crítico (e quais outros parâmetros do
treino) tem mais chance de funcionar bem, antes de gastar tempo testando
várias?**

## Os três eixos que encontramos até agora

### Eixo 1 — o equilíbrio é uma lei de potência?

**Como verificar**: derive (ou solucione numericamente) o equilíbrio
simétrico teórico $b^*(v)$. Se ele tiver a forma $b^*(v)=c\,v^p$ para
algum $p$, a distribuição do lance é exatamente $\mathrm{Beta}(1/p,1)$ —
teste isso diretamente ajustando um `BetaCritic` a uma amostra grande
tirada *da distribuição teórica verdadeira* (não de autoplay) e medindo o
erro máximo de CDF. Perto de zero → é lei de potência. Um viés de alguns
pontos percentuais que não desaparece → não é.

**Decisão**: se é lei de potência, use `BetaCritic` e pare por aqui. Ele é
exato, rapidíssimo (ajuste fechado, sem iteração), e funciona como um
regularizador implícito contra o ruído transitório do autoplay (ver eixo
2). Confirmado em primeiro-preço, all-pay, e todo o contínuo entre eles,
inclusive com N=5 sem nenhuma outra firula (`docs/findings_beta_critic.md`).

Se não é lei de potência (o caso motivador: aversão a risco assimétrica,
`docs/findings_risk_aversion.md`), siga para o eixo 2 antes de escolher
*quanto* de flexibilidade acrescentar.

### Eixo 2 — a condição de ótimo depende do nível do crítico, ou só da derivada?

**Como verificar**: escreva o payoff esperado do agente em função do
próprio lance $b$ e de $P(\text{vencer}\mid b)$, derive a condição de
primeira ordem (FOC) em $b$, e veja se $P(\text{vencer}\mid b)$ aparece
nela diretamente (nível) ou só através de sua derivada $dP/db$.

- **Primeiro preço**: payoff $=(v-b)P(\text{vencer}\mid b)$ → a FOC tem o
  nível de $P$ e sua derivada. O nível "ancora" a solução mesmo se a
  derivada estiver ruidosa.
- **All-pay**: payoff $=vP(\text{vencer}\mid b)-b$ → a FOC
  (`docs/findings_all_pay.md`) depende **só** da derivada. Qualquer
  imperfeição na densidade estimada (ruído de amostragem, ou — o achado
  novo — flexibilidade demais no crítico perseguindo estrutura
  transitória) quebra a convergência com muito mais facilidade.

**Decisão**:
- Se o nível entra na FOC (primeiro-preço e variantes, inclusive aversão a
  risco — mesma estrutura, só troca a utilidade): relativamente seguro
  usar um crítico mais flexível quando precisar (mistura de Betas,
  Kumaraswamy, rede monotônica). Escolha por custo computacional, nessa
  ordem de preferência: `BetaCritic`/`MixtureBetaCritic`/`KumaraswamyCritic`
  (fechados ou quase-fechados) antes de `MonotonicNNCritic` (caro).
- Se só a derivada entra (all-pay e o que mais se parecer com isso):
  **desconfie de qualquer crítico com graus de liberdade extras**, mesmo
  que ele contenha a resposta certa como caso particular — é exatamente
  o que a Kumaraswamy mostrou (`docs/findings_kumaraswamy_critic.md`):
  a flexibilidade deixa o crítico perseguir a mistura não-estacionária de
  lances que o próprio autoplay gera antes de assentar, e isso **drena**
  o parâmetro extra para longe do valor correto, mesmo quando as métricas
  de convergência ainda parecem boas. Prefira o crítico **menos**
  flexível que ainda capture a forma aproximada, e considere mitigar o
  viés residual por outras vias (continuação — eixo 3 — ou os
  regularizadores baseados na FOC documentados em
  `docs/findings_pinn_regularizer.md`) antes de aumentar a flexibilidade
  do crítico.

### Eixo 3 — quantos agentes, e dá para usar continuação?

**Padrão observado**: quanto mais agentes, mais fácil o autoplay colapsar
numa configuração assimétrica estável-porém-errada (vários agentes
convergem juntos, um fica para trás) — visto tanto com `MixtureBetaCritic`
quanto com `KumaraswamyCritic` em all-pay N≥3-4. Continuação (crescer N
aos poucos a partir de uma solução já convergida, ou avançar um parâmetro
como $\lambda$/$\rho$ em degraus pequenos) ajuda porque evita que o
crítico veja dados muito fora do equilíbrio por muito tempo seguido — mas
só funciona se cada degrau tiver rounds suficientes para reequilibrar
antes do próximo (ver a regressão e correção documentadas em
`docs/findings_mixture_beta_critic.md`).

**Decisão**: para N≥3 (ou qualquer configuração "extrema" — $\rho$ bem
assimétrico, $\lambda=1$ puro), prefira atingir o alvo via continuação a
partir de uma configuração mais fácil e já convergida, em vez de treinar
direto do zero na configuração-alvo — mesmo que o treino direto pareça
promissor nas primeiras rodadas (foi exatamente esse falso sinal positivo
que vimos com Kumaraswamy em all-pay N=4).

## Síntese em forma de árvore

1. O equilíbrio é lei de potência (verificável por teste direto)?
   - **Sim** → `BetaCritic`. Fim.
   - **Não** → vá para 2.
2. A condição de ótimo depende só da derivada do crítico?
   - **Não** (nível entra) → escolha um crítico flexível pela ordem de
     custo computacional acima; risco de instabilidade é baixo.
   - **Sim** (só derivada) → prefira o crítico menos flexível possível;
     considere regularização por FOC e/ou continuação antes de aumentar
     a flexibilidade do crítico.
3. Independente de 1-2: se N≥3 ou o alvo é uma configuração extrema,
   prefira continuação (homotopia e/ou crescimento em N) a um treino
   direto do zero no alvo.

## E os outros parâmetros (`belief_decay` etc.)?

Aqui o guia é bem mais fraco — temos um contraexemplo recente que mostra
que a intuição óbvia ("mais memória sempre ajuda a estabilizar") **não**
é uma regra segura: subir `belief_decay` de 0.95 para 0.98 piorou o
resultado do Kumaraswamy em all-pay N=4, em vez de melhorar
(`docs/findings_kumaraswamy_critic.md`). O efeito de `belief_decay`
parece interagir com a escolha do crítico de um jeito que ainda não
entendemos bem — por enquanto, a recomendação prática é: ao ajustar
`belief_decay` (ou outros hiperparâmetros do ator/buffer), sempre comparar
sob as mesmas condições exatas com mais de um critério de crítico, em vez
de assumir que o ajuste que funcionou para um vai funcionar para outro.

## Pendências que devem revisar este guia

- **Segundo preço**: aqui a estratégia dominante é lançar o próprio valor,
  então a própria noção de "FOC do eixo 2" muda de figura — esse teste
  deve esclarecer se o framework de 3 eixos faz sentido para um leilão
  sem ótimo interior não-trivial, ou se precisa de um eixo novo.
- **Valores comuns**: nenhum dos eixos atuais fala de estrutura
  informacional (sinal vs. valor verdadeiro, maldição do vencedor) — é
  bem possível que isso exija um quarto eixo inteiramente novo.
- **Aversão a risco com N=3**: testa se o eixo 3 (mais agentes = mais
  risco de colapso assimétrico) generaliza para equilíbrios
  não-lei-de-potência, ou se é um fenômeno específico do all-pay.

Assim que esses três saírem, este guia deve ser revisado — tanto para
confirmar os eixos acima quanto para decidir se precisa de mais algum.
