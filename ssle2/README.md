# SSLE-2 "Nexus" — rede neural própria para títulos (PT)

Um **modelo de linguagem neural feito do zero em NumPy** — sem PyTorch, sem
transformer, sem LSTM/GRU, sem n-gram. Autograd próprio (modo reverso),
otimizador Adam próprio e uma arquitetura própria de **atenção + raciocínio +
memória** treinada em um **corpus REAL** de títulos em português.

> Objetivo: "um LLM, mas só para títulos" — gera títulos novos, coerentes e
> condicionados a **tema + palavras-chave**, aprendidos de texto real (não de
> templates).

## Por que não é um transformer

A arquitetura toma emprestada a ideia de *atenção*, mas opera de forma
deliberadamente diferente:

| Componente | O que faz | Diferença vs. transformer |
|---|---|---|
| **Resonance Attention** | atenção causal multi-cabeça de conteúdo, **envolta numa porta GLU**: cada posição decide quanto do contexto atendido admitir. | não é a atenção "pura" do transformer — há um portão multiplicativo aprendido na entrada. |
| **Concept Memory** | banco de **slots key-value treináveis** que toda posição lê por atenção de conteúdo (conhecimento global, independente da posição). | memória externa fixa, separada da self-attention; não existe no transformer básico. |
| **Reasoning Loop (weight-tied)** | em vez de empilhar N camadas independentes, **o mesmo bloco é aplicado R vezes** como refinamento iterativo ("pensar") do estado latente. | profundidade = passos de raciocínio, **não** número de parâmetros. |

Tudo isso roda sobre o autograd escrito à mão em `nn/` (verificado por
gradient-checking — ver `tests/test_autograd.py`).

## Pacotes

```
nn/                 framework neural próprio
  autograd.py       Tensor + backward (modo reverso), softmax, cross-entropy
  module.py         Module/Parameter, Linear, Embedding, LayerNorm
  optim.py          Adam (com clipping e weight decay)
ssle2/
  tokenizer.py      BPE subword do zero (treino incremental, rápido)
  data.py           corpus real, rótulo de tema, amostras tema+keywords
  model.py          NexusLM (Resonance Attention + Concept Memory + Reasoning)
  trainer.py        teacher forcing + cross-entropy mascarada + Adam
  generate.py       amostragem autoregressiva (temperatura / top-k / top-p)
  serialize.py      modelo `.nx` (JSON gzip: config + tokenizer + pesos)
```

## Dados

Corpus real de títulos em PT em `data/raw_titles/` (notícias de órgãos públicos +
dataset de clickbait), rotulados por tema a partir da fonte:
`TURISMO`, `ESPORTE`, `CULTURA`, `SAUDE`, `GERAL`.

Cada amostra de treino é:

```
<bos> <thm:TEMA> [palavras-chave] <sep> [título...] <eos>
```

A perda só é aplicada do `<sep>` em diante (o modelo é pontuado por **gerar o
título dado tema + palavras-chave**). As palavras-chave são amostradas do próprio
título no treino (e em 40% das vezes nenhuma), então o modelo aprende a usar
palavras-chave quando fornecidas e também a gerar sem elas.

## Uso

```bash
# treinar
python train_nexus.py --preset nano --epochs 10 --out models/nexus_nano.nx
python train_nexus.py --preset base --epochs 8  --out models/nexus_base.nx

# gerar
python gen_nexus.py --model models/nexus_base.nx \
    --theme ESPORTE --keywords brasil copa --n 5

# benchmark Nano vs Base
python benchmarks/benchmark_nexus.py
```

### Presets

| Preset | vocab | dim | heads | reasoning | memory | params |
|---|---|---|---|---|---|---|
| nano | 4.000 | 96 | 4 | 2 | 24 | ~0.9M |
| base | 8.000 | 192 | 6 | 3 | 56 | ~3.8M |

## Métrica do benchmark

Como Nano e Base usam **vocabulários BPE diferentes**, a perplexidade por token
não é comparável entre eles. O benchmark usa **bits-per-character (BPC)** no
conjunto de validação — normalizado por caractere, é comparável entre
tokenizers. Também reporta parâmetros, tamanho em disco e velocidade de geração.

Resultados completos: [`benchmarks/RESULTS_NEXUS.md`](../benchmarks/RESULTS_NEXUS.md).
