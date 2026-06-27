# SSLE-2 "Nexus" — rede neural própria para títulos (PT)

Um **modelo de linguagem neural feito do zero em NumPy puro** — **sem PyTorch,
sem transformer, sem LSTM/GRU, sem n-gram**. Autograd próprio (modo reverso),
otimizador Adam próprio e uma arquitetura própria de **atenção + raciocínio +
memória**, treinada num **corpus REAL** de títulos em português.

> "Um LLM, mas só para títulos": gera títulos novos, coerentes e condicionados a
> **tema + palavras-chave**, aprendidos de texto real (não de templates).

## Arquitetura (não é transformer)

| Componente | O que faz |
|---|---|
| **Resonance Attention** | atenção causal multi-cabeça envolta numa **porta GLU** — cada posição decide quanto do contexto atendido admitir. |
| **Concept Memory** | banco de **slots key-value treináveis** lido por atenção de conteúdo (conhecimento global, separado da self-attention). |
| **Reasoning Loop (weight-tied)** | o **mesmo bloco aplicado R vezes** como refinamento iterativo do estado latente — profundidade = passos de raciocínio, não nº de parâmetros. |

Tudo roda sobre o autograd escrito à mão em `nn/` (verificado por
gradient-checking em `tests/test_autograd.py`).

## Estrutura

```
nn/                framework neural próprio (autograd, módulos, Adam)
ssle2/             tokenizer BPE, dados, modelo NexusLM, treino, geração, serialização
data/raw_titles/   corpus real de títulos em PT (um título por linha)
models/            checkpoints .nx treinados (nano, base)
benchmarks/        benchmark_nexus.py + RESULTS_NEXUS.md
train_nexus.py     CLI de treino
gen_nexus.py       CLI de geração
```

## Uso

```bash
# treinar
python train_nexus.py --preset nano --epochs 10 --out models/nexus_nano.nx
python train_nexus.py --preset base --epochs 8  --out models/nexus_base.nx

# gerar
python gen_nexus.py --model models/nexus_base.nx --theme ESPORTE --keywords brasil copa --n 5

# benchmark Nano vs Base
python benchmarks/benchmark_nexus.py
```

Para treinar com seus próprios dados, coloque arquivos `.txt` (um título por
linha) em `data/raw_titles/` e rode o treino novamente. Temas disponíveis:
`TURISMO`, `ESPORTE`, `CULTURA`, `SAUDE`, `GERAL`.

| Preset | vocab | dim | heads | reasoning | memory | params |
|---|---|---|---|---|---|---|
| nano | 4.000 | 96 | 4 | 2 | 24 | ~0.9M |
| base | 8.000 | 192 | 6 | 3 | 56 | ~3.8M |

Detalhes de arquitetura: [`ssle2/README.md`](ssle2/README.md) ·
Benchmark completo: [`benchmarks/RESULTS_NEXUS.md`](benchmarks/RESULTS_NEXUS.md).

## Requisitos

Python ≥ 3.10 e NumPy. `pip install -r requirements.txt`.
