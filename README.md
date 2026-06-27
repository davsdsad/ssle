# SSLE-1 v2.1 — Sumai Sequential Learning Engine

Modelo de linguagem **probabilístico, leve e 100% CPU** (sem redes neurais
profundas, sem GPU). Implementa a especificação SSLE-1 v2.1: uma cadeia de
Markov n-gram com embeddings aprendidos por SGD, acrescida de um **Motor de
Raciocínio Semântico** e um **Buffer de Contexto de Longo Alcance** que dão
coerência e direcionamento temático à geração.

> Em resumo: gera títulos/headlines temáticos (estilo conteúdo) treinando em
> segundos numa CPU comum, com modelos de poucos MB.

---

> ### Veja também: SSLE-2 "Nexus" — rede neural própria do zero
>
> Para geração com aprendizado de língua **real** (não n-gram, não templates),
> o repositório também traz a **SSLE-2 "Nexus"**: uma rede neural construída do
> zero em NumPy puro — **autograd próprio, atenção própria (Resonance Attention),
> memória de conceitos key-value e um loop de reasoning** — **sem PyTorch,
> transformer, LSTM/GRU ou n-gram**. Treinada num corpus REAL de títulos em PT.
> Veja [`ssle2/README.md`](ssle2/README.md) e o benchmark Nano vs Base em
> [`benchmarks/RESULTS_NEXUS.md`](benchmarks/RESULTS_NEXUS.md).

## Arquitetura

```
INPUT → TOKENIZER → EMBEDDING LAYER → CONTEXT ENCODER → SEMANTIC REASONING ENGINE
→ TRANSITION MATRIX (N-gram) → LONG-RANGE CONTEXT BUFFER → PATTERN MEMORY
→ SAMPLING ENGINE → OUTPUT
```

Componentes (em `core/`):

| Módulo | Arquivo | Função |
|---|---|---|
| Tokenizer | `tokenizer.py` | uppercase → remove acentos (NFD) → remove pontuação → split. Tokens especiais PAD/UNK/BOS/EOS/SEP. |
| Embeddings + Encoder | `encoder.py` | Init Xavier, update SGD; média ponderada de contexto + viés por similaridade de cosseno. |
| Transition Matrix | `matrix.py` | Contagens n-gram com **backoff** (ordem N→N-1→…→unigrama) + logits aprendidos. |
| Semantic Reasoning | `semantic.py` | Concept Graph (coocorrência + cosseno), inferência de intenção, coerência, memória temática. |
| Long-Range Buffer | `buffer.py` | 4 camadas: recente, janela deslizante, resumo comprimido (decay) e âncoras semânticas. |
| Theme Profiles | `theme.py` | Prior PMI por tema, aprendido do dataset — direciona o vocabulário para o tema pedido. |
| Pattern Memory | `memory.py` | Abstração de templates com `success_score = freq / (1 + avg_nll)`. |
| Sampling | `sampler.py` | softmax(temperature) → penalidade de repetição → top-k → top-p (nucleus). |
| Engine | `engine.py` | Unifica tudo; serialização `.snm` (JSON gzip). |
| Trainer | `trainer.py` | Loop de treino: contagens + grafo + atualização de gradiente token a token. |

## Instalação

```bash
pip install -r requirements.txt   # apenas numpy
```

## Uso

### 1. Gerar um dataset (.sds2)

```bash
python scripts/dataset_gen.py --count 6000 --output data/dataset.sds2
```

Formato `.sds2`:

```
[SAMPLE]
theme=FORTNITE
context=MIRA|CONSTRUCAO|RANKED|DICAS
target=COMO MELHORAR SUA MIRA NO FORTNITE EM 2025
weight=1.0
[/SAMPLE]
```

### 2. Treinar

```bash
# Nano (vocab 5k, dim 64, n=2)
python train.py --dataset data/dataset.sds2 --epochs 8 \
    --n-order 2 --dim 64 --vocab 5000 --output models/ssle_nano.snm

# Base (vocab 20k, dim 256, n=3)
python train.py --dataset data/dataset.sds2 --epochs 8 \
    --n-order 3 --dim 256 --vocab 20000 --output models/ssle_base.snm
```

### 3. Gerar texto

```bash
python generate.py --model models/ssle_base.snm \
    --theme FORTNITE --context "MIRA RANKED DICAS" \
    --temperature 0.8 --top-k 10 --top-p 0.9 --count 5
```

### 4. Avaliar e comparar (benchmark)

```bash
python scripts/eval.py --model models/ssle_base.snm --dataset data/dataset.sds2
python benchmarks/benchmark.py --dataset data/dataset.sds2 --epochs 8 --presets nano base
```

## Presets

| Preset | vocab | dim | n-gram |
|---|---|---|---|
| nano | 5.000 | 64 | 2 |
| mini | 10.000 | 128 | 3 |
| base | 20.000 | 256 | 3 |
| medium | 50.000 | 256 | 4 |
| large | 100.000 | 512 | 5 |

## Benchmark Nano vs Base

Resultados completos em [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md)
(dataset de 6.000 amostras, 8 épocas, CPU). Resumo:

| Métrica | Nano | Base |
|---|---|---|
| Tamanho do modelo | ~0.2 MB | ~0.6 MB |
| Loss final | ~1.98 | **~1.78** |
| Perplexidade ↓ | ~7.3 | **~5.7** |
| Geração | ~130 tok/s | ~130 tok/s |

O **Base** (n-gram de ordem 3 + embeddings maiores) tem perplexidade
significativamente menor e gera títulos mais coerentes e fiéis ao tema; o
**Nano** treina/serializa menor e é ideal para protótipos.

Exemplo (tema FORTNITE, base):

```
COMO TREINAR SUA ESTRATEGIA
O METODO INACREDITAVEL PARA OTIMIZAR SUA ECONOMIA NO PC
5 FORMAS DE TREINAR EDICAO
```

## Modelo `.snm`

Arquivo único **JSON comprimido com gzip** contendo: versão, config, vocabulário
do tokenizer, pesos dos embeddings, contagens/logits da matriz, grafo de
conceitos, perfis temáticos, padrões e estatísticas de treino.

## Testes

```bash
python -m pytest tests/ -q
ruff check .
```

## Observação sobre o vocabulário

O gerador de dataset usa templates composicionais, então o vocabulário efetivo
é compacto (centenas de tokens). A engine suporta vocabulários de até 100k
tokens (preset `large`) — basta treinar em um corpus real maior para explorar
toda a capacidade dos presets `base`/`medium`/`large`.
