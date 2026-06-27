# SSLE-2 "Nexus" — Benchmark Nano vs Base

Rede neural própria (autograd em NumPy, sem PyTorch/transformer/LSTM/n-gram),
treinada no corpus REAL de títulos em PT (`data/raw_titles/`, 34.204 títulos
brutos → 32.043 usáveis após filtro: turismo, esporte, cultura, saúde/ANVISA,
geral/clickbait). CPU, 2 núcleos.

## Configuração

| Preset | vocab | dim | heads | reasoning steps | memory slots | params | épocas |
|---|---|---|---|---|---|---|---|
| Nano | 4.000 | 96 | 4 | 2 | 24 | 920.224 | 10 (melhor: 7) |
| Base | 8.000 | 192 | 6 | 3 | 56 | 3.776.192 | 8 (melhor: 5) |

Split: 30.441 treino / 1.602 validação (seed 0). Early-stopping pelo melhor
val loss (o Base começou a sobreajustar após a época 5).

## Resultados

| Métrica | Nano | Base |
|---|---|---|
| Parâmetros | 920k | **3.78M** |
| Tamanho em disco (`.nx`) | 8.56 MB | 35.08 MB |
| **Val bits-per-char (BPC) ↓** | 1.622 | **1.500** |
| Geração | **35.1 títulos/s** (422 tok/s) | 13.2 títulos/s (142 tok/s) |

### Por que bits-per-character e não perplexidade?

Nano e Base usam **vocabulários BPE diferentes** (4k vs 8k). A perplexidade por
token depende do tamanho do vocabulário, então **não é comparável** entre os dois
(um vocabulário menor tende a uma perplexidade-por-token menor sem significar um
modelo melhor). O **BPC** normaliza pela quantidade de caracteres e é justo entre
tokenizers distintos. Pelo BPC, o **Base modela a língua melhor que o Nano**
(1.500 < 1.622) — exatamente o "Base supera Nano" que não acontecia na SSLE-1
(onde os dois ficavam presos nos mesmos ~227 tokens de templates).

Curva de validação (perplexidade por token, dentro de cada modelo):

```
Nano: 78.6 → 49.5 → 40.7 → 37.3 → 34.9 → 33.9 → 33.9(*)        (* melhor, época 7)
Base: 116.1 → 63.3 → 50.9 → 46.8 → 46.1(*) → 49.6(sobreajuste) (* melhor, época 5)
```

## Geração (Base) — títulos REAIS condicionados a tema + palavras-chave

```
SAUDE  + anvisa vacina
  - anvisa recebe pedido de uso emergencial da vacina covaxin
  - anvisa aprova novo pedido emergencial de vacina covaxin
  - anvisa aprova vacina da pfizer
SAUDE  + medicamento
  - anvisa suspende medicamento falsificado sem registro
  - anvisa suspende lote de palmito da empresa
ESPORTE + brasil copa
  - brasil se despede da copa do mundo de basquete
  - brasil apresenta os atletas e a cidade do rio de janeiro
CULTURA + festival música
  - festival de brasília recebe visita do instituto brasil
  - festival da amazônia é destaque no rio de janeiro
TURISMO + praia nordeste
  - mtur disponibiliza imagens de santa catarina
GERAL  + economia
  - novo modelo de segurança para a economia do brasil
  - mercado imobiliário é eleito mais de vendas no país
GERAL  (sem keywords, estilo clickbait do corpus)
  - novo aplicativo facilita reserva de hospedagem em tempo real
  - saiba como lucrar com a queda do dólar em tempos de crise!
```

A amostragem usa temperatura + top-k + top-p, com **penalidade de repetição** e
**bloqueio de bigrama repetido** para evitar laços.

## Contraste com a SSLE-1 v2.1 (n-gram + dataset sintético)

A SSLE-1, mesmo no preset Base, só remontava templates ocos a partir de ~227
tokens efetivos:

```
COMO TREINAR SUA ESTRATEGIA
O METODO INACREDITAVEL PARA OTIMIZAR SUA ECONOMIA NO PC
5 FORMAS DE TREINAR EDICAO
```

A SSLE-2, treinada em texto real, gera títulos **novos e semânticos** (não
presentes no dataset), e o Base de fato supera o Nano.

## Limitações honestas

- Modelo pequeno em CPU: a qualidade é melhor onde o corpus é denso (saúde/ANVISA,
  esporte/copa, cultura/festival). Palavras-chave raras (ex.: um time específico)
  podem gerar ruído, porque há poucos exemplos no corpus.
- É um modelo de **títulos**: gera headlines fluentes e temáticas, não responde
  perguntas nem "raciocina" como um LLM grande. O reasoning loop melhora a
  coerência, mas não é AGI.
- Mais dados do domínio do usuário → salto direto de qualidade (basta acrescentar
  arquivos `.txt` em `data/raw_titles/` e re-treinar).

## Reproduzir

```bash
python train_nexus.py --preset nano --epochs 10 --out models/nexus_nano.nx
python train_nexus.py --preset base --epochs 8  --out models/nexus_base.nx
python benchmarks/benchmark_nexus.py
```
