"""Model quality evaluation for SSLE-1.

Computes intrinsic metrics on a dataset and over generated samples:
    * perplexity / avg NLL on held-out targets
    * distinct-1 / distinct-2 (lexical diversity of generations)
    * repetition rate
    * average semantic coherence of generations
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, List

import numpy as np

from core.dataset import Sample, load_sds2
from core.engine import SSLEEngine
from core.sampler import softmax


def perplexity(engine: SSLEEngine, samples: List[Sample]) -> float:
    vocab_size = engine.tokenizer.vocab_size
    total_nll = 0.0
    n = 0
    for s in samples:
        ids = engine.tokenizer.encode(s.target)
        for t in range(1, len(ids)):
            recent = ids[max(0, t - (engine.config.n_order - 1)):t]
            logits = engine.matrix.get_logits(recent, vocab_size)
            probs = softmax(logits, temperature=1.0)
            tgt = ids[t]
            p = float(probs[tgt]) if tgt < vocab_size else 1e-12
            total_nll += -math.log(max(p, 1e-12))
            n += 1
    avg = total_nll / max(n, 1)
    return math.exp(avg)


def distinct_n(texts: List[str], n: int) -> float:
    grams = set()
    total = 0
    for t in texts:
        toks = t.split()
        for i in range(len(toks) - n + 1):
            grams.add(tuple(toks[i:i + n]))
            total += 1
    return len(grams) / total if total else 0.0


def repetition_rate(texts: List[str]) -> float:
    rep = 0
    total = 0
    for t in texts:
        toks = t.split()
        total += max(len(toks) - 1, 0)
        for i in range(1, len(toks)):
            if toks[i] == toks[i - 1]:
                rep += 1
    return rep / total if total else 0.0


def avg_coherence(engine: SSLEEngine, texts: List[str]) -> float:
    g = engine.semantic.graph
    scores = []
    for t in texts:
        ids = [engine.tokenizer.token_id(w) for w in t.split()]
        ids = [i for i in ids if i not in (0, 1)]
        pair = [g.strength(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
        if pair:
            scores.append(float(np.mean(pair)))
    return float(np.mean(scores)) if scores else 0.0


def evaluate(engine: SSLEEngine, samples: List[Sample], gen_themes: List[str],
             n_gen: int = 5, seed: int = 123) -> Dict[str, float]:
    texts: List[str] = []
    for theme in gen_themes:
        texts.extend(engine.generate_many(theme=theme, count=n_gen, seed=seed))
    return {
        "perplexity": round(perplexity(engine, samples), 3),
        "distinct_1": round(distinct_n(texts, 1), 3),
        "distinct_2": round(distinct_n(texts, 2), 3),
        "repetition_rate": round(repetition_rate(texts), 4),
        "avg_coherence": round(avg_coherence(engine, texts), 4),
        "n_generations": len(texts),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate an SSLE-1 model")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--themes", nargs="*", default=["FORTNITE", "ACADEMIA", "CULINARIA"])
    args = ap.parse_args()

    engine = SSLEEngine.load(args.model)
    samples = load_sds2(args.dataset)
    metrics = evaluate(engine, samples, args.themes)
    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
