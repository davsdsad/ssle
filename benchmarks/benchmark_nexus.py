"""Benchmark SSLE-2 Nexus models (Nano vs Base) on the real PT corpus.

Metrics:
  - parameter count and on-disk size
  - validation bits-per-character (BPC) -- comparable across tokenizers, unlike
    per-token perplexity which depends on the vocabulary
  - generation throughput (titles/s, tokens/s)
  - qualitative samples per theme + keyword conditioning
"""

from __future__ import annotations

import math
import os
import random
import time
from typing import List, Tuple

import numpy as np

from ssle2.data import load_corpus
from ssle2.generate import generate
from ssle2.model import NexusLM
from ssle2.serialize import load_model
from ssle2.tokenizer import EOS_ID, BPETokenizer, normalize

VAL_FRAC = 20  # 1/20 held out (matches training split with seed=0)


def val_split(data_dir: str) -> List[Tuple[str, str]]:
    pairs = load_corpus(data_dir)
    rng = random.Random(0)
    rng.shuffle(pairs)
    n_val = max(200, len(pairs) // VAL_FRAC)
    return pairs[:n_val]


def bits_per_char(model: NexusLM, tk: BPETokenizer,
                  val: List[Tuple[str, str]], limit: int = 600) -> float:
    """Sum token NLL (nats) over title bodies, normalized by characters."""
    total_nll = 0.0
    total_chars = 0
    for _, title in val[:limit]:
        ids = tk.encode_text(title) + [EOS_ID]
        if len(ids) < 2:
            continue
        seq = [tk.themes.get("GERAL", 0)] + ids  # minimal context
        idx = np.array([seq[:-1]], dtype=np.int64)
        logits = model(idx).data[0]                      # (T, V)
        # log-softmax
        m = logits.max(axis=-1, keepdims=True)
        logsm = logits - m - np.log(np.exp(logits - m).sum(axis=-1, keepdims=True))
        tgt = np.array(seq[1:])
        total_nll += float(-logsm[np.arange(len(tgt)), tgt].sum())
        total_chars += len(normalize(title))
    return (total_nll / math.log(2)) / max(total_chars, 1)


def gen_speed(model: NexusLM, tk: BPETokenizer, themes: List[str],
              n: int = 20) -> Tuple[float, float]:
    rng = random.Random(1)
    t0 = time.time()
    tokens = 0
    for i in range(n):
        theme = rng.choice(themes)
        txt = generate(model, tk, theme, None, temperature=0.85, seed=i)
        tokens += len(tk.encode_text(txt))
    dt = time.time() - t0
    return n / dt, tokens / dt


def report(path: str, data_dir: str, val: List[Tuple[str, str]]) -> dict:
    model, tk = load_model(path)
    from nn.module import parameter_count
    params = parameter_count(model)
    size_mb = os.path.getsize(path) / 1e6
    bpc = bits_per_char(model, tk, val)
    tps, toks = gen_speed(model, tk, sorted(tk.themes))
    return dict(path=path, params=params, size_mb=size_mb, vocab=tk.size,
                bpc=bpc, titles_s=tps, tokens_s=toks, model=model, tk=tk)


SHOWCASE = [
    ("ESPORTE", ["brasil", "copa"]),
    ("ESPORTE", ["flamengo"]),
    ("SAUDE", ["anvisa", "vacina"]),
    ("SAUDE", ["medicamento"]),
    ("TURISMO", ["praia", "nordeste"]),
    ("CULTURA", ["festival", "música"]),
    ("GERAL", ["economia"]),
    ("GERAL", None),
]


def main() -> None:
    data_dir = "data/raw_titles"
    val = val_split(data_dir)
    print(f"validation titles: {len(val)}\n")

    rows = []
    for preset in ["nano", "base"]:
        p = f"models/nexus_{preset}.nx"
        if not os.path.exists(p):
            print(f"skip {p} (not found)")
            continue
        r = report(p, data_dir, val)
        rows.append((preset, r))
        print(f"[{preset}] params={r['params']:,} vocab={r['vocab']} "
              f"size={r['size_mb']:.2f}MB val_BPC={r['bpc']:.3f} "
              f"gen={r['titles_s']:.1f} titles/s ({r['tokens_s']:.0f} tok/s)")

    print("\n=== qualitative generation (Base) ===")
    base = next((r for n, r in rows if n == "base"), None)
    target = base or (rows[0][1] if rows else None)
    if target:
        model, tk = target["model"], target["tk"]
        for theme, kws in SHOWCASE:
            txt = generate(model, tk, theme, kws, temperature=0.8,
                           top_k=40, top_p=0.92, seed=7)
            tag = f"{theme}/{','.join(kws) if kws else '-'}"
            print(f"  [{tag:>22}] {txt}")


if __name__ == "__main__":
    main()
