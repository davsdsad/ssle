"""Train an SSLE-2 "Nexus" title model on the real Portuguese corpus.

Usage:
    python train_nexus.py --preset nano --epochs 8 --out models/nexus_nano.nx
    python train_nexus.py --preset base --epochs 6 --out models/nexus_base.nx
"""

from __future__ import annotations

import argparse
import random
import time

from nn.module import parameter_count
from ssle2.data import build_sample, load_corpus, make_batches, theme_list
from ssle2.generate import generate
from ssle2.model import NexusConfig, NexusLM
from ssle2.serialize import save_model
from ssle2.tokenizer import BPETokenizer
from ssle2.trainer import evaluate, train

PRESETS = {
    "nano": dict(vocab_size=4000, dim=96, heads=4, reasoning_steps=2,
                 memory_slots=24, ffn_mult=3, max_len=40,
                 batch_size=48, lr=2.5e-3),
    "base": dict(vocab_size=8000, dim=192, heads=6, reasoning_steps=3,
                 memory_slots=56, ffn_mult=4, max_len=40,
                 batch_size=48, lr=2e-3),
}

PREVIEW = [
    ("ESPORTE", ["brasil", "copa"]),
    ("SAUDE", ["anvisa"]),
    ("TURISMO", ["praia"]),
    ("GERAL", None),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="nano")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--data", default="data/raw_titles")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg_kw = dict(PRESETS[args.preset])
    batch_size = args.batch or cfg_kw.pop("batch_size")
    cfg_kw.pop("batch_size", None)
    lr = cfg_kw.pop("lr")
    out = args.out or f"models/nexus_{args.preset}.nx"
    rng = random.Random(args.seed)

    pairs = load_corpus(args.data)
    rng.shuffle(pairs)
    themes = theme_list(pairs)
    n = len(pairs)
    n_val = max(200, n // 20)
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    print(f"corpus: {n} titles | train {len(train_pairs)} | val {len(val_pairs)} "
          f"| themes {themes}", flush=True)

    tk = BPETokenizer()
    t0 = time.time()
    tk.train([t for _, t in train_pairs], vocab_size=cfg_kw["vocab_size"],
             min_freq=3, themes=themes)
    print(f"tokenizer: vocab {tk.size} ({time.time()-t0:.1f}s)", flush=True)
    cfg_kw["vocab_size"] = tk.size

    train_samples = [build_sample(tk, th, ti, rng, cfg_kw["max_len"])
                     for th, ti in train_pairs]
    val_samples = [build_sample(tk, th, ti, rng, cfg_kw["max_len"])
                   for th, ti in val_pairs]
    val_batches = list(make_batches(val_samples, batch_size, cfg_kw["max_len"],
                                    rng, shuffle=False))

    cfg = NexusConfig(**cfg_kw)
    model = NexusLM(cfg, seed=args.seed)
    print(f"model '{args.preset}': {parameter_count(model):,} params", flush=True)

    best = float("inf")

    def on_epoch(epoch: int, train_loss: float) -> None:
        nonlocal best
        val_loss = evaluate(model, val_batches)
        import math
        ppl = math.exp(min(val_loss, 20))
        print(f"  >> val_loss {val_loss:.4f} | val_ppl {ppl:.2f}", flush=True)
        for theme, kws in PREVIEW:
            txt = generate(model, tk, theme, kws, temperature=0.85,
                           top_k=40, top_p=0.92, seed=epoch * 7 + 1)
            tag = f"{theme}/{','.join(kws) if kws else '-'}"
            print(f"     [{tag}] {txt}", flush=True)
        if val_loss < best:
            best = val_loss
            save_model(out, model, tk)
            print(f"     saved {out} (best)", flush=True)

    def batches_fn():
        return make_batches(train_samples, batch_size, cfg_kw["max_len"], rng)

    train(model, batches_fn, epochs=args.epochs, lr=lr, log_every=200,
          on_epoch=on_epoch)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
