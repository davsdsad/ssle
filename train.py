"""CLI training script for SSLE-1.

Example:
    python train.py --dataset data/dataset.sds2 --epochs 10 --lr 0.001 \
        --n-order 3 --dim 128 --output models/ssle_v2.snm
"""

from __future__ import annotations

import argparse
import os

from core.dataset import iter_texts, load_sds2
from core.engine import EngineConfig, SSLEEngine
from core.trainer import Trainer


def build_config(args) -> EngineConfig:
    return EngineConfig(
        n_order=args.n_order,
        embedding_dim=args.dim,
        vocab_size=args.vocab,
        lr=args.lr,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repeat_decay=args.repeat_decay,
        smoothing=args.smoothing,
        buffer_window=args.buffer_window,
        decay_factor=args.decay_factor,
        coherence_thr=args.coherence_thr,
        anchor_weight=args.anchor_weight,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Train an SSLE-1 model")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", default="models/ssle_v2.snm")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--n-order", dest="n_order", type=int, default=3)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--vocab", type=int, default=15000)
    ap.add_argument("--top-k", dest="top_k", type=int, default=10)
    ap.add_argument("--top-p", dest="top_p", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--repeat-decay", dest="repeat_decay", type=float, default=0.7)
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--buffer-window", dest="buffer_window", type=int, default=50)
    ap.add_argument("--decay-factor", dest="decay_factor", type=float, default=0.995)
    ap.add_argument("--coherence-thr", dest="coherence_thr", type=float, default=0.15)
    ap.add_argument("--anchor-weight", dest="anchor_weight", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Loading dataset: {args.dataset}")
    samples = load_sds2(args.dataset)
    print(f"  {len(samples)} samples")

    config = build_config(args)
    engine = SSLEEngine(config, seed=args.seed)

    print(f"Building vocabulary (max {args.vocab})...")
    engine.tokenizer.count(iter_texts(samples))
    engine.tokenizer.build()
    engine.sync_embeddings(seed=args.seed)
    print(f"  vocab_size = {engine.tokenizer.vocab_size}")

    trainer = Trainer(engine, lr=args.lr, seed=args.seed)
    stats = trainer.train(samples, epochs=args.epochs)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    engine.save(args.output)
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nSaved model to {args.output} ({size_mb:.2f} MB)")
    print(f"Final loss: {stats['final_loss']}  |  train time: {stats['train_seconds']}s")


if __name__ == "__main__":
    main()
