"""CLI generation script for SSLE-1.

Example:
    python generate.py --model models/ssle_v2.snm --theme FORTNITE \
        --context "MIRA RANKED DICAS" --top-k 10 --top-p 0.9 \
        --temperature 0.8 --count 5
"""

from __future__ import annotations

import argparse

from core.engine import SSLEEngine


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate text with an SSLE-1 model")
    ap.add_argument("--model", required=True)
    ap.add_argument("--theme", default="")
    ap.add_argument("--context", default="")
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--max-tokens", dest="max_tokens", type=int, default=40)
    ap.add_argument("--top-k", dest="top_k", type=int, default=None)
    ap.add_argument("--top-p", dest="top_p", type=float, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    engine = SSLEEngine.load(args.model)
    if args.top_k is not None:
        engine.config.top_k = args.top_k
    if args.top_p is not None:
        engine.config.top_p = args.top_p
    if args.temperature is not None:
        engine.config.temperature = args.temperature

    outputs = engine.generate_many(
        theme=args.theme, context=args.context, count=args.count,
        max_tokens=args.max_tokens, seed=args.seed)
    for i, text in enumerate(outputs, 1):
        print(f"{i:2d}. {text}")


if __name__ == "__main__":
    main()
