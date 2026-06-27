"""Generate titles from a trained SSLE-2 "Nexus" model.

Usage:
    python gen_nexus.py --model models/nexus_base.nx --theme ESPORTE \
        --keywords brasil copa --n 5
"""

from __future__ import annotations

import argparse

from ssle2.generate import generate
from ssle2.serialize import load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/nexus_base.nx")
    ap.add_argument("--theme", default="GERAL")
    ap.add_argument("--keywords", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--top_p", type=float, default=0.92)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    model, tk = load_model(args.model)
    themes = sorted(tk.themes)
    theme = args.theme.upper()
    if theme not in tk.themes:
        print(f"theme '{theme}' unknown; available: {themes}")
        theme = themes[0] if themes else theme

    for i in range(args.n):
        seed = None if args.seed is None else args.seed + i
        txt = generate(model, tk, theme, args.keywords,
                       temperature=args.temperature, top_k=args.top_k,
                       top_p=args.top_p, seed=seed)
        print(f"{i+1:>2}. {txt}")


if __name__ == "__main__":
    main()
