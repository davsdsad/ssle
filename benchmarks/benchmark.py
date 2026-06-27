"""Benchmark comparing SSLE-1 model presets (e.g. Nano vs Base).

Trains (or loads) each preset on the same dataset, then measures:
    * training time and final loss
    * model file size
    * perplexity on the dataset
    * lexical diversity (distinct-1/2) and repetition rate of generations
    * average semantic coherence of generations
    * generation throughput (tokens/sec)

Outputs a JSON report and a Markdown table under benchmarks/.

Usage:
    python benchmarks/benchmark.py --dataset data/dataset.sds2 \
        --epochs 8 --presets nano base
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from eval import evaluate  # noqa: E402

from core.dataset import iter_texts, load_sds2  # noqa: E402
from core.engine import EngineConfig, SSLEEngine  # noqa: E402
from core.trainer import Trainer  # noqa: E402

PRESETS: Dict[str, Dict[str, int]] = {
    "nano":   {"vocab_size": 5000,  "embedding_dim": 64,  "n_order": 2},
    "mini":   {"vocab_size": 10000, "embedding_dim": 128, "n_order": 3},
    "base":   {"vocab_size": 20000, "embedding_dim": 256, "n_order": 3},
    "medium": {"vocab_size": 50000, "embedding_dim": 256, "n_order": 4},
}

GEN_THEMES = ["FORTNITE", "ACADEMIA", "CULINARIA", "FINANCAS", "PROGRAMACAO"]
SHOWCASE = [
    ("FORTNITE", "MIRA RANKED DICAS"),
    ("ACADEMIA", "TREINO HIPERTROFIA DIETA"),
    ("CULINARIA", "RECEITA RAPIDO FORNO"),
    ("FINANCAS", "INVESTIR RENDA DINHEIRO"),
]


def train_preset(name: str, samples, epochs: int, seed: int) -> Dict:
    p = PRESETS[name]
    config = EngineConfig(
        n_order=p["n_order"], embedding_dim=p["embedding_dim"],
        vocab_size=p["vocab_size"])
    engine = SSLEEngine(config, seed=seed)
    engine.tokenizer.count(iter_texts(samples))
    engine.tokenizer.build()
    engine.sync_embeddings(seed=seed)

    t0 = time.time()
    stats = Trainer(engine, seed=seed).train(samples, epochs=epochs,
                                              log=lambda m: None)
    train_time = time.time() - t0

    out_path = os.path.join("models", f"ssle_{name}.snm")
    engine.save(out_path)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)

    # Generation throughput.
    t0 = time.time()
    total_tokens = 0
    for theme in GEN_THEMES:
        for txt in engine.generate_many(theme=theme, count=4, max_tokens=40, seed=7):
            total_tokens += len(txt.split())
    gen_time = time.time() - t0
    tps = total_tokens / gen_time if gen_time else 0.0

    metrics = evaluate(engine, samples, GEN_THEMES, n_gen=5, seed=123)

    showcase = {}
    for theme, ctx in SHOWCASE:
        showcase[f"{theme} | {ctx}"] = engine.generate_many(
            theme=theme, context=ctx, count=3, seed=2025)

    return {
        "preset": name,
        "config": {"vocab_size": engine.tokenizer.vocab_size,
                   "embedding_dim": p["embedding_dim"], "n_order": p["n_order"]},
        "train_seconds": round(train_time, 2),
        "final_loss": stats["final_loss"],
        "loss_history": stats["loss_history"],
        "model_size_mb": round(size_mb, 3),
        "tokens_per_sec": round(tps, 1),
        "metrics": metrics,
        "showcase": showcase,
    }


def to_markdown(results: List[Dict], dataset_size: int, epochs: int) -> str:
    lines: List[str] = []
    lines.append("# SSLE-1 Benchmark — Nano vs Base\n")
    lines.append(f"- Dataset: **{dataset_size} amostras** (.sds2)")
    lines.append(f"- Épocas: **{epochs}**")
    lines.append("- Hardware: CPU (sem GPU)\n")

    headers = ["Métrica"] + [r["preset"].upper() for r in results]
    lines.append("## Tabela comparativa\n")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    def row(label: str, fn):
        lines.append("| " + " | ".join([label] + [str(fn(r)) for r in results]) + " |")

    row("Vocab size", lambda r: r["config"]["vocab_size"])
    row("Embedding dim", lambda r: r["config"]["embedding_dim"])
    row("N-gram order", lambda r: r["config"]["n_order"])
    row("Tamanho do modelo (MB)", lambda r: r["model_size_mb"])
    row("Tempo de treino (s)", lambda r: r["train_seconds"])
    row("Loss final", lambda r: r["final_loss"])
    row("Perplexidade ↓", lambda r: r["metrics"]["perplexity"])
    row("Distinct-1 ↑", lambda r: r["metrics"]["distinct_1"])
    row("Distinct-2 ↑", lambda r: r["metrics"]["distinct_2"])
    row("Taxa de repetição ↓", lambda r: r["metrics"]["repetition_rate"])
    row("Coerência semântica ↑", lambda r: r["metrics"]["avg_coherence"])
    row("Geração (tokens/s) ↑", lambda r: r["tokens_per_sec"])

    lines.append("\n## Exemplos de geração\n")
    for r in results:
        lines.append(f"### {r['preset'].upper()}\n")
        for key, gens in r["showcase"].items():
            lines.append(f"**{key}**")
            for g in gens:
                lines.append(f"- {g}")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark SSLE-1 presets")
    ap.add_argument("--dataset", default="data/dataset.sds2")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--presets", nargs="*", default=["nano", "base"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    samples = load_sds2(args.dataset)
    print(f"Dataset: {len(samples)} samples")

    results = []
    for name in args.presets:
        print(f"\n=== Training preset: {name} ===")
        res = train_preset(name, samples, args.epochs, args.seed)
        print(f"  loss={res['final_loss']} size={res['model_size_mb']}MB "
              f"ppl={res['metrics']['perplexity']} "
              f"coherence={res['metrics']['avg_coherence']} "
              f"tps={res['tokens_per_sec']}")
        results.append(res)

    os.makedirs("benchmarks", exist_ok=True)
    report = {"dataset_size": len(samples), "epochs": args.epochs,
              "results": results}
    with open("benchmarks/results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    md = to_markdown(results, len(samples), args.epochs)
    with open("benchmarks/RESULTS.md", "w", encoding="utf-8") as f:
        f.write(md)
    print("\nWrote benchmarks/results.json and benchmarks/RESULTS.md")


if __name__ == "__main__":
    main()
