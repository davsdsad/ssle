"""Model serialization to a compressed .nx file (config + tokenizer + params)."""

from __future__ import annotations

import gzip
import json

from .model import NexusConfig, NexusLM
from .tokenizer import BPETokenizer

FORMAT_VERSION = "ssle2-nexus-1"


def save_model(path: str, model: NexusLM, tk: BPETokenizer) -> None:
    blob = {
        "format": FORMAT_VERSION,
        "config": model.cfg.to_dict(),
        "tokenizer": tk.to_dict(),
        "params": model.state_dict(),
    }
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(blob, f)


def load_model(path: str) -> tuple[NexusLM, BPETokenizer]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        blob = json.load(f)
    cfg = NexusConfig.from_dict(blob["config"])
    model = NexusLM(cfg, seed=0)
    model.load_state_dict(blob["params"])
    tk = BPETokenizer.from_dict(blob["tokenizer"])
    return model, tk
