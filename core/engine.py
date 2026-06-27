"""Main SSLE-1 engine — unites all modules (documentation 4 & 7).

Handles construction, generation and serialization to the compact .snm
(gzip-compressed JSON) format.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Set

from .buffer import LongRangeContextBuffer
from .encoder import ContextEncoder, Embeddings
from .matrix import TransitionMatrix
from .memory import PatternMemory
from .sampler import Sampler
from .semantic import SemanticReasoningEngine
from .theme import ThemeProfiles
from .tokenizer import BOS_ID, EOS_ID, Tokenizer, normalize


@dataclass
class EngineConfig:
    n_order: int = 3
    embedding_dim: int = 128
    vocab_size: int = 15000
    lr: float = 0.001
    top_k: int = 10
    top_p: float = 0.9
    temperature: float = 0.8
    repeat_decay: float = 0.7
    smoothing: float = 0.1
    buffer_window: int = 50
    decay_factor: float = 0.995
    coherence_thr: float = 0.15
    anchor_weight: float = 0.5
    theme_strength: float = 0.6

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EngineConfig":
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in fields})


class SSLEEngine:
    def __init__(self, config: EngineConfig, tokenizer: Tokenizer | None = None,
                 seed: int = 42):
        self.config = config
        self.tokenizer = tokenizer or Tokenizer(max_vocab=config.vocab_size)
        vs = self.tokenizer.vocab_size or config.vocab_size
        self.embeddings = Embeddings(vs, config.embedding_dim, lr=config.lr, seed=seed)
        self.encoder = ContextEncoder(self.embeddings)
        self.matrix = TransitionMatrix(config.n_order, config.smoothing)
        self.semantic = SemanticReasoningEngine(
            self.tokenizer, self.embeddings, coherence_threshold=config.coherence_thr)
        self.memory = PatternMemory()
        self.theme_profiles = ThemeProfiles()
        self.training_stats: dict = {}

    # ------------------------------------------------------------------ #
    # After vocabulary is finalized, (re)allocate embeddings to match.
    # ------------------------------------------------------------------ #
    def sync_embeddings(self, seed: int = 42) -> None:
        vs = self.tokenizer.vocab_size
        self.config.vocab_size = vs
        self.embeddings = Embeddings(vs, self.config.embedding_dim,
                                     lr=self.config.lr, seed=seed)
        self.encoder = ContextEncoder(self.embeddings)
        self.semantic.embeddings = self.embeddings
        self.semantic.intent.tokenizer = self.tokenizer

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def _theme_anchors(self, theme: str, context: str) -> Set[int]:
        anchors: Set[int] = set()
        for tok in (normalize(theme).split() + normalize(context).split()):
            tid = self.tokenizer.token_id(tok)
            if tid not in (0, 1):
                anchors.add(tid)
        return anchors

    def generate(self, theme: str = "", context: str = "", max_tokens: int = 40,
                 seed: int | None = None, greedy: bool = False) -> str:
        cfg = self.config
        sampler = Sampler(cfg.top_k, cfg.top_p, cfg.temperature, cfg.repeat_decay,
                          seed=seed)
        buffer = LongRangeContextBuffer(
            self.embeddings, self.encoder, cfg.buffer_window,
            decay_factor=cfg.decay_factor, anchor_weight=cfg.anchor_weight)

        anchors = self._theme_anchors(theme, context)
        buffer.reset(anchors)
        self.semantic.memory.reset()

        intent = self.semantic.intent.infer(normalize(context).split(), theme)

        # Seed the context vector from theme + context tokens.
        ctx_ids = list(anchors)
        ctx_vec = self.encoder.encode(ctx_ids) if ctx_ids else None
        vocab_size = self.tokenizer.vocab_size
        # Data-driven theme prior (primary theme-steering signal).
        theme_prior = self.theme_profiles.prior(theme, vocab_size)

        generated: List[int] = [BOS_ID]
        seen: Dict[int, int] = {}

        for _ in range(max_tokens):
            recent = generated[-(cfg.n_order - 1):] if cfg.n_order > 1 else []
            logits = self.matrix.get_logits(recent, vocab_size)
            logits = self.semantic.apply(logits, intent, anchors)
            logits = logits + buffer.context_bias(self.semantic.graph)
            # Theme prior steers vocabulary toward the requested theme.
            logits = logits + cfg.theme_strength * theme_prior
            if ctx_vec is not None:
                logits = logits + self.encoder.context_bias(ctx_vec)

            nxt = sampler.sample(logits, seen, greedy=greedy)
            if nxt == EOS_ID:
                break
            if nxt in (BOS_ID,):
                continue
            generated.append(nxt)
            seen[nxt] = seen.get(nxt, 0) + 1
            buffer.append(nxt)
            self.semantic.memory.append(nxt)

        return self.tokenizer.decode(generated)

    def generate_many(self, theme: str = "", context: str = "", count: int = 5,
                      max_tokens: int = 40, seed: int | None = None) -> List[str]:
        out = []
        for i in range(count):
            s = None if seed is None else seed + i
            out.append(self.generate(theme, context, max_tokens, seed=s))
        return out

    # ------------------------------------------------------------------ #
    # Serialization (.snm = gzip JSON)
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "version": 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": self.config.to_dict(),
            "tokenizer": self.tokenizer.to_dict(),
            "embeddings": self.embeddings.to_dict(),
            "matrix": self.matrix.to_dict(),
            "concept_graph": self.semantic.graph.to_dict(),
            "theme_profiles": self.theme_profiles.to_dict(),
            "patterns": self.memory.to_list(k=200),
            "training_stats": self.training_stats,
        }

    def save(self, path: str) -> None:
        data = self.to_dict()
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        with gzip.open(path, "wb") as f:
            f.write(raw)

    @classmethod
    def load(cls, path: str) -> "SSLEEngine":
        from .semantic import ConceptGraph
        from .theme import ThemeProfiles

        with gzip.open(path, "rb") as f:
            data = json.loads(f.read().decode("utf-8"))
        config = EngineConfig.from_dict(data["config"])
        tokenizer = Tokenizer.from_dict(data["tokenizer"])
        engine = cls(config, tokenizer=tokenizer)
        engine.embeddings = Embeddings.from_dict(data["embeddings"], lr=config.lr)
        engine.encoder = ContextEncoder(engine.embeddings)
        engine.matrix = TransitionMatrix.from_dict(data["matrix"])
        engine.semantic.embeddings = engine.embeddings
        engine.semantic.graph = ConceptGraph.from_dict(data["concept_graph"])
        engine.semantic.memory.graph = engine.semantic.graph
        engine.memory = PatternMemory.from_list(data.get("patterns", []))
        engine.theme_profiles = ThemeProfiles.from_dict(data.get("theme_profiles", {}))
        engine.training_stats = data.get("training_stats", {})
        return engine
