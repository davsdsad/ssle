"""Semantic Reasoning Engine for SSLE-1 (documentation 4.4).

Sub-modules:
    * Concept Graph        - semantic strength between tokens
    * Intent Inference     - infers the intent category of a sequence
    * Semantic Coherence   - coherence score of a candidate vs context
    * Thematic Memory      - persistent semantic field of the whole generation
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Set

import numpy as np

from .encoder import Embeddings, cosine_similarity

# Intent categories (documentation 4.4.2). Each maps to a set of trigger
# tokens (already normalized) that hint at the intent in the context/target.
INTENT_KEYWORDS: Dict[str, List[str]] = {
    "TUTORIAL": ["COMO", "FAZER", "APRENDA", "PASSO", "TUTORIAL", "GUIA"],
    "LISTA": ["DICAS", "MELHORES", "FORMAS", "MANEIRAS", "RAZOES", "ERROS", "TOP"],
    "GUIA": ["GUIA", "COMPLETO", "DEFINITIVO", "TUDO", "SOBRE"],
    "COMPARATIVO": ["VS", "VERSUS", "OU", "MELHOR", "DIFERENCA", "COMPARACAO"],
    "CLICKBAIT": ["INACREDITAVEL", "CHOCANTE", "SEGREDO", "NINGUEM", "VOCE",
                  "NUNCA", "REVELADO", "INCRIVEL"],
}

NUMERAL_TOKENS = {str(n) for n in range(0, 101)}


class ConceptGraph:
    """Sparse semantic-strength graph between token ids.

    strength = 0.6 * cooccurrence + 0.4 * cosine_sim
    """

    def __init__(self):
        # cooccurrence[a][b] = raw co-occurrence count
        self.cooc: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.edges: Dict[int, Dict[int, float]] = defaultdict(dict)
        self.centrality: Dict[int, float] = {}

    def update_cooccurrence(self, ids: Sequence[int], window: int = 4) -> None:
        ids = list(ids)
        n = len(ids)
        for i in range(n):
            a = ids[i]
            lo = max(0, i - window)
            hi = min(n, i + window + 1)
            for j in range(lo, hi):
                if i == j:
                    continue
                b = ids[j]
                self.cooc[a][b] += 1.0

    def finalize(self, embeddings: Embeddings) -> None:
        """Compute normalized strengths and centrality after counting."""
        # Normalize cooccurrence per source node to [0, 1].
        for a, neigh in self.cooc.items():
            if not neigh:
                continue
            mx = max(neigh.values()) or 1.0
            va = embeddings.vec(a)
            for b, c in neigh.items():
                cooc_norm = c / mx
                cos = max(0.0, cosine_similarity(va, embeddings.vec(b)))
                strength = 0.6 * cooc_norm + 0.4 * cos
                if strength > 0.0:
                    self.edges[a][b] = float(strength)
        # Centrality = sum of outgoing strengths (normalized).
        raw = {a: sum(m.values()) for a, m in self.edges.items()}
        mx = max(raw.values()) if raw else 1.0
        mx = mx or 1.0
        self.centrality = {a: v / mx for a, v in raw.items()}

    def strength(self, a: int, b: int) -> float:
        return self.edges.get(a, {}).get(b, 0.0)

    def strength_to_set(self, candidate: int, anchors: Set[int]) -> float:
        if not anchors:
            return 0.0
        return float(np.mean([self.strength(candidate, a) for a in anchors]))

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        edges = {str(a): {str(b): round(s, 5) for b, s in m.items()}
                 for a, m in self.edges.items() if m}
        centrality = {str(a): round(v, 5) for a, v in self.centrality.items()}
        return {"edges": edges, "centrality": centrality}

    @classmethod
    def from_dict(cls, data: dict) -> "ConceptGraph":
        g = cls()
        for a, m in data.get("edges", {}).items():
            g.edges[int(a)] = {int(b): float(s) for b, s in m.items()}
        g.centrality = {int(a): float(v) for a, v in data.get("centrality", {}).items()}
        return g


class ThematicMemory:
    """Persistent semantic field of the whole generation (documentation 4.4.4)."""

    def __init__(self, concept_graph: ConceptGraph, max_active: int = 12):
        self.graph = concept_graph
        self.token_history: List[int] = []
        self.field_weights: Dict[int, float] = defaultdict(float)
        self.max_active = max_active

    def reset(self) -> None:
        self.token_history = []
        self.field_weights = defaultdict(float)

    def append(self, token: int) -> None:
        self.token_history.append(token)
        # Strengthen the semantic field around this token.
        self.field_weights[token] += 1.0
        for b, s in self.graph.edges.get(token, {}).items():
            self.field_weights[b] += s

    def active_fields(self) -> Set[int]:
        if not self.field_weights:
            return set()
        ordered = sorted(self.field_weights.items(), key=lambda kv: -kv[1])
        return {tok for tok, _w in ordered[: self.max_active]}

    def coherence(self, candidate: int) -> float:
        """avg semantic_strength(candidate, token) over the active field."""
        fields = self.active_fields()
        if not fields:
            return 0.0
        return float(np.mean([self.graph.strength(candidate, f) for f in fields]))


class IntentInference:
    """Infers the intent category and scores candidate tokens (4.4.2)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def infer(self, context_tokens: Sequence[str], theme: str = "") -> str:
        text = " ".join(context_tokens) + " " + theme
        scores: Dict[str, int] = {k: 0 for k in INTENT_KEYWORDS}
        words = set(text.upper().split())
        for intent, kws in INTENT_KEYWORDS.items():
            scores[intent] = sum(1 for kw in kws if kw in words)
        best = max(scores.items(), key=lambda kv: kv[1])
        if best[1] == 0:
            return "GUIA"  # neutral default
        return best[0]

    def score(self, intent: str, candidate_id: int) -> float:
        """Boost a candidate token id given the inferred intent."""
        tok = self.tokenizer.id_to_token.get(candidate_id, "")
        if not tok:
            return 0.0
        if intent == "LISTA" and tok in NUMERAL_TOKENS:
            return 1.0
        kws = INTENT_KEYWORDS.get(intent, [])
        if tok in kws:
            return 1.0
        return 0.0


class SemanticReasoningEngine:
    """Top-level engine combining the four sub-modules.

    Applies a reasoning boost on top of the n-gram logits during generation
    (documentation flow in 4.4 "COMO IMPLEMENTAR O RACIOCINIO NA PRATICA").
    """

    def __init__(self, tokenizer, embeddings: Embeddings,
                 coherence_threshold: float = 0.15,
                 cooc_window: int = 4):
        self.tokenizer = tokenizer
        self.embeddings = embeddings
        self.graph = ConceptGraph()
        self.intent = IntentInference(tokenizer)
        self.memory = ThematicMemory(self.graph)
        self.coherence_threshold = coherence_threshold
        self.cooc_window = cooc_window

    def update_graph(self, ids: Sequence[int]) -> None:
        self.graph.update_cooccurrence(ids, window=self.cooc_window)

    def finalize_graph(self) -> None:
        self.graph.finalize(self.embeddings)

    def apply(self, logits: np.ndarray, intent: str, anchors: Set[int]) -> np.ndarray:
        """Apply semantic reasoning over a dense logit vector.

        logits[c] += coherence*0.4 + intent_alignment*0.4 + concept_boost*0.2
        Candidates with coherence < threshold are penalized.
        """
        vocab_size = logits.shape[0]
        boost = np.zeros(vocab_size, dtype=np.float32)
        active = self.memory.active_fields()
        # Restrict heavy work to tokens that already have any logit signal or
        # are part of the active field / anchors (keeps generation fast).
        candidates = set(np.nonzero(logits > np.log(1e-3))[0].tolist())
        candidates |= active | anchors
        for c in candidates:
            if c >= vocab_size:
                continue
            coherence = self.memory.coherence(c)
            intent_alignment = self.intent.score(intent, c)
            concept_boost = self.graph.strength_to_set(c, anchors)
            boost[c] = coherence * 0.4 + intent_alignment * 0.4 + concept_boost * 0.2
            if coherence < self.coherence_threshold and active:
                boost[c] -= 0.5  # penalize incoherent candidates
        return logits + boost
