"""Pattern Memory Layer for SSLE-1 (documentation 4.7).

Stores templates of successful sequences scored by frequency and quality.
A template is produced by abstracting concrete tokens into slot types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

# Coarse slot categories used to abstract a target into a template.
_NUMERALS = {str(n) for n in range(0, 1001)}
_VERB_HINTS = ("AR", "ER", "IR", "OU", "AM")
_STOPWORDS = {"DE", "DA", "DO", "DAS", "DOS", "EM", "NO", "NA", "NOS", "NAS",
              "PARA", "POR", "COM", "SEM", "SUA", "SEU", "SUAS", "SEUS",
              "O", "A", "OS", "AS", "E", "OU", "UM", "UMA", "QUE"}


@dataclass
class Pattern:
    template: str
    frequency: int = 0
    total_nll: float = 0.0

    @property
    def avg_nll(self) -> float:
        return self.total_nll / self.frequency if self.frequency else 0.0

    @property
    def success_score(self) -> float:
        # frequency / (1 + avg_nll) -- higher is better.
        return self.frequency / (1.0 + self.avg_nll)

    def to_dict(self) -> dict:
        return {"template": self.template, "frequency": self.frequency,
                "total_nll": round(self.total_nll, 5),
                "success_score": round(self.success_score, 5)}


def _slot(token: str) -> str:
    if token in _NUMERALS:
        return "{NUM}"
    if token in _STOPWORDS:
        return token  # keep structural words literal
    if len(token) > 3 and token.endswith(_VERB_HINTS):
        return "{VERBO}"
    return "{TEMA}"


def abstract_template(tokens: Sequence[str]) -> str:
    return " ".join(_slot(t) for t in tokens)


class PatternMemory:
    def __init__(self):
        self.patterns: Dict[str, Pattern] = {}

    def observe(self, tokens: Sequence[str], nll: float = 0.0) -> None:
        tmpl = abstract_template(tokens)
        if not tmpl:
            return
        pat = self.patterns.get(tmpl)
        if pat is None:
            pat = Pattern(template=tmpl)
            self.patterns[tmpl] = pat
        pat.frequency += 1
        pat.total_nll += nll

    def top(self, k: int = 20) -> List[Pattern]:
        return sorted(self.patterns.values(), key=lambda p: -p.success_score)[:k]

    def to_list(self, k: int | None = None) -> List[dict]:
        items = sorted(self.patterns.values(), key=lambda p: -p.success_score)
        if k is not None:
            items = items[:k]
        return [p.to_dict() for p in items]

    @classmethod
    def from_list(cls, data: List[dict]) -> "PatternMemory":
        mem = cls()
        for d in data:
            pat = Pattern(template=d["template"], frequency=int(d.get("frequency", 0)),
                          total_nll=float(d.get("total_nll", 0.0)))
            mem.patterns[pat.template] = pat
        return mem
