"""Parser for the .sds2 dataset format (documentation section 6)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List


@dataclass
class Sample:
    theme: str = ""
    context: List[str] = field(default_factory=list)
    target: str = ""
    weight: float = 1.0


def parse_sds2(text: str) -> List[Sample]:
    samples: List[Sample] = []
    cur: dict | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "[SAMPLE]":
            cur = {"theme": "", "context": [], "target": "", "weight": 1.0}
            continue
        if line == "[/SAMPLE]":
            if cur is not None:
                samples.append(Sample(theme=cur["theme"], context=cur["context"],
                                      target=cur["target"], weight=cur["weight"]))
            cur = None
            continue
        if cur is None or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip().lower()
        val = val.strip()
        if key == "theme":
            cur["theme"] = val
        elif key == "context":
            cur["context"] = [t for t in val.split("|") if t]
        elif key == "target":
            cur["target"] = val
        elif key == "weight":
            try:
                cur["weight"] = float(val)
            except ValueError:
                cur["weight"] = 1.0
    return samples


def load_sds2(path: str) -> List[Sample]:
    with open(path, "r", encoding="utf-8") as f:
        return parse_sds2(f.read())


def iter_texts(samples: List[Sample]) -> Iterator[str]:
    """Yield all text (target + theme + context) for vocab building."""
    for s in samples:
        yield s.target
        if s.theme:
            yield s.theme
        if s.context:
            yield " ".join(s.context)
