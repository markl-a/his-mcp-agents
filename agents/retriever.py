"""Minimal BM25 over markdown chunks (one chunk per line) with source citations. Swap for a vector store in production."""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, folder: Path, k1: float = 1.5, b: float = 0.75):
        self.chunks = []
        for f in sorted(folder.glob("*.md")):
            for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if line.strip() and not line.startswith("#"):
                    self.chunks.append({"source": f"{f.name}:L{i}", "text": line.strip(), "toks": tokenize(line)})
        self.k1, self.b = k1, b
        self.avgdl = sum(len(c["toks"]) for c in self.chunks) / max(len(self.chunks), 1)
        df = Counter(t for c in self.chunks for t in set(c["toks"]))
        n = len(self.chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, k: int = 3, min_score: float = 0.5) -> list[dict]:
        q = tokenize(query)
        scored = []
        for c in self.chunks:
            tf = Counter(c["toks"])
            s = sum(self.idf.get(t, 0) * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * len(c["toks"]) / self.avgdl)) for t in q)
            if s >= min_score:
                scored.append((s, c))
        scored.sort(key=lambda x: -x[0])
        return [{"source": c["source"], "text": c["text"], "score": round(s, 2)} for s, c in scored[:k]]
