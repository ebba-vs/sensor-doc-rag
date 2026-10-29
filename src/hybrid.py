"""Hybrid retrieval: dense (embeddings) + sparse (BM25), fused with Reciprocal Rank Fusion,
optionally re-ranked by Jev.

Why: embeddings capture meaning ("tissue oxygenation" ~ "StO2") but are weak on exact
tokens such as model names, K-numbers and acronyms ("FLIR A50", "K181623", "HbR").
BM25 is the opposite. Fusing both usually beats either alone.
"""
import math
import re
from collections import Counter

TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")  # keeps "spectrum-2", "k181623", "60601-1-2"


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class BM25:
    """Okapi BM25 (Robertson & Zaragoza). score(q, d) = sum over query terms t of
    idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))"""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = sum(self.lens) / max(len(self.lens), 1)
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for tf, dl in zip(self.tfs, self.lens):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out

    def top(self, query: str, k: int) -> list[int]:
        s = self.scores(query)
        return [i for i in sorted(range(len(s)), key=lambda i: -s[i])[:k] if s[i] > 0]


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(id) = sum 1 / (k + rank). Robust because it only
    uses ranks, so BM25 scores and cosine distances never need to be on the same scale."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])
