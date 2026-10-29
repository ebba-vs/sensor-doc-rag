"""Pluggable embedding backends so the same pipeline runs locally, via API, or in CI."""
import hashlib
import math

from config import settings

_local_model = None


def _hash_embed(text: str, dim: int = 256) -> list[float]:
    """Deterministic bag-of-words hashing embedding. Only for tests/CI - not semantic."""
    vec = [0.0] * dim
    for tok in text.lower().split():
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def embed(texts: list[str]) -> list[list[float]]:
    global _local_model
    if settings.embeddings == "hash":
        return [_hash_embed(t) for t in texts]
    if settings.embeddings == "openai":
        from openai import OpenAI
        resp = OpenAI().embeddings.create(model=settings.openai_embedding_model, input=texts)
        return [d.embedding for d in resp.data]
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer(settings.local_embedding_model)
    return _local_model.encode(texts, normalize_embeddings=True).tolist()
