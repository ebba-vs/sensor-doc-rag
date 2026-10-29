"""Central configuration, read from environment variables (see .env.example)."""
import os
from dataclasses import dataclass, field


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


@dataclass(frozen=True)
class Settings:
    tracking_uri: str = _env("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment: str = _env("MLFLOW_EXPERIMENT", "sensor-doc-rag")

    llm_provider: str = _env("LLM_PROVIDER", "openai")  # openai | ollama | mock
    llm_model: str = _env("LLM_MODEL", "gpt-4o-mini")
    ollama_base_url: str = _env("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    ollama_model: str = _env("OLLAMA_MODEL", "llama3.1:8b")

    embeddings: str = _env("EMBEDDINGS", "local")  # local | openai | hash
    local_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"

    chroma_path: str = _env("CHROMA_PATH", "data/chroma")
    collection: str = "sensor_docs"
    chunk_size: int = 800       # characters
    chunk_overlap: int = 50
    top_k: int = int(_env("TOP_K", "8"))
    retrieval: str = _env("RETRIEVAL", "dense")   # dense | hybrid
    rerank: str = _env("RERANK", "none")          # none | jev
    candidates: int = 20                          # per retriever, before fusion / re-ranking

    prompt_name: str = "sensor-doc-qa"
    judge_model: str = _env("JUDGE_MODEL", "openai:/gpt-4o-mini")

    # USD per 1M tokens (input, output). ASSUMPTION: update from the provider's
    # current price list before you quote numbers anywhere. Self-hosted = 0 marginal
    # token cost; its cost is GPU/instance time (see compare_serving.py).
    prices: dict = field(default_factory=lambda: {
        "gpt-4o-mini": (0.15, 0.60),
        "llama3.1:8b": (0.0, 0.0),
        "mock": (0.0, 0.0),
    })


settings = Settings()
