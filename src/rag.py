"""RAG pipeline with MLflow Tracing: retrieval, prompt (from registry), generation.

  python src/rag.py "What does the device measure and how?"

Each call produces one trace: answer (CHAIN) -> retrieve (RETRIEVER) -> generate (LLM),
with prompt version, provider, token usage, latency and estimated cost as attributes.
"""
import re
import sys
import time

import chromadb
import mlflow
from mlflow.entities import Document, SpanType

from config import settings
from embeddings import embed

mlflow.set_tracking_uri(settings.tracking_uri)
mlflow.set_experiment(settings.experiment)
mlflow.openai.autolog()  # traces OpenAI-compatible calls (OpenAI and Ollama) incl. token usage

_collection = None


def collection():
    global _collection
    if _collection is None:
        _collection = chromadb.PersistentClient(path=settings.chroma_path).get_collection(settings.collection)
    return _collection


_bm25 = None
_corpus = None  # (ids, texts, metadatas) of every chunk, for BM25


def _load_corpus():
    global _bm25, _corpus
    if _corpus is None:
        from hybrid import BM25
        data = collection().get(include=["documents", "metadatas"])
        _corpus = (data["ids"], data["documents"], data["metadatas"])
        # BM25 scores the body only: the "[K-number | device]" header repeats generic words
        # ("Thermal", "Camera", "Imaging") in every chunk and would inflate whole documents.
        _bm25 = BM25([re.sub(r"^\[[^\]]*\]\s*", "", t) for t in _corpus[1]])
    return _corpus


def _doc(cid: str, text: str, m: dict, **extra) -> Document:
    return Document(page_content=text, metadata={"doc_uri": f"{m['source']}#page={m['page']}", **extra}, id=cid)


def _dense(query: str, k: int) -> list[Document]:
    res = collection().query(query_embeddings=embed([query]), n_results=k)
    return [_doc(cid, t, m, distance=float(d))
            for cid, t, m, d in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])]


def _sparse(query: str, k: int) -> list[Document]:
    ids, texts, metas = _load_corpus()
    return [_doc(ids[i], texts[i], metas[i]) for i in _bm25.top(query, k)]


@mlflow.trace(span_type=SpanType.RERANKER)
def _jev_rerank(query: str, docs: list[Document], k: int) -> list[Document]:
    """One Jev call: a Noul per candidate, sorted by p(relevant) - TypeSafe's re-ranking pattern."""
    from jev_judge import client
    from typesafe_sdk import Noul
    q = "Does the `passage` contain information that helps answer the `question`?"
    r = client().system_one(
        state={"question": query},
        questions={f"p{i}": Noul(instructions={"passage": d.page_content, "question": q}) for i, d in enumerate(docs)},
    )
    for i, d in enumerate(docs):
        d.metadata["relevance"] = round(r.answers[f"p{i}"].noul, 3)
    return sorted(docs, key=lambda d: -d.metadata["relevance"])[:k]


@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve(query: str, k: int = settings.top_k) -> list[Document]:
    if settings.retrieval == "hybrid":
        from hybrid import rrf
        dense, sparse = _dense(query, settings.candidates), _sparse(query, settings.candidates)
        by_id = {d.id: d for d in dense + sparse}
        fused = rrf([[d.id for d in dense], [d.id for d in sparse]])
        docs = [by_id[i] for i, _ in fused]
        for d, (_, score) in zip(docs, fused):
            d.metadata["rrf"] = round(score, 4)
    else:
        docs = _dense(query, settings.candidates if settings.rerank == "jev" else k)
    if settings.rerank == "jev":
        return _jev_rerank(query, docs[: settings.candidates], k)
    return docs[:k]


def _client_and_model(provider: str):
    from openai import OpenAI
    if provider == "ollama":
        return OpenAI(base_url=settings.ollama_base_url, api_key="ollama"), settings.ollama_model
    return OpenAI(), settings.llm_model


@mlflow.trace(span_type=SpanType.LLM, name="mock_llm")
def _mock_generate(messages: list[dict]) -> str:
    """Offline stand-in so the pipeline, tracing and eval wiring can be tested without keys."""
    return "Based on the documents, the answer is in passage [1]."


def generate(messages: list[dict], provider: str) -> tuple[str, int, int, str]:
    if provider == "mock":
        return _mock_generate(messages), 0, 0, "mock"
    client, model = _client_and_model(provider)
    resp = client.chat.completions.create(model=model, messages=messages, temperature=0)
    u = resp.usage
    return resp.choices[0].message.content, (u.prompt_tokens if u else 0), (u.completion_tokens if u else 0), model


def _cost(model: str, tin: int, tout: int) -> float:
    pin, pout = settings.prices.get(model, (0.0, 0.0))
    return (tin * pin + tout * pout) / 1_000_000


@mlflow.trace(span_type=SpanType.CHAIN)
def run(question: str, prompt_alias: str = "production", provider: str | None = None) -> dict:
    provider = provider or settings.llm_provider
    t0 = time.perf_counter()
    docs = retrieve(question)
    context = "\n\n".join(f"[{i + 1}] ({d.metadata['doc_uri']}) {d.page_content}" for i, d in enumerate(docs))

    prompt = mlflow.genai.load_prompt(f"prompts:/{settings.prompt_name}@{prompt_alias}")
    messages = prompt.format(question=question, context=context)

    text, tin, tout, model = generate(messages, provider)
    latency = time.perf_counter() - t0
    meta = {
        "provider": provider,
        "model": model,
        "prompt_version": prompt.version,
        "prompt_alias": prompt_alias,
        "tokens_in": tin,
        "tokens_out": tout,
        "latency_s": round(latency, 3),
        "cost_usd": _cost(model, tin, tout),
    }
    span = mlflow.get_current_active_span()
    if span:
        span.set_attributes(meta)
    mlflow.update_current_trace(tags={"prompt_version": str(prompt.version), "provider": provider})
    return {"answer": text, "sources": [d.metadata["doc_uri"] for d in docs], **meta}


def answer(question: str, prompt_alias: str = "production") -> str:
    """Plain-string entry point used as predict_fn in evaluation."""
    return run(question, prompt_alias)["answer"]


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What does the device measure?"
    out = run(q)
    print(out["answer"], "\n\nSources:", *out["sources"], sep="\n")
    print(f"\n[{out['model']} | prompt v{out['prompt_version']} | {out['latency_s']}s | ${out['cost_usd']:.5f}]")
    print("Trace:", mlflow.get_last_active_trace_id())
