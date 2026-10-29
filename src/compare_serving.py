"""Serving-strategy comparison: same questions through each provider, measure
latency / tokens / cost, and write a decision report.

  python src/compare_serving.py --providers openai ollama

Output: reports/serving_report.md  (the 'cost, latency and capacity framework').
"""
import argparse
import json
import statistics
from pathlib import Path

from rag import run

# ASSUMPTIONS for the self-hosting break-even estimate - edit to your real numbers.
GPU_HOURLY_USD = 1.0          # e.g. a single-GPU cloud instance, on-demand
SELF_HOST_QPS_CAPACITY = 2.0  # sustained queries/second one instance can serve (measure it!)


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def bench(provider: str, questions: list[str]) -> dict:
    rows = [run(q, provider=provider) for q in questions]
    lat = [r["latency_s"] for r in rows]
    return {
        "provider": provider,
        "model": rows[0]["model"],
        "n": len(rows),
        "p50_s": round(statistics.median(lat), 3),
        "p95_s": round(pct(lat, 0.95), 3),
        "avg_tokens_in": round(statistics.mean(r["tokens_in"] for r in rows)),
        "avg_tokens_out": round(statistics.mean(r["tokens_out"] for r in rows)),
        "cost_per_1k_queries_usd": round(1000 * statistics.mean(r["cost_usd"] for r in rows), 3),
    }


def report(results: list[dict]) -> str:
    lines = [
        "# Serving strategy report",
        "",
        "| Provider | Model | n | p50 (s) | p95 (s) | avg tok in | avg tok out | $ / 1k queries |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['provider']} | {r['model']} | {r['n']} | {r['p50_s']} | {r['p95_s']} | "
                     f"{r['avg_tokens_in']} | {r['avg_tokens_out']} | {r['cost_per_1k_queries_usd']} |")
    monthly_gpu = GPU_HOURLY_USD * 24 * 30
    lines += [
        "",
        "## Self-hosting break-even (assumptions in compare_serving.py)",
        f"- One always-on GPU instance: ~${monthly_gpu:,.0f}/month at ${GPU_HOURLY_USD}/h.",
        f"- Capacity: ~{SELF_HOST_QPS_CAPACITY} QPS sustained -> "
        f"~{SELF_HOST_QPS_CAPACITY * 3600 * 24 * 30:,.0f} queries/month max.",
    ]
    api = [r for r in results if r["cost_per_1k_queries_usd"] > 0]
    if api:
        c = api[0]["cost_per_1k_queries_usd"]
        lines.append(f"- API ({api[0]['model']}) costs ${c}/1k queries -> break-even at "
                     f"~{monthly_gpu / c * 1000:,.0f} queries/month. Below that, the API is cheaper.")
    lines += [
        "",
        "## Decision guide",
        "- **Third-party API**: lowest ops burden, best quality per effort; check data-processing terms for health data.",
        "- **Managed external models (e.g. Databricks External Models / gateway)**: same models, central governance, rate limits and key management.",
        "- **Self-hosted open model**: data never leaves your infra and cost is flat - worth it at high, steady volume or strict data residency; you own GPU capacity, scaling and model updates.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", nargs="+", default=["openai", "ollama"])
    ap.add_argument("--data", default="eval/eval_set.json")
    a = ap.parse_args()
    qs = [row["inputs"]["question"] for row in json.loads(open(a.data, encoding="utf-8").read())]
    res = [bench(p, qs) for p in a.providers]
    Path("reports").mkdir(exist_ok=True)
    Path("reports/serving_report.md").write_text(report(res), encoding="utf-8")
    print(report(res))
