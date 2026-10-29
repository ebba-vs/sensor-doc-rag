"""Side-by-side comparison of two evaluation runs, per question and per metric.

  python src/compare_runs.py eval-prompt-production eval-prompt-candidate

Prints a summary (mean per metric, A vs B) and the questions where a metric
changed, then writes the full table to reports/compare_<A>_vs_<B>.md.
"""
import sys
from pathlib import Path

import mlflow

from config import settings


def _latest_run_id(exp_id: str, run_name: str) -> str:
    runs = mlflow.search_runs(
        experiment_ids=[exp_id],
        filter_string=f"tags.mlflow.runName = '{run_name}' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise SystemExit(f"No finished run named '{run_name}'.")
    return runs.iloc[0].run_id


def _scores(exp_id: str, run_id: str) -> dict[str, dict]:
    """{question: {metric: value, '_answer': str}} for one run."""
    df = mlflow.search_traces(locations=[exp_id], run_id=run_id)
    out = {}
    for _, row in df.iterrows():
        req = row["request"] or {}
        q = req.get("question") if isinstance(req, dict) else None
        if not q:
            continue  # e.g. judge/Jev call traces logged in the same run - not a RAG answer
        vals = {"_answer": str(row["response"])[:160]}
        for a in row["assessments"] or []:
            fb = a.get("feedback")
            if fb is not None and "value" in fb:
                vals[a["assessment_name"]] = fb["value"]
        out[q] = vals
    return out


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "✅" if v else "❌"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _mean(vals: list) -> float | None:
    nums = [float(v) for v in vals if isinstance(v, (bool, int, float))]
    return sum(nums) / len(nums) if nums else None


def main(name_a: str, name_b: str) -> None:
    mlflow.set_tracking_uri(settings.tracking_uri)
    exp = mlflow.get_experiment_by_name(settings.experiment)
    a = _scores(exp.experiment_id, _latest_run_id(exp.experiment_id, name_a))
    b = _scores(exp.experiment_id, _latest_run_id(exp.experiment_id, name_b))
    questions = sorted(set(a) | set(b))
    metrics = sorted({m for d in (a, b) for v in d.values() for m in v if not m.startswith("_")})

    lines = [f"# {name_a} (A) vs {name_b} (B)", "", "## Summary", "",
             "| Metric | A | B | B − A |", "|---|---|---|---|"]
    for m in metrics:
        ma = _mean([a.get(q, {}).get(m) for q in questions])
        mb = _mean([b.get(q, {}).get(m) for q in questions])
        delta = f"{mb - ma:+.2f}" if ma is not None and mb is not None else "-"
        lines.append(f"| {m} | {_fmt(ma)} | {_fmt(mb)} | {delta} |")

    lines += ["", "## Questions where a metric changed", ""]
    changed = 0
    for q in questions:
        diffs = [m for m in metrics if a.get(q, {}).get(m) != b.get(q, {}).get(m)]
        if not diffs:
            continue
        changed += 1
        lines.append(f"### {q}")
        lines += ["| Metric | A | B |", "|---|---|---|"]
        lines += [f"| {m} | {_fmt(a.get(q, {}).get(m))} | {_fmt(b.get(q, {}).get(m))} |" for m in diffs]
        lines += ["", f"- **A:** {a.get(q, {}).get('_answer', '-')}",
                  f"- **B:** {b.get(q, {}).get('_answer', '-')}", ""]
    if not changed:
        lines.append("_No differences._")

    report = "\n".join(lines)
    Path("reports").mkdir(exist_ok=True)
    out = Path(f"reports/compare_{name_a}_vs_{name_b}.md")
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
