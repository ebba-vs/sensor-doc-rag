"""Human-in-the-loop review: ask, read the answer, rate it -> logged on the trace.

  python src/feedback.py

Ratings appear as assessments on each trace in the MLflow UI and can be turned into
new eval examples (bad answers -> regression tests).
"""
import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType

from rag import run

REVIEWER = "ebba"

if __name__ == "__main__":
    while True:
        q = input("\nQuestion (blank to quit): ").strip()
        if not q:
            break
        out = run(q)
        trace_id = mlflow.get_last_active_trace_id()
        print("\n" + out["answer"])
        rating = input("Correct and grounded? [y/n]: ").strip().lower() == "y"
        note = input("Comment (optional): ").strip() or None
        mlflow.log_feedback(
            trace_id=trace_id,
            name="human_correct",
            value=rating,
            rationale=note,
            source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id=REVIEWER),
        )
        print(f"Logged on trace {trace_id}")
