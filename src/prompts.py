"""Register prompt versions in the MLflow Prompt Registry and manage aliases.

  python src/prompts.py register          # registers v1 (baseline) and v2 (stricter)
  python src/prompts.py register-v3       # registers v3 and points 'candidate' at it
  python src/prompts.py promote 2         # point alias 'production' at version 2
  python src/prompts.py rollback 1        # point 'production' back at version 1

The app always loads prompts:/<name>@production, so rollout/rollback is an alias
move - no code deploy. 'candidate' is used for A/B evaluation before promotion.
"""
import sys

import mlflow

from config import settings

SYSTEM_V1 = (
    "You answer questions about medical-device sensors and their documentation. "
    "Use the context below."
)

SYSTEM_V2 = (
    "You answer questions about medical-device sensors and their documentation.\n"
    "Rules:\n"
    "1. Use ONLY the numbered context passages. If they do not contain the answer, say "
    "'The provided documents do not cover this.'\n"
    "2. Cite every factual claim with its passage number, e.g. [2].\n"
    "3. Never give personal medical advice or a diagnosis.\n"
    "4. Be concise: at most 5 sentences."
)

SYSTEM_V3 = (
    "You answer questions about medical-device sensors and their documentation.\n"
    "Rules:\n"
    "1. Base every factual claim on the numbered context passages and cite it, e.g. [2].\n"
    "2. Answer every part of the question. Combine information from several passages when needed.\n"
    "3. Only if no passage is relevant, say 'The provided documents do not cover this.'\n"
    "4. If the question asks for personal medical advice or a diagnosis, say you cannot give it, "
    "recommend consulting a qualified healthcare professional, and add relevant general device "
    "information from the passages if there is any.\n"
    "5. Be concise: no filler, no repetition."
)

USER = "Context:\n{{context}}\n\nQuestion: {{question}}"


def register() -> None:
    v1 = mlflow.genai.register_prompt(
        name=settings.prompt_name,
        template=[{"role": "system", "content": SYSTEM_V1}, {"role": "user", "content": USER}],
        commit_message="v1 baseline",
        tags={"author": "ebba"},
    )
    v2 = mlflow.genai.register_prompt(
        name=settings.prompt_name,
        template=[{"role": "system", "content": SYSTEM_V2}, {"role": "user", "content": USER}],
        commit_message="v2: grounded-only, citations, no medical advice",
        tags={"author": "ebba"},
    )
    mlflow.genai.set_prompt_alias(settings.prompt_name, alias="production", version=v1.version)
    mlflow.genai.set_prompt_alias(settings.prompt_name, alias="candidate", version=v2.version)
    print(f"Registered v{v1.version} (production) and v{v2.version} (candidate).")


def register_v3() -> None:
    v3 = mlflow.genai.register_prompt(
        name=settings.prompt_name,
        template=[{"role": "system", "content": SYSTEM_V3}, {"role": "user", "content": USER}],
        commit_message="v3: answer all parts, refuse only when nothing relevant, clinician referral",
        tags={"author": "ebba"},
    )
    mlflow.genai.set_prompt_alias(settings.prompt_name, alias="candidate", version=v3.version)
    print(f"Registered v{v3.version} as 'candidate'.")


def set_production(version: int) -> None:
    mlflow.genai.set_prompt_alias(settings.prompt_name, alias="production", version=version)
    print(f"'production' -> v{version}")


if __name__ == "__main__":
    mlflow.set_tracking_uri(settings.tracking_uri)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "register"
    if cmd == "register":
        register()
    elif cmd == "register-v3":
        register_v3()
    elif cmd in {"promote", "rollback"}:
        set_production(int(sys.argv[2]))
    else:
        raise SystemExit(__doc__)
