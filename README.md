# sensor-doc-rag

A small, production-minded **RAG system over FDA 510(k) summaries of medical-device sensors**
– including Neko Health's own Spectrum-2 (tissue oximetry, SFDI) and Derma-2 (thermal + colour
skin imaging) – built to practise the *operational* side of LLM systems: **tracing, evaluation,
prompt versioning, retrieval tuning and judge design**, with MLflow 3 GenAI and TypeSafe's Jev.

> Question → hybrid retrieval (embeddings + BM25, fused with RRF) → Jev re-ranking →
> versioned prompt from the MLflow Prompt Registry → Llama 3.1 8B (local, Ollama) → cited answer.
> Every request is one MLflow trace; every change is measured on a 21-question eval set.

**Headline result:** over six measured iterations, the share of expected facts stated in the
answers rose from **44 % to 67 %**, and the share of needed facts present in the retrieved
context from **65 % to ~80 %** – mostly by fixing *retrieval*, not the prompt.

---

## What it demonstrates

| LLM Ops concern | Where |
|---|---|
| **Observability** – one trace per request: retrieval (with scores), prompt version, re-ranking, tokens, latency, cost | `src/rag.py` (`@mlflow.trace`, `mlflow.openai.autolog()`) |
| **Evaluation suite** – deterministic code scorers, Jev typed judges, a ground-truth judge against an answer key, optional LLM judges | `src/evaluate.py`, `src/jev_judge.py` |
| **Error localisation** – *fact recall* (answer vs. answer key) vs. *context recall* (retrieved passages vs. answer key) separates retrieval errors from generation errors | `jev_facts` in `src/jev_judge.py` |
| **Prompt versioning & safe rollout** – Prompt Registry with `production` / `candidate` aliases; A/B by evaluating both; promote or roll back by moving an alias | `src/prompts.py` |
| **Retrieval engineering** – dense, BM25 (implemented from scratch), Reciprocal Rank Fusion, Jev re-ranking, contextual chunk headers | `src/hybrid.py`, `src/rag.py`, `src/ingest.py` |
| **Run comparison** – per-question diff of two runs, written to Markdown | `src/compare_runs.py` |
| **Human feedback loop** – reviewer ratings logged as assessments on traces; low-confidence judge verdicts flagged `needs_review` | `src/feedback.py`, `src/jev_judge.py` |
| **Serving strategy** *(scaffolded, not yet run)* – script to run the eval set through an API vs. a self-hosted model and report p50/p95 latency and $/1k queries | `src/compare_serving.py` |
| **Infra as code** – MLflow server on EC2, artifacts in S3 (optional) | `infra/main.tf` |

## Architecture

```
                        ┌──────────── MLflow: traces · prompt registry · eval runs ────────────┐
question ─► predict ─► run [CHAIN]                                                             │
                        ├─ retrieve [RETRIEVER]                                                │
                        │    ├─ dense: sentence-transformer embeddings (Chroma, cosine) ─┐     │
                        │    ├─ sparse: BM25 over chunk bodies ──────────────────────────┼─ RRF│
                        │    └─ _jev_rerank [RERANKER]: one Noul per candidate ◄─────────┘     │
                        ├─ load_prompt("sensor-doc-qa@production")                             │
                        └─ Completions [LLM]: Llama 3.1 8B via Ollama ─► answer with [n] cites │
                        └──────────────────────────────────────────────────────────────────────┘
evaluate.py ── code scorers + Jev judges + ground-truth judge over eval/eval_set.json
compare_runs.py ── per-question diff of two runs → reports/compare_A_vs_B.md
```

**Chunks** are ~800 characters with 50 overlap, each prefixed with a contextual header
`[K253911 | Derma-2]` extracted automatically from the document.

## Evaluation design

**Eval set:** 21 questions written from the 13 PDFs, each with `expected_facts` taken verbatim
from the documents: 11 single-document facts, 4 multi-document (e.g. the predicate chain
Spectrum-2 → Clarifi → Ox-Imager CS → OxyVu-1), 4 comparisons, 1 medical-advice probe,
1 out-of-scope question.

**Scorers**

| Scorer | Type | Measures |
|---|---|---|
| `cites_sources`, `concise` | code | answer cites `[n]`; ≤ ~5 sentences |
| `jev_grounded` | Jev Choice | do the retrieved passages *support / contradict / say nothing about* the answer |
| `jev_no_medical_advice` | Jev Noul | answer gives no personal medical advice or diagnosis |
| `jev_admits_unknown_correctly` | 2 × Jev Noul, combined in code | says "not covered" **iff** the context really lacks the answer |
| `jev_completeness` | Jev Score (3 levels) | how fully the question is addressed |
| `jev_fact_recall` | Jev Noul per expected fact | share of answer-key facts stated in the answer (**ground truth**) |
| `jev_context_recall` | Jev Noul per expected fact | share of answer-key facts present in the retrieved passages |

Jev returns typed answers with calibrated probabilities in one parallel call, so all judges for
an answer cost one request. Noul thresholds follow TypeSafe's guidance (≥ 0.7 yes,
≤ 0.3 no); values in between, and Choice/Score confidence < 0.5, are flagged `needs_review`.

## Results

All runs: Llama 3.1 8B (Ollama, temperature 0), 13 documents / 335 chunks, 21 questions.
With n = 21, differences of about ±0.05 are within run-to-run judge noise (see *Lessons*).

### 1. Prompt A/B – stricter prompt (dense retrieval, top_k = 4)

v2 adds: answer only from context, cite every claim, no medical advice, ≤ 5 sentences.

| Metric | v1 | v2 |
|---|---|---|
| cites_sources | 0.14 | **0.81** |
| concise | 0.76 | **1.00** |
| jev_grounded | **0.95** | 0.81 |
| jev_completeness | **0.84** | 0.78 |
| total eval time | 2:59 | **2:16** |

v2 achieved its goal and was faster (shorter answers), but lost groundedness. Reading the
per-question diff showed **only one real regression** (the medical question: a bare "not
covered" instead of a clinician referral); the other drops were questions where *both*
versions failed because the right chunk was never retrieved.

### 2. Retrieval: top_k 4 → 8, overlap 150 → 50 (+ judge fix, see Lessons)

| Metric | v1 | v2 |
|---|---|---|
| jev_grounded | 0.95 → 1.00 | 0.81 → **1.00** |
| jev_completeness | 0.84 → 0.88 | 0.78 → **0.92** |
| jev_admits_unknown_correctly | 0.81 → 0.86 | 0.86 → **0.95** |

With better retrieval, v2 matched or beat v1 on every metric → **v2 promoted**. (Three things
changed at once here; the next experiments change one thing at a time.)

### 3. Adding a ground-truth judge – v2 vs. v3

v3 keeps citations but: answer every part of the question, refuse only if *no* passage is
relevant, and for medical questions refer to a clinician plus give general device info.

| Metric | v2 | v3 |
|---|---|---|
| **jev_fact_recall** (vs. answer key) | 0.44 | **0.60** |
| jev_context_recall | 0.65 | 0.67 |
| medical question – fact recall | 0.00 | **1.00** |
| jev_admits_unknown_correctly | **1.00** | 0.81 |

The proxy judges (answer vs. context) said v3 was slightly worse; the ground-truth judge said it
was clearly better. The proxy drop came largely from judges penalising the *desired* clinician
referral → **v3 promoted**.

### 4. Retrieval strategy (prompt v3 fixed, one change per run)

| Metric | dense | + hybrid (BM25 + RRF) | + Jev re-rank | + chunk headers |
|---|---|---|---|---|
| jev_context_recall | 0.67 | 0.75 | **0.80** | 0.78 |
| jev_fact_recall | 0.60 | 0.59 | 0.65 | **0.67** |
| jev_grounded | 0.95 | 0.90 | **1.00** | 0.95 |
| jev_admits_unknown_correctly | 0.81 | 0.76 | 0.86 | **0.90** |
| jev_completeness | 0.89 | 0.88 | 0.89 | **0.92** |

- **Hybrid** raised context recall (+8 pp): exact tokens like "FLIR A50" and "35 mK" are found by
  BM25 but missed by embeddings. Fact recall did not move – more context alone did not help.
- **Jev re-ranking** pushed the most relevant chunks to the top: context recall 0.80, fact recall +6 pp.
- **Chunk headers** fixed a specific entity mix-up (Derma-2's camera had been reported as the
  Med-Hot camera from another document) but were neutral overall.

**Cost of better retrieval.** Going from top_k 4 to 8 roughly doubled the prompt: average input
tokens per query rose from ≈766 to ≈1,514, and chunk headers added another ≈5 % (≈1,593).
On a laptop (Llama 3.1 8B via Ollama, plus Jev re-ranking), the final configuration takes
≈15 s per question in a sequential eval run (21 questions in 5:24) and ≈27 s for a single
cold query from the CLI. Self-hosted, the marginal cost per query is zero; the price is latency.

## Lessons learned

1. **Retrieval was the bottleneck, not the prompt.** About a third of the needed facts were never
   retrieved at first; no prompt can fix that. Measuring *context recall* separately made this visible.
2. **Proxy judges and ground truth can disagree – keep both.** Judges that compare the answer
   with the retrieved context cannot tell whether the context was the right one.
3. **Grounded ≠ correct.** With re-ranking, an answer cited a real camera spec – from the wrong
   device. `jev_grounded` passed it; only `jev_fact_recall` caught it.
4. **Judges need testing too.** Two judge bugs surfaced only by reading answers: correct refusals
   were scored as ungrounded/incomplete, and the citation regex rejected `[1, 2, 3]`, making a
   prompt look worse than it was.
5. **Retrieval components interact.** Chunk headers helped embeddings but hurt BM25 (generic words
   like "Thermal Camera" in every chunk boosted whole documents), so BM25 scores bodies only.
6. **Know your noise before trusting small deltas.** Two identical runs of the final configuration
   agreed exactly on fact recall (0.67), context recall (0.78), groundedness and refusal correctness;
   only `concise` and `completeness` moved, by ≤ 0.05. Wording and citation numbers can still vary
   between runs because Jev's re-ranking order shifts slightly for near-tied passages.
7. **Measure the harness, not just the model.** A 3-minute "LLM latency" was MLflow evaluating in
   parallel against a single local Ollama instance – requests were queueing. Sequential eval fixed it.

## Known limitations & next steps

- **Tables:** PDF text extraction flattens comparison tables, so the model still swaps MIMOSA's
  wavelengths with its predicate's (same document, different column). Next: layout-aware parsing
  (e.g. PyMuPDF/pdfplumber tables) or table-to-text conversion at ingest.
- **Multi-hop questions:** the second hop of the predicate chain (Ox-Imager CS → OxyVu-1) is often
  missed. Next: query decomposition (one retrieval per sub-question).
- **Eval set size:** 21 questions; extend to 50–100 and run each configuration 2–3 times.
- **Medical questions:** add a dedicated judge for "appropriate clinician referral" instead of
  relying on generic grounding/completeness judges.
- **Routing:** `jev_judge.route()` (answer / decline-medical / out-of-scope) exists but is not yet in
  the serving path; it would skip retrieval + generation for ~10 % of questions.
- **Serving comparison:** `compare_serving.py` is written but not yet run against a hosted API,
  so there are no API cost/latency numbers yet.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # add TYPESAFE_API_KEY (and OPENAI_API_KEY if using OpenAI)
set -a; source .env; set +a

mlflow server --port 5001         # separate terminal; UI at http://127.0.0.1:5001
ollama pull llama3.1:8b           # if LLM_PROVIDER=ollama

# Put PDFs in data/raw/ (see Data), then:
python src/ingest.py --reset
python src/prompts.py register && python src/prompts.py register-v3 && python src/prompts.py promote 3
python src/rag.py "What imaging technology does Spectrum-2 use?"

# Evaluate and compare
python src/evaluate.py --alias production --offline --jev --name my-run
python src/compare_runs.py my-run other-run
```

`--offline` skips the optional OpenAI LLM judges; Jev judges run whenever `--jev` is set.
Offline smoke test without any API keys:

```bash
MLFLOW_TRACKING_URI=sqlite:///mlflow.db LLM_PROVIDER=mock EMBEDDINGS=hash RETRIEVAL=hybrid \
  sh -c "python src/ingest.py --reset && python src/prompts.py register && python src/evaluate.py --offline"
```

## Data

13 public FDA 510(k) summaries in `data/raw/` (not committed): Neko Health's **Spectrum-2**
(K254129) and **Derma-2** (K253911); their predicate chains – Clarifi (K181623) → Ox-Imager CS
(K153426) → OxyVu-1 (K073656), and Thermidas IR (K200999) → Med-Hot (K171928); plus related
devices: MIMOSA (K190334), OxiplexTS200 (K232385), Intra.Ox 2.0 (K191676), InTouch Thermal Camera
(K181716), Barco Demetra (K192829) and deepLive LC-OCT (K240610). Download the "Summary" PDFs from
the FDA 510(k) database by K-number.

## Why this matters in a regulated setting

In a medical-device QMS you need to show what the system did, with which version, and that it
was tested before release. Here every answer is traceable to the retrieved passages, the prompt
version and the model; prompt changes are evaluated before promotion and reversible by alias;
and uncertain judge verdicts are routed to human review rather than silently accepted.
