# Plan – bygg på 2 kvällar (+ valfri tredje)

## Kväll 1 – kör igång (≈3 h)
1. `pip install -r requirements.txt`, starta `mlflow server`.
2. Ladda ner 5–10 FDA 510(k)-sammanfattningar (t.ex. pulsoximeter, EKG, hudavbildning) till `data/raw/`.
3. `ingest.py` → `prompts.py register` → ställ 5 frågor med `rag.py`.
4. Öppna MLflow UI → Traces: kolla att du ser run → retrieve → LLM med prompt-version, tokens och kostnad.

## Kväll 2 – utvärdering + A/B (≈3 h)
1. Skriv 15–20 frågor i `eval/eval_set.json` (inkl. 2–3 "utanför dokumenten" och 2 "medicinsk rådgivning").
2. `evaluate.py --alias production` och `--alias candidate` → jämför i UI.
3. Promota vinnaren med `prompts.py promote 2`. Skriv ner resultatet i README → Results.
4. Kör `feedback.py` på 5 frågor så att du har mänsklig feedback i traces.

## Kväll 3 (valfri) – serving + infra
1. Installera Ollama, `ollama pull llama3.1:8b`, kör `compare_serving.py`.
2. `infra/` med Terraform: MLflow på EC2 + artefakter i S3 (kom ihåg `terraform destroy`).

## Till ansökan
- Lägg repot publikt på GitHub med ifylld Results-sektion och en skärmdump av en trace.
- I CV:t under Projects: "Sensor-doc RAG – LLM Ops with MLflow 3: tracing, LLM-judge eval suite incl. custom clinical-safety judge, prompt registry A/B + rollback, API vs self-hosted cost/latency analysis."
- I brevet: koppla direkt till Nekos 6–12-månadersleveranser (tabellen i README).
