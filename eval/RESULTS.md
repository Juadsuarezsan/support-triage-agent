# Resultados de evaluación

Generado por `python -m eval.run`. Cada número proviene de un archivo en `eval/runs/`; las celdas sin corrida quedan como `pendiente (...)`.

## Tabla comparativa obligatoria

Eval set: `data/eval/queries.jsonl` (tickets sintéticos escritos a mano, ground truth manual, 27 intents). Macro-F1 sobre las 27 clases; latencia p95 end-to-end en el proceso de evaluación; costo estimado con los precios configurados por token.

| Sistema | Macro-F1 | Auto-resolve | Latencia p95 | Costo/ticket |
|---|---|---|---|---|
| Zero-shot Claude | pendiente (requiere ANTHROPIC_API_KEY) | pendiente (requiere ANTHROPIC_API_KEY) | pendiente (requiere ANTHROPIC_API_KEY) | pendiente (requiere ANTHROPIC_API_KEY) |
| DistilBERT fine-tuneado solo | pendiente (requiere extra `ml` + pesos del modelo) | — | pendiente (requiere extra `ml` + pesos del modelo) | pendiente (requiere extra `ml` + pesos del modelo) |
| Híbrido (este sistema) | pendiente (requiere ANTHROPIC_API_KEY + extra `ml`) | pendiente (requiere ANTHROPIC_API_KEY + extra `ml`) | pendiente (requiere ANTHROPIC_API_KEY + extra `ml`) | pendiente (requiere ANTHROPIC_API_KEY + extra `ml`) |
| Fallback determinista, sin LLM (heurística + plantilla + encoder hash) — NO es el sistema en producción | 0.948 | 0.0% | 3 ms | $0.0000 |

## Corrida offline (fallback determinista, sin LLM)

Archivo: `eval/runs/2026-09-29-offline_fallback.json` · n = 108 · clasificador = `keyword_heuristic` · encoder = `cached(deterministic_hash_256)`

| Métrica | Valor |
|---|---|
| Macro-F1 (27 intents) | 0.948 |
| Intent top-1 / top-3 | 93.5% / 93.5% |
| Decision agreement | 49.1% |
| Auto-resolution rate | 0.0% |
| False-escalation rate | 17.9% |
| False auto-resolve rate | 0.0% |
| Priority band match | 86.1% |
| Latencia p50 / p95 | 3 ms / 3 ms |
| Costo por ticket | $0.0000 (sin llamadas a LLM) |

### Ablación del `confidence_score` (misma corrida, decisiones recalculadas)

| Variante | Decision agreement | Auto-resolve | False escalation | False auto-resolve |
|---|---|---|---|---|
| full | 49.1% | 0.0% | 17.9% | 0.0% |
| no_similarity | 54.6% | 0.0% | 4.2% | 0.0% |
| no_sentiment_penalty | 49.1% | 0.0% | 17.9% | 0.0% |
| intent_only | 54.6% | 0.0% | 4.2% | 0.0% |

## LLM-as-judge (helpfulness / tone / correctness, rúbrica en `src/eval/judge.py`)

pendiente (requiere ANTHROPIC_API_KEY) (100 respuestas generadas por el sistema híbrido).

## Clasificador DistilBERT+LoRA (métricas de entrenamiento versionadas)

Fuente: `models/intent-classifier-lora/training_metrics.json`. Alcance exacto: split de test de **449 ejemplos** de un subconjunto estratificado de 4482 filas de Bitext (no el 10 % completo de 2.687). Re-entrenar sobre el split completo requiere red a HF Hub y el extra `ml`.

- Macro-F1 (test, 449 ej.): **0.9864**
- Épocas: 2 · batch 16 · lr 0.0005 · 199 s en CPU
- F1 por clase: ver `docs/classifier_per_class_f1.svg` y `docs/error_analysis.md`

## Cómo regenerar

```bash
python -m eval.run              # offline: fallback determinista + ablación
ANTHROPIC_API_KEY=... python -m eval.run --systems zero_shot_claude hybrid --judge
pip install -e '.[ml]' && python -m eval.run --systems distilbert_only
```
