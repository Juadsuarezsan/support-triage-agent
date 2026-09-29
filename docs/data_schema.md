# Esquema de datos

Este documento describe cada archivo de datos que el proyecto lee o produce,
con tipos, rangos y procedencia. Ningún dataset crudo se versiona: `data/raw/`
y `data/processed/` están en `.gitignore`.

## 1. Bitext Customer Support (entrenamiento del clasificador)

- **Fuente:** `https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset`
- **Licencia:** CC BY 4.0
- **Descarga:** `python scripts/download_data.py --bitext` (CSV directo del repo del dataset; requiere red a `huggingface.co`)
- **Tamaño:** 26.872 filas, 27 intents, 11 categorías

| Columna | Tipo | Descripción |
|---|---|---|
| `flags` | str | Etiquetas de estilo lingüístico (p. ej. `B` básico, `Q` con typos, `W` coloquial, `Z` con ruido). Se ignoran. |
| `instruction` | str | Texto del usuario (la "query"). Entrada del clasificador. Longitud típica 5-40 palabras. |
| `category` | str | Una de 11 categorías gruesas (`ORDER`, `REFUND`, `ACCOUNT`, ...). No se usa. |
| `intent` | str | Una de las 27 clases. Etiqueta objetivo. |
| `response` | str | Respuesta plantilla del dataset. No se usa para entrenar el clasificador. |

**Splits.** `python scripts/make_splits.py` genera `data/splits/{train,val,test}.txt`
con índices de fila (base 0) del CSV, estratificado por `intent` en proporciones
80/10/10 con `random.Random(20260516)`, y `SPLIT_INFO.json` con el SHA-256 del
CSV fuente y los conteos por intent. El script de entrenamiento
(`notebooks/01_classifier_training.py`) usó además un submuestreo estratificado
a 4.482 filas (~166 por clase) para caber en CPU; sus métricas versionadas
corresponden a ese subconjunto (test = 449 ejemplos), no al 10 % completo
(2.687). Ver `models/intent-classifier-lora/training_metrics.json`.

## 2. Customer Support on Twitter (evaluación end-to-end, pendiente)

- **Fuente:** `https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter`
- **Licencia:** CC BY-NC-SA 4.0
- **Descarga:** `python scripts/download_data.py --twitter` (API de Kaggle, requiere `KAGGLE_USERNAME`/`KAGGLE_KEY`)
- **Tamaño:** ~3 M tweets

| Columna | Tipo | Descripción |
|---|---|---|
| `tweet_id` | int | Identificador único del tweet. |
| `author_id` | str | Autor anonimizado; las marcas conservan su handle. |
| `inbound` | bool | `True` si lo escribió un cliente hacia una marca. |
| `created_at` | str | Fecha en formato Twitter. |
| `text` | str | Contenido del tweet. |
| `response_tweet_id` | str | IDs (separados por coma) de las respuestas. |
| `in_response_to_tweet_id` | float | ID del tweet al que responde (NaN si inicia hilo). |

La muestra de 200 conversaciones para la evaluación end-to-end (spec) no se ha
construido: depende de credenciales de Kaggle. Cuando exista, se guardará en
`data/eval/twitter_conversations.jsonl` con el mismo esquema de `queries.jsonl`
más `thread: list[str]`.

## 3. `data/eval/queries.jsonl` (eval set, versionado)

108 tickets **sintéticos escritos a mano** (4 por intent), con ground truth
manual. Marcados con `synthetic: true`. No provienen de Bitext ni de Twitter.

| Campo | Tipo | Rango / valores | Descripción |
|---|---|---|---|
| `id` | str | `q-001` … `q-108` | Identificador único. |
| `query` | str | 20-120 caracteres | Cuerpo del ticket. |
| `expected_intent` | str | uno de los 27 intents Bitext | Intent correcto. |
| `expected_decision` | str | `auto_resolve` / `suggest` / `escalate` | Decisión de triage esperada. |
| `expected_priority_range` | list[str] | subconjunto de `P0..P3` | Prioridades aceptables. |
| `synthetic` | bool | `true` | Declaración de origen sintético. |

**Política de etiquetado de `expected_decision`** (aplicada a mano, caso por caso):

- `escalate`: `complaint`, `contact_human_agent`, `delete_account` (acción irreversible / GDPR) y `payment_issue` con dinero perdido (cobro doble).
- `auto_resolve`: consultas informativas con respuesta estándar: `check_refund_policy`, `check_cancellation_fee`, `check_payment_methods`, `delivery_options`, `delivery_period`, `newsletter_subscription`, `contact_customer_service`, `track_order`, `track_refund`, `get_invoice`.
- `suggest`: acciones sobre pedidos o cuentas que un agente debe confirmar: `cancel_order`, `change_order`, `change_shipping_address`, `set_up_shipping_address`, `create_account`, `edit_account`, `switch_account`, `recover_password`, `registration_problems`, `review`, `place_order`, `check_invoice`, `get_refund`, `payment_issue` sin pérdida de dinero.

## 4. `data/eval/seed_tickets.jsonl` (base de conocimiento, versionado)

15 tickets resueltos sintéticos que se indexan al arrancar el API y en la evaluación.

| Campo | Tipo | Descripción |
|---|---|---|
| `ticket_id` | str | `t-001` … |
| `intent` | str | Intent Bitext. |
| `body` | str | Texto original del ticket. |
| `resolution` | str | Resolución aplicada; alimenta el drafter. |

## 5. `data/MANIFEST.txt`

TSV con `name`, `sha256`, `bytes`, `path`, `license`, `source`, `downloaded_at`.
Se actualiza con `python scripts/download_data.py --local` (archivos versionados)
y con cada descarga real. Cambiar un archivo de datos sin regenerar el manifiesto
rompe la trazabilidad.

## 6. Salidas de evaluación

- `eval/runs/<AAAA-MM-DD>-<sistema>.json`: `EvalReport` completo (métricas agregadas, F1 por clase, casos individuales con intents, similares, sentimiento y urgencia) más la ablación.
- `eval/RESULTS.md`: regenerado por `python -m eval.run` únicamente a partir de los JSON anteriores.
