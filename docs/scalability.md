# Escalabilidad

Cómo se comporta el sistema al crecer el volumen de tickets y dónde se rompe.
Las cifras de latencia y costo con LLM son estimaciones con supuestos
explícitos: la única corrida versionada hasta ahora es el fallback sin LLM
(`eval/RESULTS.md`).

## Perfil de una petición (configuración de producción)

| Paso | Recurso | Tiempo estimado | Costo |
|---|---|---|---|
| DistilBERT+LoRA en CPU | CPU del API | 50-110 ms (bake anterior, commit `a6a409a`) | 0 |
| `analyze_priority` (Claude, ~300 tokens in / ~60 out) | red + API | 1-3 s | ≈ $0.0018 |
| `retrieve_similar` (mpnet en CPU + Qdrant) | CPU + red local | 30-80 ms | 0 |
| `draft_solution` (Claude, ~600 in / ~120 out) | red + API | 2-4 s | ≈ $0.0036 |
| `decide_route` + `record_audit` | CPU + Postgres | < 5 ms | 0 |

Supuestos: precios $3 / $15 por millón de tokens (`PRICE_*_PER_MTOK`), sin
cache de prompt. **Costo estimado por ticket ≈ $0.005-0.008; latencia p95
estimada 4-7 s**, dominada por las dos llamadas a Claude en serie.

## Capacidad de una réplica

Con `uvicorn --workers 4` y llamadas async, la réplica está limitada por la
concurrencia que el proveedor permita, no por CPU: cada petición pasa ~90 % de
su tiempo esperando la red. 4 workers × ~20 peticiones en vuelo ≈ 80 tickets
en paralelo ≈ **600-900 tickets/min** si el rate limit de la cuenta lo
permite. La inferencia local (DistilBERT + mpnet) consume ~150 ms de CPU por
ticket, es decir un núcleo soporta ~400 tickets/min.

## 100× (≈ 1.000 tickets/hora sostenidos)

Funciona sin cambios de arquitectura:

- Postgres con el índice `triage_audit_created_at_idx` absorbe 1.000 inserts/h
  sin tocar configuración.
- Qdrant con 100 K tickets resueltos cabe en memoria (768 × 4 bytes × 100 K ≈
  300 MB).
- El cache de embeddings por SHA-256 evita re-codificar consultas repetidas.

Qué vigilar: el rate limit de Anthropic (pedir tier superior) y el costo, que
a $0.007/ticket son ~$170/día a ese volumen. La primera palanca es
paralelizar `analyze_priority` con `retrieve_similar` (no dependen entre sí),
que recorta ~1-2 s del p95 sin costo.

## 1.000× (≈ 10.000 tickets/hora)

- Sacar Claude del camino síncrono: el API encola el ticket, un worker corre el
  grafo y publica el resultado; el Kanban lee `/api/audit/recent` o un
  websocket. Sin esto, cada réplica retiene miles de conexiones abiertas.
- Saltar `draft_solution` cuando `decide_route` va a escalar por regla dura
  (`complaint`, `contact_human_agent`): hoy se redacta antes de decidir; con
  reordenar el grafo se ahorra la llamada más cara en ~10-15 % de tickets.
- Prompt caching del system prompt del drafter y del analizador: reduce el
  costo de entrada ~80 % en esas dos llamadas.
- Réplica de lectura de Postgres para el Kanban y los reportes.
- Qdrant con `on_disk` para vectores y HNSW `m=16`, `ef=128`, y sharding por
  tenant cuando la colección supere ~5 M puntos.

## 10.000× (≈ 100.000 tickets/hora)

Territorio de varios servicios:

- Clasificación e inferencia local en un servicio propio (Triton o un
  `transformers` server) con autoscaling por CPU, separado del API.
- Colas particionadas por tenant; workers con presupuesto de tokens por tenant
  para evitar que un cliente agote la cuota de todos.
- Modelo más pequeño (Haiku) para `analyze_priority`, que es una tarea de
  clasificación estructurada, y reservar Sonnet para el borrador.
- Audit log a un almacén columnar (ClickHouse/BigQuery) por lotes; Postgres
  solo para el estado vivo.

## Costo de operar con 1.000 usuarios mensuales

Supuesto: 1.000 usuarios finales generan ~5 tickets/usuario/mes = 5.000
tickets/mes.

| Concepto | Cálculo | USD/mes |
|---|---|---|
| Claude (2 llamadas/ticket) | 5.000 × $0.007 | ≈ 35 |
| API + inferencia local | 1 VM 2 vCPU / 4 GB | ≈ 25 |
| Postgres gestionado (pequeño) | | ≈ 15 |
| Qdrant (contenedor en la misma VM o cloud free tier) | | 0-25 |
| **Total** | | **≈ 75-100 USD/mes** |

El coste marginal es casi todo Claude; el fijo, la VM. A 50.000 tickets/mes la
factura de tokens sube linealmente (~$350) y conviene activar prompt caching y
el salto del drafter en escalaciones.

## Lo que no se escala automáticamente

La evaluación (`python -m eval.run`) corre en un solo proceso y contra el
mismo eval set para que los números sean comparables entre corridas. Escalar
la evaluación significa añadir eval sets (Twitter, multi-idioma, tickets
largos), no paralelizarla.
