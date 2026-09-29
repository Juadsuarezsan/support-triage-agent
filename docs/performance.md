# Rendimiento y perfilado

## Lo medido

| Configuración | Fuente | p50 | p95 | Costo/ticket |
|---|---|---|---|---|
| Fallback determinista, sin LLM (108 tickets) | `eval/runs/2026-09-29-offline_fallback.json` | 3 ms | 3 ms | $0 |
| DistilBERT+LoRA, inferencia CPU por ticket (10 tickets) | `demo/predictions.json` del commit `a6a409a` | ~60 ms | ~113 ms | $0 |
| Producción (DistilBERT + Claude + mpnet + Qdrant) | pendiente (requiere ANTHROPIC_API_KEY + extra `ml`) | — | — | — |

El fallback mide el costo fijo del grafo (LangGraph, Pydantic, encoder hash,
coseno sobre 15 semillas, auditoría en memoria): **3 ms por ticket**. Todo lo
que se añada por encima de eso es red o inferencia.

## Dónde está el cuello de botella

En producción el tiempo se reparte, según las estimaciones de
`docs/scalability.md`, así:

```
classify_intent (CPU)      ~0.1 s   ██
analyze_priority (Claude)  ~2 s     ████████████████████████
retrieve_similar (CPU)     ~0.05 s  █
draft_solution (Claude)    ~3 s     ████████████████████████████████████
decide + audit             <0.01 s
```

**Las dos llamadas a Claude son >90 % de la latencia y el 100 % del costo
variable.** Por eso el `ClaudeClient` tiene timeout de 15 s y como máximo 3
intentos: un ticket nunca bloquea más de ~45 s ni multiplica el gasto sin
límite.

## Lo que ya está optimizado

- **Cache de embeddings** (`CachedEncoder`): SHA-256 del texto como clave, LRU
  de 4.096 entradas, y solo los *misses* van al encoder, en un único batch. En
  el run offline el encoder se llama una vez por consulta; en producción evita
  recodificar consultas repetidas y las semillas al reindexar.
- **Indexación por lotes**: `InMemoryTicketStore.index` codifica todas las
  semillas en una llamada; `QdrantTicketStore` sube puntos en lotes de 64.
- **Pool de conexiones** Postgres (`AsyncConnectionPool`, `min 1 / max 8`) y
  una sola transacción por petición para las dos escrituras.
- **Modelo cargado una vez**: `HFTransformerClassifier` y `HFEncoder`
  construyen el pipeline en la primera llamada y lo reutilizan.
- **Endpoints async** con `uvicorn`; ninguna llamada de red es bloqueante.

## Próximas optimizaciones, en orden de retorno

1. **Paralelizar `analyze_priority` y `retrieve_similar`** con un fan-out en
   LangGraph: no comparten entradas más allá de `body`. Ahorro: ~el 100 % del
   tiempo del más corto; en la práctica el de retrieval (50 ms) queda oculto
   bajo el de Claude, pero abre la puerta a mover el análisis a Haiku sin
   alargar el p95.
2. **Decidir antes de redactar** cuando aplica una regla dura: el borrador es
   la llamada más cara y en `complaint` / `contact_human_agent` se descarta.
3. **Prompt caching** del system prompt del drafter (~200 tokens) y del
   analizador: recorta el costo de entrada de esas llamadas.
4. **Batch de embeddings en la ingesta**: cuando se carguen decenas de miles de
   tickets históricos, `HFEncoder.encode` debe recibir lotes de 256 con
   `batch_size` explícito.

## Cómo perfilar

```bash
python -m src.eval.runner --json | python -c "import json,sys; r=json.load(sys.stdin); print(r['latency_p50_ms'], r['latency_p95_ms'])"
LOG_LEVEL=DEBUG uvicorn src.api.main:app   # cada nodo loguea entrada/salida con trace_id
```

Con `LOG_LEVEL=DEBUG` las líneas `node=<nombre>` llevan timestamp con
milisegundos; la diferencia entre nodos consecutivos es el tiempo de cada uno.
