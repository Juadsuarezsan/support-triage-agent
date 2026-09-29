# Decisiones técnicas

Cada entrada registra qué se eligió, qué alternativa se descartó y con qué
criterio. Cuando la decisión fue validada con una medición, se cita el archivo.

## 1. DistilBERT + LoRA para el intent, no Claude para todo

El intent es una clasificación cerrada de 27 clases con miles de ejemplos
etiquetados (Bitext). Un encoder pequeño fine-tuneado lo resuelve en ~50-110 ms
en CPU (mediciones del bake anterior, commit `a6a409a`) y sin costo por token;
Claude tarda segundos y cuesta dinero en cada ticket. Claude se reserva para lo
que un clasificador no hace: leer matices de urgencia y redactar. Con el
subconjunto estratificado de Bitext, el adapter alcanzó macro-F1 0.9864 en 449
ejemplos de test (`models/intent-classifier-lora/training_metrics.json`).

**Descartado:** RoBERTa-base (el doble de parámetros para una tarea donde
DistilBERT ya satura) y zero-shot Claude como clasificador principal (queda
como baseline y como fallback cuando no hay pesos).

## 2. LoRA en vez de fine-tune completo

Con r=8 sobre `q_lin`/`v_lin` se entrenan ~750 K parámetros de 67 M. El adapter
pesa unos pocos MB, se versiona aparte del modelo base y se entrena en 199 s
en CPU, lo que hace reproducible el experimento sin GPU. Un fine-tune completo
no habría cambiado el resultado en una tarea saturada y sí el costo de
reentrenar y publicar.

## 3. Qdrant y no Chroma ni pgvector para los tickets similares

- Chroma es cómodo en un notebook pero su servidor no ofrece filtrado por
  payload con índices ni snapshots; para una base de tickets que crece a diario
  y se filtra por tenant/intent, Qdrant lo hace de fábrica.
- pgvector habría reutilizado el Postgres del audit log, pero mezcla cargas de
  trabajo: el índice HNSW compite en memoria con las tablas transaccionales.
- Qdrant expone un cliente async con tipos, corre en un contenedor de 100 MB y
  su distancia coseno con vectores normalizados coincide con la implementación
  en memoria usada en tests, así que la misma suite valida ambos backends.

## 4. Umbrales 0.85 / 0.60 y reglas duras antes del score

El score combina 0.5·intent + 0.4·similitud + 0.1 − penalizaciones. El 0.85
para auto-resolver exige a la vez intent alto (>0.9) y una resolución previa
muy parecida (>0.8): el sistema solo cierra solo lo que ya se resolvió antes
igual. Por debajo de 0.60 no hay evidencia suficiente ni para sugerir. Las
reglas duras (`complaint`, `contact_human_agent`, `urgency ≥ 0.9`) van antes
del score porque un cliente que pide un humano no debe recibir una respuesta
automática por muy alta que sea la confianza.

La ablación del run offline (`eval/RESULTS.md`) muestra que con el encoder
hash el término de similitud sube las escalaciones falsas de 4.2 % a 17.9 %:
los umbrales están calibrados para similitudes semánticas reales, no léxicas.
Esa es la razón por la que el fallback nunca se presenta como el sistema.

## 5. LangGraph y no una cadena lineal a mano

El flujo es lineal hoy, pero el estado tipado, los nombres de nodo y el
`ainvoke` con trazas por nodo son lo que permite añadir ramas (por ejemplo,
saltar el drafter cuando la decisión será `escalate`, o paralelizar
`analyze_priority` y `retrieve_similar`) sin reescribir el orquestador.
También es la ruta natural hacia LangSmith. El costo es una dependencia más y
la regla de no repetir claves de estado como nombres de nodo.

## 6. Cliente `anthropic` directo en vez de `langchain-anthropic`

Un único `ClaudeClient` con timeout, `tenacity` y contabilidad de tokens es
más fácil de probar (un endpoint HTTP mockeado con `respx`) y de auditar que
tres `ChatAnthropic` construidos dentro de cada agente. Se mantiene
`langchain-core` porque LangGraph lo requiere.

## 7. Fallback determinista en cada componente, etiquetado como tal

Sin `ANTHROPIC_API_KEY` y sin el extra `ml` el servicio arranca, la suite pasa
y `python -m eval.run` produce números. Esos números se etiquetan como
"fallback determinista, sin LLM" en el JSON, en `RESULTS.md` y en el demo.
La alternativa (fallar sin llave) habría impedido CI y la reproducción local;
la otra alternativa (presentar el fallback como el sistema) sería mentir.

## 8. Datos sintéticos escritos a mano para el eval set

Bitext y Twitter no eran alcanzables desde el entorno de esta fase. En vez de
inventar descargas, el eval set son 108 tickets escritos a mano con etiqueta
manual y `synthetic: true`, cuatro por intent, y `docs/data_schema.md` fija la
política de etiquetado. Cuando exista acceso a Kaggle, la muestra de 200
conversaciones de Twitter se añade como archivo separado sin tocar este.

## 9. `psycopg` 3 con pool async y esquema creado al arrancar

`asyncpg` es algo más rápido, pero `psycopg` 3 comparte tipos con el resto del
ecosistema Postgres, soporta `dict_row` y su pool async se puede sustituir por
un doble en tests sin monkeypatching del driver. El esquema es `CREATE TABLE
IF NOT EXISTS` porque no hay migraciones que gestionar todavía.

## 10. Piso de cobertura 70 % con la lógica de dominio al 100 %

El 70 % es el gate de CI; la cobertura real está en 96.8 % porque cada módulo
de dominio se prueba con dobles. Lo que queda sin cubrir es la conexión real a
Postgres y a Qdrant, que necesita Docker.
