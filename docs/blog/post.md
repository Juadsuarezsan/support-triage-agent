# Triage de tickets de soporte con un clasificador pequeño, un LLM y reglas que no se negocian

*Borrador para Medium / Dev.to. Código: https://github.com/Juadsuarezsan/support-triage-agent*

Cuando un equipo de soporte recibe dos mil tickets al día, el problema no es
responder: es decidir qué merece la atención de una persona. La mitad de los
tickets son variaciones de tres preguntas con respuesta estándar; una fracción
pequeña son emergencias que deberían saltarse la cola; y en el medio hay una
masa gris de casos que un agente resuelve en un minuto si alguien le pone
delante la resolución de un caso parecido. Este artículo cuenta cómo construí
un agente de triage que hace exactamente esa separación, qué decisiones de
diseño tomé, cómo lo evalué sin engañarme y qué dejé documentado como
pendiente por falta de llaves.

## El flujo en una frase

Un ticket entra por un endpoint HTTP, un clasificador fine-tuneado le pone
intent, un LLM lee urgencia y tono, un buscador semántico trae los cinco
tickets resueltos más parecidos, el LLM redacta una respuesta apoyada en esas
resoluciones y una función de reglas puras decide entre tres salidas:
`auto_resolve` (se envía sola), `suggest` (un agente revisa) o `escalate`
(va a la cola de humanos con justificación). Todo se registra en un audit log
con puntuaciones, tokens, costo en dólares y latencia.

Esto corre como un grafo LangGraph de seis nodos. Cada nodo escribe en un
estado tipado, loguea las claves que leyó y lo que devolvió, y comparte un
`trace_id` que aparece en cada línea de log, en la respuesta HTTP y en la
fila de auditoría.

## Decisión 1: un clasificador pequeño para lo que es una clasificación

La tentación con un LLM a mano es pedirle todo: "dime el intent, la
prioridad y escribe la respuesta". Funciona, y es lento y caro en cada
ticket. El intent es una clasificación cerrada de 27 clases con 26.872
ejemplos etiquetados (dataset Bitext). Eso lo resuelve un DistilBERT con un
adapter LoRA de 750 mil parámetros entrenado en 199 segundos en CPU, con
macro-F1 de 0.9864 sobre un split de test de 449 ejemplos. Inferencia:
50-110 milisegundos por ticket, sin costo por token.

Claude queda para las dos cosas que un clasificador no hace: leer matices
("hoy", "amenazo con demandar", "es la tercera vez") y convertirlos en una
prioridad P0-P3 con justificación, y redactar una respuesta que use la
evidencia de casos anteriores sin inventar políticas.

Sobre el número 0.9864 conviene ser preciso: es sobre un subconjunto
estratificado de 4.482 filas (~166 por clase), no sobre el 10 % completo del
dataset. Con 17 ejemplos de test por clase, un solo error mueve el F1 de una
clase seis centésimas. Lo digo en el README, en la model card y aquí, porque
es la diferencia entre una métrica y una anécdota.

## Decisión 2: reglas duras antes del score

El score de confianza combina tres señales: `0.5·intent + 0.4·similitud +
0.1`, menos 0.15 si el sentimiento es negativo y otros 0.15 si la urgencia
supera 0.85. Auto-resolver exige score ≥ 0.85; eso solo se alcanza con un
intent muy seguro *y* una resolución previa muy parecida. Es decir: el sistema
solo cierra solo lo que ya se resolvió antes de la misma manera.

Pero hay dos casos donde el score no importa. Si el intent es `complaint` o
`contact_human_agent`, el ticket escala, aunque la confianza sea 0.99. Un
cliente que pide hablar con una persona y recibe una respuesta automática
perfecta ha recibido una mala respuesta. Lo mismo con urgencia ≥ 0.9. Esas
reglas están en una función sin dependencias, probada con tests unitarios,
antes de cualquier umbral configurable.

## Decisión 3: cada componente tiene un fallback determinista, y se etiqueta

Sin `ANTHROPIC_API_KEY` y sin instalar torch, el servicio arranca, la suite
de 131 tests pasa y la evaluación produce números. Cada componente cae a una
versión determinista: un clasificador por palabras clave con 27 reglas, una
heurística de prioridad, una plantilla de respuesta, un encoder que hashea
tokens. Esto existe para CI, para reproducir el proyecto en un portátil y para
el demo estático.

La trampa sería presentar esos números como el sistema. No lo son, y el
código lo impide: el JSON de cada corrida lleva `"label": "fallback
determinista, sin LLM"`, la tabla de resultados tiene una fila separada con
ese nombre y las filas del sistema real quedan como `pendiente (requiere
ANTHROPIC_API_KEY)` hasta que exista una corrida guardada. El demo dice en
su cabecera con qué modo se generaron las tarjetas.

## Decisión 4: un solo cliente de Claude, mockeado a nivel HTTP

Todas las llamadas al modelo pasan por una clase de 100 líneas que envuelve
el SDK oficial con un timeout de 15 segundos, reintentos con backoff
exponencial sobre 429, 5xx, 529 y timeouts (gobernados por `tenacity`, con
los reintentos del SDK apagados para que la política esté en un solo sitio),
contabilidad de tokens y un parser que convierte JSON mal formado en una
excepción tipada en vez de un `KeyError` en medio del grafo.

Probar esto no requiere llave: `respx` intercepta el endpoint `/v1/messages`
y devuelve respuestas fabricadas, incluidas secuencias "529 y luego 200" para
verificar el reintento, y "400" para verificar que un error de petición no se
reintenta. El pipeline entero, con las tres llamadas al modelo mockeadas,
acumula 300 tokens de entrada y 60 de salida y calcula el costo exacto con los
precios configurados. Ese test es el que garantiza que `cost_usd` en
producción no es un número decorativo.

## Lo que la evaluación offline enseñó (y lo que no puede enseñar)

El eval set son 108 tickets sintéticos escritos a mano, cuatro por intent,
con decisión y rango de prioridad etiquetados manualmente según una política
documentada. Corriendo el fallback sobre ellos:

- Macro-F1 de intent 0.948: el router léxico acierta casi siempre, y sus siete
  fallos son sinónimos y artículos ("create an account" no coincide con
  "create account").
- Tasa de auto-resolución **0 %** y acuerdo de decisión 49 %: con un encoder
  léxico la similitud media es 0.51 y el score nunca llega a 0.85. Nada se
  cierra solo.
- La ablación lo confirma: quitando el término de similitud, el acuerdo sube a
  54.6 % y las escalaciones falsas bajan de 17.9 % a 4.2 %. Con un encoder
  léxico la similitud es ruido.

¿Conclusión? No que haya que bajar el peso de la similitud. Los umbrales
están calibrados para similitud semántica real (`all-mpnet-base-v2`), donde
un ticket "¿dónde está mi pedido?" y "mi paquete no llega" sí se parecen. La
corrida offline mide la lógica de enrutamiento con señales pobres; la corrida
con el sistema real, que necesita una llave, es la que dirá si 0.85 es el
umbral correcto. Ambas cosas están escritas en `docs/error_analysis.md`.

## Observabilidad que cabe en una línea de log

Cada petición produce líneas como:

```
2026-09-29 01:11:12.034 | DEBUG | trace=3f1c… | node=decide_route in_keys=[…] out={'decision': "TriageDecision(decision='suggest', confidence=0.71…", 'latency_ms': '5'}
2026-09-29 01:11:12.036 | INFO  | trace=3f1c… | triage done ticket_id=q-01 intent=get_refund decision=suggest latency_ms=5 tokens=0/0 cost_usd=0.0
```

El `trace_id` vive en un `ContextVar`, así que no hay que pasarlo por
parámetros. LangSmith está cableado por variables de entorno para cuando haya
llave; el grafo no cambia.

## El bug que impedía arrancar, y lo que enseña sobre LangGraph

El repositorio llegó a esta fase con el API sin arrancar. La causa cabía en
una línea: el grafo tenía un nodo llamado `sentiment` y el estado tenía una
clave llamada `sentiment`. LangGraph lo rechaza en tiempo de construcción
("'sentiment' is already being used as a state key"), y como el grafo se
construía al importar el módulo, ni el servidor ni la evaluación llegaban a
ejecutar nada. Había dos colisiones más esperando detrás (`similar` y
`draft`).

La corrección fue renombrar los nodos con verbo (`classify_intent`,
`analyze_priority`, `retrieve_similar`, `draft_solution`, `decide_route`,
`record_audit`) y dejar la regla escrita en un comentario junto al grafo.
Pero lo interesante es por qué la regla existe: en LangGraph un nodo es
también un destino de enrutamiento y un canal por el que fluyen escrituras;
compartir nombre con una clave del estado haría ambiguo `add_edge("x", ...)`.
Nombrar nodos como acciones y claves como datos resuelve la ambigüedad y,
de paso, hace que los logs por nodo se lean como una historia.

## Lo que queda, y por qué está escrito como pendiente

- Correr las tres filas de la tabla comparativa (zero-shot, clasificador
  solo, híbrido) con llave y el extra `ml`, y el juez LLM sobre cien
  respuestas con una rúbrica de nueve criterios numerados que ya está en el
  código.
- Publicar el adapter en Hugging Face Hub (la model card está lista).
- Reentrenar sobre el split completo y guardar la matriz de confusión (el
  script ya la escribe).
- Una muestra de 200 conversaciones reales de Twitter para ver cuánto pierde
  un clasificador entrenado sobre plantillas cuando llegan typos y emojis.

Nada de esto requiere cambiar código; requiere llaves, red y un poco de
tiempo de CPU. La disciplina del proyecto fue no rellenar esas celdas con
estimaciones y dejar que el README diga "pendiente" donde corresponde.

## Para llevarse

1. Si la tarea es una clasificación cerrada con datos etiquetados, un encoder
   pequeño con LoRA la resuelve más rápido y más barato que un LLM.
2. Las decisiones que no se negocian (un humano pidió un humano) van en reglas
   antes del score, no en un prompt.
3. Un fallback determinista hace el proyecto reproducible; etiquetarlo como
   fallback lo hace honesto.
4. Mockea el LLM a nivel HTTP: así pruebas reintentos, timeouts y
   contabilidad de tokens, no solo el camino feliz.
5. Una métrica sin su alcance (n, split, condiciones) es una anécdota.
