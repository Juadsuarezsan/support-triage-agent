# Post de LinkedIn (borrador)

Publiqué un agente de triage para tickets de soporte, y lo más útil que aprendí no fue de IA sino de disciplina con las métricas.

Qué hace: un ticket entra por HTTP, un DistilBERT con LoRA le pone intent (27 clases, 199 s de entrenamiento en CPU), Claude lee urgencia y tono, un buscador semántico trae casos resueltos parecidos, Claude redacta con esa evidencia y una función de reglas decide: se envía sola, la revisa un agente o escala. Todo queda en un audit log con trace_id, tokens, costo y latencia.

Tres decisiones que defendería en una entrevista:

1. Un clasificador pequeño para lo que es una clasificación. Macro-F1 0.9864 sobre 449 ejemplos de test, 50-110 ms por ticket, cero costo por token. Claude se reserva para matices y redacción.

2. Reglas duras antes del score. "Quiero hablar con una persona" escala siempre, aunque la confianza sea 0.99. Eso no va en un prompt; va en una función con tests.

3. Fallback determinista en cada componente, etiquetado como tal. Sin llave, el servicio arranca y 131 tests pasan. Pero la tabla de resultados dice "pendiente (requiere ANTHROPIC_API_KEY)" en las filas del sistema real, y la fila del fallback se llama "fallback determinista, sin LLM". Ninguna celda se rellena con estimaciones.

Lo que la evaluación offline enseñó: con un encoder léxico, el término de similitud es ruido (la ablación lo muestra: quitarlo baja las escalaciones falsas de 17.9 % a 4.2 %). Los umbrales están calibrados para similitud semántica real, y la corrida que lo confirme está pendiente de una llave. Está escrito así en el repo.

Stack: FastAPI, LangGraph, anthropic SDK (timeout + tenacity + mocks HTTP con respx), Qdrant, Postgres con psycopg, DistilBERT+PEFT, pytest con 96.8 % de cobertura, mypy --strict, gitleaks en CI.

Repo: https://github.com/Juadsuarezsan/support-triage-agent
Write-up completo en docs/blog/post.md.

#AIEngineering #LLM #CustomerSupport #LangGraph #MLOps
