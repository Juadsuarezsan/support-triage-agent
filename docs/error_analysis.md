# Análisis de errores

Dos fuentes, con alcances distintos que no deben mezclarse:

1. **Clasificador DistilBERT+LoRA** sobre el split de test del subconjunto
   estratificado de Bitext (449 ejemplos): `models/intent-classifier-lora/training_metrics.json`.
2. **Pipeline completo en modo fallback determinista (sin LLM)** sobre los 108
   tickets sintéticos: `eval/runs/2026-09-29-offline_fallback.json`.

## 1. Los diez intents donde el clasificador fine-tuneado falla más

F1 por clase ordenado de peor a mejor (macro-F1 global 0.9864; 18 de 27 clases
con F1 = 1.0). Gráfico: `classifier_per_class_f1.svg`.

| # | Intent | F1 | Hipótesis de la causa |
|---|---|---|---|
| 1 | `delete_account` | 0.929 | Comparte vocabulario con `edit_account` y `switch_account` ("my account", "change"); Bitext contiene variantes tipo "I don't want my account anymore" que se parecen a un downgrade. |
| 2 | `track_refund` | 0.941 | Se confunde con `get_refund` y `check_refund_policy`: las tres hablan de "refund"; la diferencia (pedir vs. consultar estado vs. consultar política) está en verbos débiles. |
| 3 | `get_refund` | 0.952 | Simétrico al anterior; además "money back" aparece en `track_refund`. |
| 4 | `review` | 0.952 | Textos cortos ("I want to leave feedback") con poco contexto; "rate" también aparece en preguntas de precios de envío. |
| 5 | `edit_account` | 0.963 | Frontera difusa con `switch_account` (cambiar tipo de cuenta vs. editar datos). |
| 6 | `contact_human_agent` | 0.971 | Confusión con `contact_customer_service`: "talk to someone" vs. "how do I contact support". |
| 7 | `newsletter_subscription` | 0.973 | "unsubscribe" aparece también en cancelaciones de servicio; con 166 ejemplos por clase el modelo ve pocas variantes. |
| 8 | `contact_customer_service` | 0.974 | Espejo del #6. |
| 9 | `switch_account` | 0.976 | Espejo del #5. |
| 10 | `cancel_order` | 1.000 | Ya no hay error: las 17 clases restantes tienen F1 = 1.0 en este split. |

Patrón: los errores están en **pares o tríos semánticamente vecinos** (refund ×3,
account ×3, contact ×2). No son errores de tokenización ni de longitud; son
fronteras de etiqueta que el propio dataset dibuja fino. Dos acciones
razonables: (a) reentrenar sobre el split completo de 2.687 ejemplos de test y
el 80 % de entrenamiento (21.497 filas) para que las clases vecinas tengan más
contraste, y (b) guardar la matriz de confusión completa (el script de
entrenamiento ya la escribe en `confusion_matrix.json` en su próxima corrida;
`scripts/plot_classifier_metrics.py` la dibuja).

Limitación del dato: 449 ejemplos de test son ~17 por clase; un solo error
mueve el F1 de una clase ~0.06. Los rankings 5-9 están dentro de ese ruido.

## 2. Fallos del pipeline en modo fallback (108 tickets)

Este modo **no es el sistema**: es el clasificador por palabras clave, la
heurística de prioridad, la plantilla de respuesta y el encoder hash. Se
analiza porque es lo único medible sin llave y porque expone qué parte de la
lógica de enrutamiento depende de señales que el fallback no produce.

### 2.1 Intent (7 errores de 108, macro-F1 0.948)

| Caso | Esperado | Obtenido | Causa |
|---|---|---|---|
| q-003 | `get_refund` | `check_refund_policy` | "Can I get a refund?" dispara la regla de política antes que la de solicitud. |
| q-029, q-030 | `get_invoice` | `check_invoice` | "invoice for" está en la regla de *check*; el orden de reglas prioriza la más específica y aquí es la equivocada. |
| q-033, q-036 | `payment_issue` | `unknown` | "rejected" y "declined ... bank" no están en la lista de palabras clave. |
| q-056 | `create_account` | `unknown` | "create an account" no coincide con "create account" (artículo). |
| q-059 | `change_shipping_address` | `unknown` | "change the delivery address for" no está cubierto por las frases exactas. |

Todos son límites de un router léxico: sinónimos y artículos. No se "arreglan"
añadiendo palabras (sería sobreajustar al eval set); se resuelven con el
clasificador fine-tuneado, que es la ruta de producción.

### 2.2 Decisión (55 desacuerdos de 108, 49.1 % de acuerdo)

| Esperado → obtenido | Casos | Causa |
|---|---|---|
| `auto_resolve` → `suggest` | 31 | Score máximo alcanzable ≈ 0.5·0.7 + 0.4·sim + 0.1; con similitud léxica media de 0.51 el score no llega a 0.85. **Sin LLM ni encoder semántico, nada se auto-resuelve** (auto-resolve rate 0 %). |
| `auto_resolve` → `escalate` | 7 | Similitud hash < 0.3 en consultas sin solapamiento léxico con las 15 semillas. |
| `suggest` → `escalate` | 6 | Misma causa: el término de similitud arrastra el score bajo 0.60. |
| `escalate` → `suggest` | 5 | `delete_account` y el cobro doble no tienen regla dura; la heurística de prioridad no detecta la gravedad ("charged twice" no está en las palabras urgentes). |

La ablación confirma la lectura: quitar el término de similitud sube el acuerdo
de 49.1 % a 54.6 % y baja las escalaciones falsas de 17.9 % a 4.2 %. **Con un
encoder léxico la similitud es ruido; con `all-mpnet-base-v2` es la señal que
justifica auto-resolver.** Por eso el umbral no se toca en función de esta
corrida.

### 2.3 Prioridad (15 fallos de 108, 86.1 % dentro de rango)

Dos familias: (a) urgencias que la heurística no ve (q-034 cobro doble, q-049
GDPR, q-057-059 cambio de dirección antes del envío) porque no contienen
palabras de la lista; (b) falsos positivos de urgencia por "today" en
preguntas informativas (q-061, q-074) y por "downgrade ... losing" (q-103).
Ambas son inherentes a una heurística de palabras; el analizador con Claude
recibe las guías P0-P3 completas en el system prompt.

## 3. Qué falta para cerrar este análisis

- Corrida `hybrid` y `zero_shot_claude` con `ANTHROPIC_API_KEY` para tener los
  errores del sistema real y no del fallback.
- Muestra de 200 conversaciones de Twitter (Kaggle) para medir sobre texto real
  con typos, emojis y varios idiomas, donde el clasificador entrenado en
  plantillas de Bitext previsiblemente pierde más.
- Matriz de confusión completa del clasificador sobre el split de test completo.
