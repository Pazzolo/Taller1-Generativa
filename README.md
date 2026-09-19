# Taller 1 — Foundation Models

Taller 01 de Inteligencia Artificial Generativa (USFQ). Un único problema, un único dataset, un único verificador y una única interfaz de modelo, con experimentos controlados que cambian solo la variable que se quiere medir.

## Problema

Clasificar un ticket de soporte en exactamente una categoría: `billing`, `technical` o `account`. La salida esperada es JSON:

```json
{"category": "billing"}
```

La respuesta se verifica de forma determinista (sin LLM como juez) contra el valor esperado de cada caso.

## Estructura

```text
.
├── data/                     # cases.json (10 oficiales + 5 de debug) y contaminated_cases.json (3 casos de la Parte 4.b)
├── src/
│   ├── config.py             # rutas, SEED, categorías permitidas
│   ├── runner.py             # run_case(): prompt -> modelo -> verificador -> costo -> JSONL
│   ├── results.py            # append_result() / read_results() sobre results.jsonl
│   ├── schemas.py            # Case, Expected, CATEGORY_SCHEMA, load_cases()
│   ├── prompts.py            # build_base_prompt() y variantes de prompting
│   ├── verifier.py           # verify_prediction(): parseo + exactitud
│   ├── metrics.py            # métricas centralizadas
│   ├── pricing.py            # precios con fecha de verificación, cost_usd()
│   ├── aggregation.py        # tabla de la Parte 1 desde results.jsonl
│   ├── exposure.py           # Parte 2.a: valores extremos, clasificación y tabla de la matriz
│   ├── sweeps.py             # Parte 2.b: celdas de los barridos, tablas y comprobación top_k=1 vs greedy
│   ├── planning.py           # llamadas planificadas, claves de reanudación y estimación de costo (2.b y 3)
│   ├── distribution.py       # Parte 0: softmax con temperatura, entropía, núcleo top-p, conjuntos top-k / top-p
│   ├── plots_part0.py        # gráficas de la Parte 0 (matplotlib)
│   ├── reasoning.py          # Parte 4.a: tablas por nivel de esfuerzo, saltos entre niveles y control
│   ├── plots_part4.py        # gráficas de la Parte 4
│   ├── contamination.py      # Parte 4.b: tablas por caso y nivel, y por llamada
│   ├── models/               # ModelRunner (base), MockRunner, runners OpenAI / Anthropic / Ollama, registry
│   └── experiments/          # part0 ... part4b, uno por parte del taller
├── scripts/
│   ├── run_all.py            # --part <0|1|2a|2b|3|4a|4b> | --all
│   ├── aggregate_results.py  # tablas desde results.jsonl
│   └── generate_plots.py     # gráficas desde results.jsonl
├── outputs/
│   ├── raw/                  # results.jsonl (fuente de verdad, no versionado)
│   ├── tables/
│   └── plots/
├── report/                   # report.md y figuras
├── tests/                    # pytest: verificador, dataset, pipeline con mock, precios
└── notebooks/
```

## Puesta en marcha

Requiere Python 3.11 o 3.12. El entorno se gestiona con [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.11
uv pip install -r requirements.txt
cp .env.example .env   # completar las claves localmente
```

`uv venv` crea `.venv/`; no hace falta activarlo si los comandos se ejecutan con `uv run`. Si hay otro entorno virtual activo (`echo $VIRTUAL_ENV` no vacío), ejecutar `deactivate` antes: sin `pyproject.toml`, uv usa el entorno activo en lugar de `.venv/`.

Para probar solo el Milestone 1 basta una instalación ligera (evita `torch` y `transformers`): `uv pip install pydantic python-dotenv pytest`. Para agregar una dependencia: `uv pip install <paquete>` y anotarla en `requirements.txt`.

Nunca se guardan API keys en código, notebooks, resultados ni en el repositorio. `.env` está en `.gitignore`.

## Uso

```bash
uv run scripts/run_all.py --part 1     # una parte
uv run scripts/run_all.py --all        # todas
uv run -m src.experiments.part1        # equivalente a la primera
```

Los experimentos escriben una fila por llamada en `outputs/raw/results.jsonl`. Las tablas y gráficas se derivan de ese archivo, nunca se copian a mano.

`part1` acepta `--model` y `--limit`:

```bash
uv run -m src.experiments.part1                                        # mock: sin red ni costo
uv run -m src.experiments.part1 --model propietario_economico --limit 1  # 1 caso real (gpt-4o-mini)
uv run -m src.experiments.part1 --model propietario_economico            # los 10 casos oficiales
uv run -m src.experiments.part1 --model propietario_grande               # gpt-5.5 (OpenAI)
uv run -m src.experiments.part1 --model open_weight_pequeno              # qwen3:1.7b (Ollama local)
uv run scripts/aggregate_results.py                                      # todas las tablas en outputs/tables/
uv run scripts/generate_plots.py                                         # gráficas de la Parte 4 en outputs/plots/
```

Resultados de la Parte 1 (10 casos oficiales, `outputs/tables/part1.csv`):

| Modelo | Exactitud | Latencia media | Tokens de salida (media) | Costo total |
|---|---|---|---|---|
| `gpt-5.5` | 10/10 | 0.97 s | 23.1 (9-10 de razonamiento) | $0.0097 |
| `gpt-4o-mini` | 10/10 | 0.70 s | 6.0 | $0.00012 |
| `qwen3:1.7b` (local) | 10/10 | 0.15 s | 7.0 | $0 |

Los tres aciertan todo con JSON válido, así que este problema no discrimina entre modelos (efecto techo). La latencia de `qwen3:1.7b` no incluye red y su costo 0 no cuenta el hardware, por lo que no es comparable directamente con las de pago.

Modelos de la Parte 1:

| Clave | Modelo | Requisito |
|---|---|---|
| `propietario_grande` | `gpt-5.5` | `OPENAI_API_KEY` en `.env` |
| `propietario_economico` | `gpt-4o-mini` | `OPENAI_API_KEY` en `.env` |
| `open_weight_pequeno` | `qwen3:1.7b` | Ollama en ejecución (`OLLAMA_HOST`) y `ollama pull qwen3:1.7b` |

`propietario_grande` usa `gpt-5.5` en lugar del `claude-opus-4-8` de la tabla del curso (no había clave de Anthropic). Es un modelo de razonamiento: sus tokens de razonamiento se facturan como salida. Su precio ($5 / $30 por 1M, verificado el 2026-09-19) viene de la página de precios de OpenAI, no de la tabla del curso.

Para `qwen3:1.7b` el razonamiento se desactiva (`think: false`) para que compare como un modelo instruct normal. La latencia de la primera llamada incluye la carga del modelo en memoria (`load_duration` queda en `raw_response`).

La tabla toma la última fila por modelo y caso, así una corrida de humo previa no cuenta doble. Las llamadas fallidas cuentan como incorrectas en la exactitud pero no en latencia ni tokens. La columna `qualitative_notes` sale de `report/part1_notes.json` (una nota por modelo, escrita a mano), para que no se pierda al regenerar la tabla.

Con `--model mock` los resultados van a `outputs/raw/mock_results.jsonl`; con un modelo real, a `outputs/raw/results.jsonl`. Ambos están ignorados por git. Las llamadas reales requieren `OPENAI_API_KEY` en `.env`; empezar siempre con `--limit 1`.

Cada llamada, exitosa o fallida, agrega una línea al JSONL (nunca se sobrescribe). Las fallas de API se registran con `status: "error"`, el mensaje literal y el `http_status`.

### Parte 2.a — matriz de exposición de parámetros

```bash
uv run -m src.experiments.part2a --dry-run                               # cuántas llamadas, sin llamar a la API
uv run -m src.experiments.part2a --models propietario_grande propietario_economico open_weight_pequeno
```

Para cada modelo y parámetro se envía un valor bajo y uno alto (`temperature` 0.0 / 2.0, `top_p` 0.01 / 1.0, `top_k` 1 / 100), 5 corridas cada uno, con un prompt abierto (`probe`) donde el muestreo sí cambia la salida. Un HTTP 200 no basta para decir que el parámetro actúa; el estado se decide así:

- `rejected`: la API responde 400 o 422 en algún valor. Se guarda el mensaje literal.
- `accepted_and_acts`: sin errores y hay más salidas distintas con el valor alto que con el bajo.
- `accepted_and_does_not_act`: sin errores y la diversidad no aumenta.
- `inconclusive`: fallos que no son un rechazo del parámetro (sin conexión, 429, 5xx). No cuenta como rechazo.

El criterio se apoya en solo 5 corridas por ajuste, así que es una evidencia, no una prueba. La columna `declared` es lo que documenta el proveedor; para `gpt-5.5` no encontré esa información en su página de modelo (`no documentado`). Si un ajuste falla, no se repite: repetir un rechazo no aporta información.

Resultados actuales (2026-09-19, `outputs/tables/part2a.csv`):

| Modelo | temperature | top_p | top_k |
|---|---|---|---|
| `gpt-4o-mini` | actúa (1 vs 5 salidas distintas) | actúa (1 vs 5) | rechazado, 400 |
| `gpt-5.5` | rechazado, 400 | rechazado, 400 | rechazado, 400 |
| `qwen3:1.7b` | actúa (1 vs 5) | actúa (1 vs 4) | actúa (1 vs 3) |

`gpt-5.5` acepta solo los valores por defecto (`temperature=1`, `top_p=1.0`): los valores no predeterminados fallan con `Only the default (1) value is supported` o `not supported with this model`.

### Parte 2.b — barrido de decoding

```bash
uv run -m src.experiments.part2b --dry-run      # llamadas planificadas, ya hechas y costo estimado; no llama a la API
uv run -m src.experiments.part2b                # ambos barridos
uv run -m src.experiments.part2b --sweep topk   # solo top_k (qwen3:1.7b, local)
uv run -m src.experiments.part2b --limit-cases 1 --runs 1   # piloto
```

- **temperature × top_p** sobre `gpt-4o-mini`: temperature {0, 0.3, 0.7, 1.0, 1.5} × top_p {0.5, 0.9, 1.0} = 15 celdas × 10 casos × 5 corridas = 750 llamadas. Se eligió `gpt-4o-mini` porque la Parte 2.a mostró que acepta ambos parámetros y que actúan.
- **top_k** sobre `qwen3:1.7b` (la API de OpenAI lo rechaza): top_k {1, 5, 40} con temperature 0.7, más un baseline greedy (temperature 0) = 4 celdas × 10 casos × 5 corridas = 200 llamadas.
- Se limita la salida a 128 tokens para cortar una posible degeneración a temperatura alta; la respuesta correcta ocupa unos 7. Es una condición del experimento, así que los tokens de salida medios están acotados por ese tope.
- Es reanudable: una llamada ya registrada como `ok` con los mismos parámetros se omite, así que interrumpir y repetir no vuelve a cobrar. Las llamadas con error se reintentan.
- **Definiciones** (`src/metrics.py`): exactitud = aciertos / todas las llamadas de la celda (los errores cuentan como fallo); estabilidad = por caso, parte de las corridas que coinciden con la respuesta más frecuente, promediada entre los 10 casos (una salida inválida cuenta como una sola respuesta; los errores se excluyen). `greedy_match` = parte de casos donde las 5 corridas con top_k=1 dan exactamente la salida del baseline greedy.

Resultados (2026-09-19, `outputs/tables/part2b_decoding.csv` y `part2b_topk.csv`; 950 llamadas, 0 errores, costo $0.0091):

- **temperature × top_p, `gpt-4o-mini`:** exactitud 1.00, estabilidad 1.00 y `parse_rate` 1.00 en las 15 celdas. La salida de cada caso fue idéntica en todas las corridas y celdas. Única excepción: con temperature 1.5 y top_p 1.0, una salida salió como JSON compacto sin espacio (`{"category":"billing"}`, 5 tokens en vez de 6), con la misma categoría.
- **top_k, `qwen3:1.7b`:** exactitud 1.00 y estabilidad 1.00 con el baseline greedy, top_k=1 y top_k=5. Con top_k=40, exactitud y estabilidad 0.98: un fallo en 50 llamadas (`case_10`, corrida 5: respondió `technical` en vez de `billing`). `greedy_match` con top_k=1 es 1.00: los 10 casos reproducen exactamente la salida greedy.

Lectura: en esta tarea, las palancas de muestreo no producen ningún efecto medible en exactitud ni estabilidad. La Parte 2.a ya mostró que los parámetros sí actúan sobre texto libre; aquí la decisión es un solo token de categoría con masa de probabilidad muy concentrada, así que ni siquiera temperature 1.5 lo altera. Las dos desviaciones son eventos únicos en los ajustes más permisivos: coinciden con el sentido esperado, pero con n=50 por celda no bastan para hablar de tendencia.

### Parte 3 — prompting estructurado

```bash
uv run -m src.experiments.part3 --dry-run     # llamadas planificadas y costo estimado (sin llamar a la API)
uv run -m src.experiments.part3               # 4 variantes x 10 casos x 5 corridas = 200 llamadas, gpt-4o-mini
```

Mismo modelo (`gpt-4o-mini`), mismos 10 casos oficiales y misma tarea; solo cambia el prompt:

| Variante | Qué cambia |
|---|---|
| `zero_shot` | Solo instrucciones (es el prompt base de la Parte 1). |
| `few_shot` | Instrucciones + 3 ejemplos propios, uno por categoría. Ninguno sale de `cases.json` (hay un test que lo comprueba). |
| `cot` | Pide razonar paso a paso y dar la respuesta final en la última línea. |
| `structured` | El formato lo impone la API con un esquema JSON estricto (`response_format`), no el texto del prompt. |

- **CoT:** se verifica solo la respuesta final (el último objeto JSON del texto, guardado como `final_answer`); `raw_output` conserva el razonamiento completo. Solo CoT tiene esa extracción: en las demás variantes una respuesta con texto de más falla el formato. `gpt-4o-mini` no expone tokens de razonamiento, así que el razonamiento visible cuenta como salida.
- **Formato garantizado vs. corrección semántica:** `format_compliance` mide JSON válido con categoría dentro del enum; `accuracy` mide que sea la correcta. Son independientes: un esquema estricto puede devolver un JSON válido con la categoría equivocada (hay un test que lo cubre).
- Los costos de entrada y de salida se reportan por separado. Todas las llamadas usan un tope de 512 tokens de salida (la respuesta CoT más larga fue de 265, así que no hubo truncamiento).

Resultados (2026-09-19, `outputs/tables/part3.csv`; 200 llamadas, 0 errores):

| Variante | Exactitud | Formato válido | Tokens de entrada | Tokens de salida | Costo entrada | Costo salida | Costo total |
|---|---|---|---|---|---|---|---|
| `zero_shot` | 1.00 | 1.00 | 57.1 | 6.0 | $0.00043 | $0.00018 | $0.00061 (1.0x) |
| `few_shot` | 1.00 | 1.00 | 133.1 | 6.0 | $0.00100 | $0.00018 | $0.00118 (1.9x) |
| `cot` | 1.00 | 1.00 | 82.1 | 133.6 | $0.00062 | $0.00401 | $0.00462 (7.6x) |
| `structured` | 1.00 | 1.00 | 80.1 | 5.0 | $0.00060 | $0.00015 | $0.00075 (1.2x) |

Lectura: las cuatro variantes aciertan todo, así que en esta tarea no hay diferencia de exactitud y todo el contraste es de costo. Few-shot cuesta 1.9x por sumar ~76 tokens de entrada; CoT cuesta 7.6x porque produce ~22 veces más tokens de salida (que se cobran a precio de salida, 4x el de entrada) sin mejorar la exactitud; `structured` es la más barata en salida (5 tokens, JSON compacto) pero gasta ~23 tokens más de entrada que `zero_shot` pese a tener un prompt más corto, lo que sugiere que el esquema cuenta como entrada. Como `zero_shot` ya cumplía el formato en el 100 % de las llamadas, la garantía de `structured` no se aprovechó aquí. Son resultados de una sola tarea fácil con un modelo que ya la resuelve; no permiten generalizar a tareas más difíciles.

### Parte 0 — GPT-2 base (local, sin API)

```bash
uv run -m src.experiments.part0            # 0.a, 0.b y 0.c
uv run -m src.experiments.part0 --only a   # solo una subparte (a, b o c)
```

Usa `openai-community/gpt2` en CPU (no una variante instruct). La primera ejecución descarga el modelo (~550 MB); requiere `torch`, `transformers` y `matplotlib`. Es determinista (`SEED = 42`), y las salidas de esta parte (`outputs/raw/part0*.json`) sí se versionan: son pequeñas y se reproducen sin costo.

**Elección de los prefijos (medida, no supuesta).** El plan sugiere `The capital of France is` como prefijo de alta confianza, pero GPT-2 no lo es: a T=1 su token más probable es `␣the` con 0.085 (`␣Paris` es el 5.º con 0.032) y su entropía es 8.65 bits, casi la de un prefijo incierto (9.33 bits). Por eso el script mide la entropía a T=1 de una lista fija de candidatos (`outputs/tables/part0_prefix_scan.csv`) y elige el de menor entropía: `Thank you very` (0.10 bits, `␣much` con 0.992). El prefijo de menor confianza es el del dominio, `A customer support ticket about an unexpected`.

**0.a — temperatura** (`outputs/tables/part0_temperature.csv`, `outputs/raw/part0a.json`, figuras `part0_temperature_high_confidence.png`, `part0_temperature_low_confidence.png` y `part0_entropy_vs_temperature.png`):

| Prefijo | T=0.1 | T=0.7 | T=1.0 | T=1.5 | T=2.0 |
|---|---|---|---|---|---|
| `Thank you very`: entropía (bits) | 0.00 | 0.01 | 0.10 | 2.15 | 9.77 |
| `Thank you very`: núcleo top-p 0.9 | 1 | 1 | 1 | 12 | 14 537 |
| `A customer support ticket…`: entropía (bits) | 0.95 | 6.39 | 9.33 | 12.24 | 13.54 |
| `A customer support ticket…`: núcleo top-p 0.9 | 2 | 154 | 1 550 | 8 364 | 15 726 |

La entropía y el núcleo crecen con la temperatura en ambos prefijos. El prefijo seguro se mantiene casi determinista hasta T=1 y solo se abre en T≥1.5; el incierto ya es amplio desde T=0.7. A T=2 ambos se acercan al máximo (el vocabulario tiene 50 257 tokens, log2 ≈ 15.6 bits).

**0.b — las tres palancas** (`outputs/raw/part0b.json`; prefijo `A customer support ticket about an unexpected`, 40 tokens nuevos):

1. `do_sample=False` con temperature 0.2 y 1.5: **salida idéntica** entre sí y con la decodificación greedy. La temperatura se ignora, y la librería lo avisa (`The following generation flags are not valid and may be ignored: ['temperature']`, emitido una sola vez).
2. `top_k=1` con `do_sample=True`, 5 corridas con semillas 42-46: las 5 salidas son idénticas entre sí e idénticas a la de greedy.
3. top-k=5 vs top-p=0.9 sobre la misma distribución (T=1): los conjuntos difieren en ambos prefijos, en sentido contrario. En `Thank you very` top-k conserva 5 tokens y top-p solo 1; en `A customer support ticket…` top-k conserva 5 y top-p 1 550. Es decir, top-k corta por cantidad fija y top-p se adapta a la confianza del modelo. Figura `part0_topk_vs_topp.png` (escala logarítmica, porque con un token dominante el resto sería invisible) y su tabla equivalente `part0_topk_vs_topp.csv`.
4. Degeneración: 100 tokens en greedy entran en un bucle (`…about an unexpected problem.` repetido). La salida cruda está sin editar en `part0b.json`.

Las definiciones de top-k y top-p de `src/distribution.py` se comprueban en los tests contra los `TopKLogitsWarper` y `TopPLogitsWarper` de `transformers`, que son los que aplica `generate()`.

**0.c — límite del modelo base** (`outputs/raw/part0c.json`): con el mismo prompt de clasificación de la Parte 1 (10 casos, greedy, 60 tokens), GPT-2 base obtiene **0/10** respuestas parseables. No sigue la instrucción, solo continúa el texto: en 4 de 10 casos repite el ticket tal cual, en el caso 3 la palabra «account» aparece solo dentro de una frase repetida, y en el caso 9 escribe un comando `curl` que copia el formato `{"category": ...}` del prompt con un valor sin sentido. Un modelo base predice la continuación más probable; no está alineado para obedecer instrucciones, a diferencia de los modelos de la Parte 1.

Las figuras usan una rampa azul ordinal para la temperatura y las ranuras categóricas 1-3 más gris para top-k / top-p, validadas con el validador de paletas antes de dibujar; la ranura aqua queda bajo 3:1 de contraste sobre el fondo claro, por eso cada figura va acompañada de su tabla CSV.

### Parte 4.a — esfuerzo de razonamiento

```bash
uv run -m src.experiments.part4a --limit-cases 1 --runs 1   # piloto: 1 caso por nivel (incluye max)
uv run -m src.experiments.part4a                            # 10 casos x 5 niveles x 3 corridas
uv run -m src.experiments.part4a --control                  # acertijo de control, 3 corridas por nivel
uv run scripts/aggregate_results.py && uv run scripts/generate_plots.py
```

Modelo `openai_razonamiento` (`gpt-5.6-luna`, $0.20 / $1.20 por 1M tokens, verificado el 2026-09-18 en la tabla del curso). Los tokens de razonamiento se facturan a precio de salida y se guardan aparte (`reasoning_tokens`); `visible_output_tokens` = salida total − razonamiento. Tope de seguridad de 16 000 tokens de salida por llamada.

**Hallazgos del piloto (registrados en `results.jsonl`):**

- **`max` no existe para este modelo.** La tabla del curso lo lista, pero la API responde 400: `Unsupported value: 'reasoning_effort' does not support 'max' with this model. Supported values are: 'none', 'low', 'medium', 'high', and 'xhigh'.` El barrido usa los cinco niveles aceptados.
- **El piloto sospechoso:** con 1 caso, los cinco niveles dieron 0 tokens de razonamiento. Antes de barrer se comprobó que el parámetro sí actúa con un acertijo de control (abajo).

**Resultados** (30 llamadas por nivel; `outputs/tables/part4a.csv`, `part4a_deltas.csv`, `part4a_control.csv`; figuras `part4_accuracy_vs_reasoning_tokens.png` y `part4_cost_vs_accuracy.png`):

| Esfuerzo | Exactitud | Razonamiento (media) | Salida total (media) | Costo por llamada |
|---|---|---|---|---|
| `none` | 1.00 | 0 | 8.0 | $0.0000208 |
| `low` | 1.00 | 0 | 8.0 | $0.0000208 |
| `medium` | 1.00 | 0 | 8.0 | $0.0000208 |
| `high` | 1.00 | 0.37 | 8.6 | $0.0000215 |
| `xhigh` | 1.00 | 1.57 | 10.4 | $0.0000237 |
| `max` | — | — | — | rechazado (400) |

- **Sin ganancia de exactitud:** 1.00 en todos los niveles, así que el costo por punto de exactitud no está definido (el aumento de exactitud es 0 en cada salto). Lo único que cambia es el costo: `xhigh` cuesta 14 % más por llamada que `none`.
- **Casi no razona con tickets:** solo 5 de 150 llamadas usaron tokens de razonamiento, todas en `high`/`xhigh` y solo en los casos 4 y 10 (los dos tickets con más matices de cobro). Con `none`, `low` y `medium` el modelo no razonó nunca.
- **Control:** con un acertijo (bate y pelota, 3 corridas por nivel) el dial sí actúa: 0 tokens de razonamiento en `none` y 26 / 36 / 20 / 49 en `low` / `medium` / `high` / `xhigh`. No es monótono (`high` < `medium`), pero con 3 corridas por nivel no se puede afirmar más que la tendencia general. El acertijo se resolvió bien incluso sin razonamiento (`none`), así que tampoco ahí el razonamiento mejoró la exactitud.

Lectura: el parámetro funciona, pero esta tarea es demasiado fácil para que el razonamiento se note en la exactitud; el modelo decide no razonar y el nivel solo encarece un poco. No se debe concluir que razonar no sirva en general, solo que aquí no aporta. La Parte 4.b prueba justamente casos con distracciones. Costo total de la Parte 4.a: $0.004.

### Parte 4.b — casos contaminados

```bash
uv run -m src.experiments.part4b --dry-run          # llamadas planificadas (sin llamar a la API)
uv run -m src.experiments.part4b                    # 3 casos x {low, high} x 10 corridas = 60 llamadas
uv run -m src.experiments.part4b --levels xhigh     # comprobación adicional con el esfuerzo máximo aceptado
```

Tres casos fáciles, escritos a mano en `data/contaminated_cases.json`, cada uno con una trampa distinta. La etiqueta correcta es inequívoca en los tres: la dificultad está en la trampa, no en el criterio (un test comprueba que cada ticket trae, además de la señal correcta, señales de una categoría equivocada).

| Caso | Trampa | Correcta | Qué la contamina |
|---|---|---|---|
| `contaminated_01` | Distracción | `account` | Una anécdota larga sobre un terminal de pago que se congela y un software de facturas; lo que se pide es cambiar el correo del perfil. |
| `contaminated_02` | Marco engañoso | `billing` | El usuario cree que es un fallo de los servidores y escribe a soporte técnico, pero lo que pide es el reembolso de un cobro duplicado. |
| `contaminated_03` | Correlación espuria | `technical` | Menciona «premium», «pago» y «factura», pero el problema es que la aplicación se cierra. |

Modelo `gpt-5.6-luna`, mismo prompt base que la Parte 1, 10 corridas por celda para ver variación. Los niveles `low` y `high` son los que pide el plan; `xhigh` es una comprobación adicional. Tablas: `outputs/tables/part4b.csv` (por caso y nivel) y `part4b_runs.csv` (una fila por llamada, con predicción, esperado, acierto y tokens de razonamiento).

| Caso | low: aciertos / razonamiento (media) | high | xhigh (extra) |
|---|---|---|---|
| Distracción | 10/10, 0 | 10/10, 1.6 | 10/10, 4.3 |
| Marco engañoso | 10/10, 0 | 10/10, 4.5 | 10/10, 7.8 |
| Correlación espuria | 10/10, 0 | 10/10, 9.2 | 10/10, 22.2 |

- **Las trampas no funcionaron:** 90 de 90 llamadas correctas, con predicciones idénticas en todas las corridas. No se observa escalado inverso.
- **El razonamiento sí crece con el esfuerzo** en los tres casos, pero es minúsculo: 28 tokens como máximo en una llamada. En cada nivel el caso con más tokens fue el de la correlación espuria y el de menos el de la distracción (por ejemplo, en `high`: 9.2, 4.5 y 1.6). Con 10 corridas por celda y esas cifras es un indicio, no una conclusión.
- **Contraste con Gema et al. (2025)**, *Inverse Scaling in Test-Time Compute*: el artículo reporta que más razonamiento puede empeorar la exactitud: los modelos Claude se distraen cada vez más con información irrelevante, los modelos de la serie o de OpenAI resisten los distractores pero se sobreajustan al marco del problema, y los modelos pasan de priors razonables a correlaciones espurias. Este experimento no lo reproduce, pero tampoco lo contradice: sus tareas son otras (conteo con distractores, regresión con rasgos espurios, deducción, riesgos de IA), más difíciles, con presupuestos de razonamiento mucho más largos y otros modelos, mientras que aquí hay tres casos fáciles con exactitud del 100 % y decenas de tokens de razonamiento, así que ni siquiera se llega al régimen que estudian. Para detectar un efecto inverso harían falta casos donde el modelo falle sin trampas o con más razonamiento.
- **Tres casos no demuestran una ley general**, ni a favor ni en contra.

Costo total de la Parte 4.b: $0.003.

### Pruebas

```bash
uv run pytest
```

Cubren el verificador (correcto, incorrecto, fuera del enum, no parseable, JSON que no es objeto), las reglas del dataset (10 casos oficiales, ids únicos, categorías válidas y cubiertas), el pipeline de `part1` con un mock y el cálculo de costos.

El dataset oficial queda congelado: los 10 casos con `"split": "official"` no cambian entre modelos, temperaturas ni variantes de prompt. Los de `"split": "debug"` son solo para pruebas internas.

## Estado

Milestones 1 a 4 completos. Los demás experimentos aún no están implementados: cada `partX.py` (salvo `part1`) lanza `NotImplementedError` con el milestone que le corresponde.

- [x] Estructura del proyecto, `config`, `schemas`, `verifier`, `pricing`, `build_base_prompt`, interfaz `ModelRunner`
- [x] Milestone 1 — `data/cases.json` + verificador + pruebas (41 tests)
- [x] Milestone 2 — un modelo real (gpt-4o-mini) conectado y `results.jsonl` verificado (62 tests)
- [x] Milestone 3 — Parte 1: comparación de tres modelos (gpt-5.5, gpt-4o-mini, qwen3:1.7b)
- [x] Milestone 4 — Parte 2.a: matriz de exposición de parámetros (3 modelos × 3 parámetros)
- [x] Milestone 5 — Parte 2.b: barrido de temperature × top-p y de top-k (950 llamadas)
- [x] Milestone 6 — Parte 3: prompting estructurado (4 variantes, 200 llamadas)
- [x] Milestone 7 — Parte 0: GPT-2 base (0.a, 0.b y 0.c, 4 figuras)
- [x] Milestone 8 — Parte 4.a: reasoning effort (5 niveles + control, 165 llamadas)
- [x] Milestone 9 — Parte 4.b: casos contaminados (3 casos × 3 niveles × 10 corridas)
- [ ] Milestone 10 — agregación, gráficas y reporte

## Convenciones de commits

Un commit por cambio lógico, con formato `tipo(alcance): resumen en imperativo`.

- Tipos: `feat`, `fix`, `data`, `exp`, `test`, `refactor`, `docs`, `chore`.
- Alcances: `part0`, `part1`, `part2a`, `part2b`, `part3`, `part4a`, `part4b`, `verifier`, `runner`, `pricing`, `metrics`, `report`.
- Agregar archivos por nombre, revisar `git diff --staged` antes de commitear y nunca incluir claves.
- Código y resultados de barridos caros van en commits separados.
