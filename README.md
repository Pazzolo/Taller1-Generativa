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
├── data/                     # cases.json: 10 casos oficiales + 5 de debug (campo "split")
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
uv run scripts/aggregate_results.py                                      # outputs/tables/part1.csv y part2a.csv
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
- [ ] Milestone 6 — Parte 3: prompting estructurado
- [ ] Milestone 7 — Parte 0: GPT-2 base
- [ ] Milestone 8 — Parte 4.a: reasoning effort
- [ ] Milestone 9 — Parte 4.b: casos contaminados
- [ ] Milestone 10 — agregación, gráficas y reporte

## Convenciones de commits

Un commit por cambio lógico, con formato `tipo(alcance): resumen en imperativo`.

- Tipos: `feat`, `fix`, `data`, `exp`, `test`, `refactor`, `docs`, `chore`.
- Alcances: `part0`, `part1`, `part2a`, `part2b`, `part3`, `part4a`, `part4b`, `verifier`, `runner`, `pricing`, `metrics`, `report`.
- Agregar archivos por nombre, revisar `git diff --staged` antes de commitear y nunca incluir claves.
- Código y resultados de barridos caros van en commits separados.
