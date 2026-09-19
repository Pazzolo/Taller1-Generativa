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
│   ├── models/               # ModelRunner (base), MockRunner + runners OpenAI / Ollama / Transformers
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

Requiere Python 3.11 o 3.12.

```bash
El entorno se gestiona con [uv](https://docs.astral.sh/uv/).

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
```

Con `--model mock` los resultados van a `outputs/raw/mock_results.jsonl`; con un modelo real, a `outputs/raw/results.jsonl`. Ambos están ignorados por git. Las llamadas reales requieren `OPENAI_API_KEY` en `.env`; empezar siempre con `--limit 1`.

Cada llamada, exitosa o fallida, agrega una línea al JSONL (nunca se sobrescribe). Las fallas de API se registran con `status: "error"`, el mensaje literal y el `http_status`.

### Pruebas

```bash
uv run pytest
```

Cubren el verificador (correcto, incorrecto, fuera del enum, no parseable, JSON que no es objeto), las reglas del dataset (10 casos oficiales, ids únicos, categorías válidas y cubiertas), el pipeline de `part1` con un mock y el cálculo de costos.

El dataset oficial queda congelado: los 10 casos con `"split": "official"` no cambian entre modelos, temperaturas ni variantes de prompt. Los de `"split": "debug"` son solo para pruebas internas.

## Estado

Milestones 1 y 2 completos. Los demás experimentos aún no están implementados: cada `partX.py` (salvo `part1`) lanza `NotImplementedError` con el milestone que le corresponde.

- [x] Estructura del proyecto, `config`, `schemas`, `verifier`, `pricing`, `build_base_prompt`, interfaz `ModelRunner`
- [x] Milestone 1 — `data/cases.json` + verificador + pruebas (41 tests)
- [x] Milestone 2 — un modelo real (gpt-4o-mini) conectado y `results.jsonl` verificado (62 tests)
- [ ] Milestone 3 — Parte 1: comparación de tres modelos
- [ ] Milestone 4 — Parte 2.a: matriz de exposición de parámetros
- [ ] Milestone 5 — Parte 2.b: barrido de temperature × top-p
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
