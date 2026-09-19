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
├── data/                     # cases.json (dataset verificable, >= 10 casos)
├── src/
│   ├── config.py             # rutas, SEED, categorías permitidas
│   ├── schemas.py            # Case, Expected, CATEGORY_SCHEMA, load_cases()
│   ├── prompts.py            # build_base_prompt() y variantes de prompting
│   ├── verifier.py           # verify_prediction(): parseo + exactitud
│   ├── metrics.py            # métricas centralizadas
│   ├── pricing.py            # precios con fecha de verificación, cost_usd()
│   ├── models/               # ModelRunner (base) + runners OpenAI / Ollama / Transformers
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

`uv venv` crea `.venv/`; no hace falta activarlo si los comandos se ejecutan con `uv run`. Para agregar una dependencia: `uv pip install <paquete>` y anotarla en `requirements.txt`.

Nunca se guardan API keys en código, notebooks, resultados ni en el repositorio. `.env` está en `.gitignore`.

## Uso

```bash
uv run scripts/run_all.py --part 1     # una parte
uv run scripts/run_all.py --all        # todas
uv run -m src.experiments.part1        # equivalente a la primera
```

Los experimentos escriben una fila por llamada en `outputs/raw/results.jsonl`. Las tablas y gráficas se derivan de ese archivo, nunca se copian a mano.

## Estado

Andamiaje creado. Los experimentos aún no están implementados: cada `partX.py` lanza `NotImplementedError` con el milestone que le corresponde.

- [x] Estructura del proyecto, `config`, `schemas`, `verifier`, `pricing`, `build_base_prompt`, interfaz `ModelRunner`
- [ ] Milestone 1 — `data/cases.json` + pruebas del verificador
- [ ] Milestone 2 — un modelo real conectado y `results.jsonl` correcto
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
