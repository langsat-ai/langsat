# 17 · Compare — the fifteen bar and column forms, and which question each one answers

**Use case.** Most charts anyone builds are a bar chart. Langsat offers fifteen shapes of one, and they are not interchangeable: a stacked column answers "how big is the whole, and what is it made of", a clustered column answers "which of these is biggest", and a 100% stacked column answers "what is the MIX" while deliberately throwing away the size. Picking the wrong one draws a chart that is readable and answers a question nobody asked.

**What you will learn**
1. `column` and `bar` — the same trace, one with `orientation: "h"`, and when horizontal wins
2. Stacked, clustered and 100% stacked, in both orientations — six charts, one `series` well
3. `top_n` on an axis with 6,469 values, and the three charts where the parameter does not exist
4. `dot` — a bar chart with the bars removed, for when the baseline is not zero
5. `histogram(edges=…)` — you give the boundaries, the server mints the labels
6. `facet` — small multiples, one panel per value
7. Draw the whole tab as one image with `viz.draw_tab`

**What this costs.** nothing. Fifteen charts, one table scan each, no model.

> Every cell below ran for real against `api.langsat.ai` — the outputs are what the API returned. Re-running is safe:
> projects are found by name and reused, and a finished model is not retrained.

Notebook: [`17_compare.ipynb`](17_compare.ipynb) · outputs are from a real run on 2026-09-30.

## Result

| | |
|---|---|
| task | the bar family · 15 forms of one trace |
| model / lane | — |
| charts_built | 9 |
| brands_in_data | 426 |
| price_buckets | 7 |
| credits charged (this run) | 0 |
| project | `5c40e337-9753-4618-a7b3-574419f93c1e` |

Full numbers: [`results/metrics.json`](results/metrics.json).

## Run it

```bash
pip install -r ../../../requirements.txt && langsat login
jupyter lab 17_compare.ipynb
```
