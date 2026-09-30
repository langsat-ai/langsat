# 19 · Part of a whole, and the grid — pie, donut, funnel area, funnel, treemap, heatmap, matrix, KPI, combo

**Use case.** The nine chart types that are not a bar and not a line. Each one answers a question the bar family cannot, and each one has a rule that follows from what it means: a composition has no Top N because truncating a whole is a lie about it, a funnel has an order because the order IS the funnel, a heatmap has cell caps because a grid loses whole rows rather than a tail, and a KPI has no axis and therefore no drill. With this chapter the book has covered all 24.

**What you will learn**
1. `pie` · `donut` · `funnelarea` — one additive whole, and why none of them takes a Top N
2. `funnel` — the one composition that is ordered by its measure, with no limit
3. `treemap` — 426 leaves against a 200-leaf bound, and what the engine does about it
4. `heatmap` and `matrix` — two axes and a value; `series` is the y channel, not a legend
5. `kpi` — one number over the whole filtered table, and why `charts.kpi(charts.count())` alone refuses
6. `combo` — two measures, one axis, a real second y-axis, and why a legend is refused
7. The three surfaces this server has switched off, as code you can copy

**What this costs.** nothing.

> Every cell below ran for real against `api.langsat.ai` — the outputs are what the API returned. Re-running is safe:
> projects are found by name and reused, and a finished model is not retrained.

Notebook: [`19_composition.ipynb`](19_composition.ipynb) · outputs are from a real run on 2026-09-30.

## Result

| | |
|---|---|
| task | compositions, grids, KPI and combo · the last 9 of the 24 types |
| model / lane | — |
| charts_built | 12 |
| treemap_leaves | — |
| types_available_here | 23 |
| xy_scatter_enabled | no |
| credits charged (this run) | 0 |
| project | `5c40e337-9753-4618-a7b3-574419f93c1e` |

Full numbers: [`results/metrics.json`](results/metrics.json).

## Run it

```bash
pip install -r ../../../requirements.txt && langsat login
jupyter lab 19_composition.ipynb
```
