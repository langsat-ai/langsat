# 16 · The grammar of a chart — what a recipe is, and why `chart_type` is not a chart name

**Use case.** Everything from here to chapter 25 is the half of Langsat that never calls a model: you describe a chart, the server compiles it to SQL and runs it on your data. It costs nothing, it re-aggregates itself every time the data refreshes, and it is the same object the app's chart builder writes. This chapter is the grammar — read it once and the nine chapters after it are vocabulary.

**What you will learn**
1. Read the whole grammar from the server: 24 chart types, your columns, the joins, every bound
2. Why `chart_type` is a **Plotly trace type** and writing `"line"` draws a scatter while reporting success
3. Preview a recipe, then persist **the recipe the preview returned** — not the one you composed
4. The three things a recipe adds for you, so it bounds itself
5. Read a refusal: the reason comes back as a sentence, before anything is saved
6. Prove from the credit ledger that none of it charged

**What this costs.** nothing. Every call in this chapter is free — no model is in the path. The ledger is read at the start and at the end and the difference is printed.

> Every cell below ran for real against `api.langsat.ai` — the outputs are what the API returned. Re-running is safe:
> projects are found by name and reused, and a finished model is not retrained.

Notebook: [`16_chart_grammar.ipynb`](16_chart_grammar.ipynb) · outputs are from a real run on 2026-09-30.

## Result

| | |
|---|---|
| task | the recipe grammar · 24 chart types, preview → persist, refusals |
| model / lane | — |
| chart_types | 24 |
| trace_types | 10 |
| columns | 15 |
| not_authorable | 4 |
| xy_scatter_enabled | no |
| raw_date_groups | 2772 |
| raw_date_drawn | 50 |
| credits charged (this run) | 0 |
| project | `5c40e337-9753-4618-a7b3-574419f93c1e` |

Full numbers: [`results/metrics.json`](results/metrics.json).

## Run it

```bash
pip install -r ../../../requirements.txt && langsat login
jupyter lab 16_chart_grammar.ipynb
```
