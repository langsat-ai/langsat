# 14 · The cleaning plan — review it, measure it, change it, then clean

**Use case.** After schema detection Langsat writes a **cleaning plan**: one decision per column (keep / transform / drop, with the operations and the reason) that compiles to the SQL the clean step runs. Chapter 01 accepted it as generated. This chapter stops before the clean, reads the plan and its SQL, measures a candidate change on a sample, edits three columns by hand — a placeholder token to NULL, an HTML entity, a brand prefix stripped with a regex — saves the overlay and runs the clean. It also shows the one thing an API key cannot do, and why.

**What you will learn**
1. Stop after `schema.detect()` — the plan is `pending_review`, nothing has run yet
2. Read the generated plan: per-column actions, ops and reasons, the compiled SQL, corrections and warnings
3. `plan(measured=True)`: what each decision does to null rates and distinct values on a sample
4. Find what the LLM missed by looking at the data yourself (`\N`, `&amp;`, `Visit Amazon's … Page`)
5. `preview()` a candidate overlay, then `save()` it and `clean()` — the plan is yours, the SQL is compiled from it
6. Hand-written SQL (`raw_expr`) is refused for an API key by design — what the app's SQL editor is for
7. `reset()` back to the generated plan; the data-model workspace next door

**What this costs.** nothing — plan reads, previews and a clean under 500K rows are free.

> Every cell below ran for real against `api.langsat.ai` — the outputs are what the API returned. Re-running is safe:
> projects are found by name and reused, and a finished model is not retrained.

Notebook: [`14_cleaning_plan.ipynb`](14_cleaning_plan.ipynb) · outputs are from a real run on 2026-09-16.

## Result

| | |
|---|---|
| task | cleaning plan · review, preview, edit, clean |
| model / lane | — |
| columns_in_plan | 15 |
| llm_transforms | 2 |
| category_nulls_after | 348 |
| brand_titles_left | 0 |
| project | `0d64a986-48a4-4bef-86aa-c5bfb5fb4fb6` |

Full numbers: [`results/metrics.json`](results/metrics.json).

## Run it

```bash
pip install -r ../../../requirements.txt && langsat login
jupyter lab 14_cleaning_plan.ipynb
```
