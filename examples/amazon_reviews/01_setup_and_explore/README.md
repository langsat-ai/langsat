# 01 · Set up a project and explore the data

**Use case.** A data team gets three CSV exports from a bookstore's review system — `customer`, `product`, `review` — and wants them queryable, joined and charted in minutes, with no ETL and no notebook full of `pd.merge`. This is the notebook every other one builds on.

**What you will learn**
1. Sign in with an API key and see who the key acts as
2. Create a project and upload the three CSVs — the API counts rows and enforces your plan's row cap
3. Let Langsat detect primary keys, foreign keys and the time column
4. See what a clean would cost, then clean (free under 500K rows)
5. Read rows into pandas — filters, sorting, search, paging
6. Build charts on a dashboard, draw them **as images** in the notebook, publish a public link
7. Let the AI build a dashboard (`generate_cards`), and ask questions in plain English — with a chart you can plot from code

**What this costs.** no training. Cleaning is 0 credits under 500K rows; `generate_cards` and each `ask` count against your AI quota (not credits).

> Every cell below ran for real against `api.langsat.ai` — the outputs are what the API returned. Re-running is safe:
> projects are found by name and reused, and a finished model is not retrained.

Notebook: [`01_setup_and_explore.ipynb`](01_setup_and_explore.ipynb) · outputs are from a real run on 2026-09-16.

## Result

| | |
|---|---|
| task | setup + explore (no model) |
| model / lane | — |
| published_link | yes |
| chat_chart | yes |
| answer_chars | 489 |
| project | `0b6ffd93-08da-4220-9dfa-5b62b5f5951f` |

Full numbers: [`results/metrics.json`](results/metrics.json).

## Run it

```bash
pip install -r ../../../requirements.txt && langsat login
jupyter lab 01_setup_and_explore.ipynb
```
