# 01 · Set up a project and explore the data

**Use case:** a data team gets three CSV exports from a bookstore's review system (`customer`, `product`, `review`)
and wants them queryable, joined and charted in minutes — with no ETL.

**Sub-tasks**
1. Sign in with an API key (`langsat login` or `LANGSAT_API_KEY`)
2. Create a project, upload the three CSVs
3. Let Langsat detect primary keys, foreign keys and time columns
4. See what a clean would cost, then clean (free under 500K rows)
5. Pull rows into pandas and look at the data
6. Build two charts on a dashboard and publish a public link
7. Ask the data a question in plain English

Everything here runs on a **`data_analysis`** project (no training). The modeling notebooks (02–06) build on the same data: 10,000 reviews by 9,821 customers of 8,607 books — every review's customer and product are in the tables, so the relational graph is complete.

Notebook: [`01_setup_and_explore.ipynb`](01_setup_and_explore.ipynb) · outputs are from a real run on 2026-09-15.

## Result

| | |
|---|---|
| task | setup + explore (no model) |
| model / lane | — |
| published_link | yes |
| answer_chars | 104 |
| project | `0b6ffd93-08da-4220-9dfa-5b62b5f5951f` |

Full numbers: [`results/metrics.json`](results/metrics.json).

## Run it

```bash
pip install -r ../../../requirements.txt && langsat login
jupyter lab 01_setup_and_explore.ipynb
```
