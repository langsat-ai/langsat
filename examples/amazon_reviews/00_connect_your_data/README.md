# 00 · Connect your data — files, a Google Sheet, a database

**Use case.** Before anything else, the data has to get in. This chapter shows the three ways, with a fresh project for each, so every step runs for real: **upload** files, **link** a Google Sheet that stays live, and (as code you can copy) a **database connector**. The other chapters reuse projects; this one starts from nothing on purpose.

**What you will learn**
1. Create a project and upload CSV files — presign → PUT → confirm; rows counted server-side, plan caps enforced
2. Let Langsat detect keys and relationships, then read the sources back (`sources.list()`)
3. Link a Google Sheet: preview its tabs, import them, re-pull with `refresh()`, put it on a schedule
4. Private sheets: share with the Langsat service account, nothing else changes
5. A Postgres / MySQL connector on the Team plan (`ls.connectors`) — the calls, without a live database
6. Delete a project when you are done

**What this costs.** nothing — uploads, links, schema detection and a refresh under 500K rows are free; the two projects are deleted at the end.

> Every cell below ran for real against `api.langsat.ai` — the outputs are what the API returned. Re-running is safe:
> projects are found by name and reused, and a finished model is not retrained.

Notebook: [`00_connect_your_data.ipynb`](00_connect_your_data.ipynb) · outputs are from a real run on 2026-09-16.

## Result

| | |
|---|---|
| task | connect data · upload, Google Sheet, connector |
| model / lane | — |
| uploaded_rows | {'customer': 9821, 'product': 8607, 'review': 10000} |
| sheet_tabs | ['customer', 'product', 'review'] |
| refresh | succeeded |

Full numbers: [`results/metrics.json`](results/metrics.json).

## Run it

```bash
pip install -r ../../../requirements.txt && langsat login
jupyter lab 00_connect_your_data.ipynb
```
