# 10 · Credits, estimates, quotas and typed errors

**Use case.** Before wiring Langsat into automation you want to know what things cost, how much is left, and that a script can branch on *what* went wrong — without parsing English.

**What you will learn**
1. Balance, usage over time (as a chart) and quota
2. `estimates()` across every project of this series — one table
3. The training estimate with its knobs (`max_mode`, `fan_out_workers`)
4. The error catalogue in action: `NotFound`, `Invalid`, `NeedsUserSession` — and what the others mean
5. Least privilege: what the key may do

**What this costs.** nothing — every call here is a read.

> Every cell below ran for real against `api.langsat.ai` — the outputs are what the API returned. Re-running is safe:
> projects are found by name and reused, and a finished model is not retrained.

Notebook: [`10_credits_estimates_and_errors.ipynb`](10_credits_estimates_and_errors.ipynb) · outputs are from a real run on 2026-09-16.

## Result

| | |
|---|---|
| task | credits, estimates, typed errors |
| model / lane | — |
| plan | team |
| scopes_on_key | 20 |
| projects_estimated | 6 |
| errors_demonstrated | ['NotFound', 'Invalid', 'NeedsUserSession'] |

Full numbers: [`results/metrics.json`](results/metrics.json).

## Run it

```bash
pip install -r ../../../requirements.txt && langsat login
jupyter lab 10_credits_estimates_and_errors.ipynb
```
