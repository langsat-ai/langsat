# 10 · Credits, estimates, quotas and typed errors

**Use case:** before wiring Langsat into automation you want to know what things cost, how much is left, and that a
script can branch on *what* went wrong — without parsing English.

**Sub-tasks**
1. Balance and usage (`credits.dashboard`, `credits.usage`, `credits.quota_status`)
2. `estimates()` on the analysis project and on a modeling project (training, serving)
3. The error catalogue: `NotFound`, `Invalid`, and what a `MissingScope` looks like
4. The key's scopes (`me()`) — least privilege

Notebook: [`10_credits_estimates_and_errors.ipynb`](10_credits_estimates_and_errors.ipynb) · outputs are from a real run on 2026-09-15.

## Result

| | |
|---|---|
| task | credits, estimates, typed errors |
| model / lane | — |
| tier | team |
| scopes_on_key | 17 |
| errors_demonstrated | ['NotFound', 'Invalid/LangsatError', 'Conflict/LangsatError'] |

Full numbers: [`results/metrics.json`](results/metrics.json).

## Run it

```bash
pip install -r ../../../requirements.txt && langsat login
jupyter lab 10_credits_estimates_and_errors.ipynb
```
