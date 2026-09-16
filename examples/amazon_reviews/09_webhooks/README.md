# 09 · Webhooks — be told when a job finishes

**Use case.** A nightly pipeline re-uploads data and cleans; instead of polling, the orchestrator is told when the clean finishes (`job.succeeded`) and kicks off the next step. The delivery is signed, retried, and logged.

**What you will learn**
1. Register an https endpoint (a throw-away https://webhook.site URL here) for `clean` jobs
2. Send a signed test ping and verify the signature the way a receiver would
3. Run a clean and watch `job.succeeded` arrive exactly once
4. Read the delivery log; rotate the secret; delete the webhook
5. Receiver code you can paste into FastAPI or Express

**What this costs.** no credits; the clean is free. Needs the `webhooks:read` / `webhooks:write` scopes on the key.

> Every cell below ran for real against `api.langsat.ai` — the outputs are what the API returned. Re-running is safe:
> projects are found by name and reused, and a finished model is not retrained.

Notebook: [`09_webhooks.ipynb`](09_webhooks.ipynb) · outputs are from a real run on 2026-09-16.

## Result

| | |
|---|---|
| task | webhooks · job.succeeded on clean |
| model / lane | — |
| ran_live | yes |
| ping_ok | yes |
| job_succeeded_seen | yes |
| project | `5c40e337-9753-4618-a7b3-574419f93c1e` |

Full numbers: [`results/metrics.json`](results/metrics.json).

## Run it

```bash
pip install -r ../../../requirements.txt && langsat login
jupyter lab 09_webhooks.ipynb
```
