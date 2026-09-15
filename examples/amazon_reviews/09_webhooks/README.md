# 09 · Webhooks — be told when a job finishes

**Use case:** a nightly pipeline re-uploads data and cleans; instead of polling, the orchestrator is told when the clean
finishes (`job.succeeded`) and kicks off the next step.

**Sub-tasks**
1. Register an https endpoint (here a throw-away https://webhook.site URL) for `clean` jobs
2. Send a signed test ping and verify the signature the way a receiver would
3. Run a clean and watch `job.succeeded` arrive exactly once
4. Read the delivery log; rotate the secret; delete the webhook

Set `WEBHOOK_URL` (and `WEBHOOK_SITE_TOKEN` — the uuid in the URL — to read what arrived) before running; without them
the notebook explains and skips the live steps.

Notebook: [`09_webhooks.ipynb`](09_webhooks.ipynb) · outputs are from a real run on 2026-09-15.

## Result

| | |
|---|---|
| task | webhooks · job.succeeded on clean |
| model / lane | — |
| ran_live | yes |
| ping_ok | yes |
| job_succeeded_seen | yes |
| project | `0b6ffd93-08da-4220-9dfa-5b62b5f5951f` |

Full numbers: [`results/metrics.json`](results/metrics.json).

## Run it

```bash
pip install -r ../../../requirements.txt && langsat login
jupyter lab 09_webhooks.ipynb
```
