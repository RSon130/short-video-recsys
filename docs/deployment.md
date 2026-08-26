# Deployment

The API ships as a self-contained container: the ~10 MB of trained artifacts are
baked into the image, so the service has no runtime dependency on object storage
or credentials. `docker run` is the whole contract, which is what makes it
portable across Cloud Run, App Runner, Fly, or a plain VM.

## Measured locally

Deploy image (`--target deploy`), 1,000 requests, concurrency 1, on an 8-core
laptop. Cold and warm are reported separately because reporting them together
overstates performance by whatever the cache hit rate happens to be.

| | p50 | p95 | p99 | max |
|---|---|---|---|---|
| **cold** — distinct users, every request misses the cache | 3.8 ms | **4.7 ms** | 5.1 ms | 107 ms |
| warm — 10 users reused, ~99% cache hits | 2.0 ms | 2.8 ms | 3.1 ms | 10 ms |

Server-side p95 on the cold path is **1.6 ms**, excluding network and client
overhead — that is the model serving path itself: FAISS query over 9,958 items,
feature assembly for 200 candidates, one batched ranker forward pass.

**Cold start: 2.0 s** (median of 3 trials, container start to healthy), first
inference after readiness ~30 ms.

Quote the **cold** numbers. The warm figures measure the cache, not the model.

> The 107 ms max on the cold path is the first request of the run — lazy
> initialisation inside torch and FAISS. It is why p95 rather than max is the
> number worth reporting, and why a single-request measurement is worthless in
> either direction.

## What you need before deploying

| requirement | notes |
|---|---|
| `gcloud` CLI | https://cloud.google.com/sdk/docs/install — then `gcloud init` |
| Google account + project | `gcloud projects create <ID>` or use an existing one |
| **Billing enabled** | Required even inside the free tier. Cloud Run will not deploy without it. |
| Docker | already in use here |

Cloud Run's free tier (2M requests, 180k vCPU-seconds, 360k GiB-seconds per
month) comfortably covers a portfolio demo at `--min-instances 0`.

## Deploy

```bash
gcloud auth login
```

```bash
gcloud config set project YOUR_PROJECT_ID
```

```bash
./scripts/deploy_cloudrun.sh
```

The script enables the required APIs, creates an Artifact Registry repository,
builds and pushes the deploy image, and deploys the service. It is idempotent —
re-run it to ship a new revision.

## Then measure

```bash
python scripts/loadtest.py --url https://YOUR-SERVICE-URL --requests 500
```

Cloud Run numbers will be worse than local: a shared vCPU rather than a
dedicated laptop core, plus real network latency. Report what is measured there
rather than reusing the local figures.

## Sizing choices

- `--memory 2Gi` — torch plus resident artifacts. 1Gi is tight.
- `--cpu 1` — the ranker forward pass is single-threaded at this batch size.
- `--min-instances 0` — scales to zero, stays inside the free tier. A cold start
  costs ~2 s locally; on Cloud Run add image pull on a revision's first start.
  Set to 1 only if a demo link must never be slow, and note that idle instances
  bill outside the free tier.
- `--concurrency 40` — FastAPI is async but scoring is CPU-bound, so this caps
  queueing behind the ranker instead of letting latency climb under load.

## Why not Cloudflare

Workers run V8 isolates — no PyTorch, and FAISS is a native C++ library that
Pyodide-based Python Workers cannot load. Cloudflare Containers could run the
image, but the platform's value is edge proximity, and this workload is
compute-bound rather than network-bound. Cloudflare is a reasonable choice for a
custom domain or a static demo page in front of the service, not for hosting it.
