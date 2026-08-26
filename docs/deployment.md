# Deployment

The API ships as a self-contained container: the ~10 MB of trained artifacts are
baked into the image, so the service has no runtime dependency on object storage
or credentials. `docker run` is the whole contract, which is what makes it
portable across Cloud Run, App Runner, Fly, or a plain VM.

## Measured locally (for comparison)

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

## Measured on Cloud Run (live)

`us-central1`, 1 vCPU / 2Gi, `min-instances 0`, 500 requests at concurrency 4,
driven from a laptop in the Bay Area.

| | p50 | p95 | p99 |
|---|---|---|---|
| **cold** — distinct users, every request a cache miss | 62.0 ms | **74.6 ms** | 117.2 ms |
| warm — 10 users reused, ~98% cache hits | 71.3 ms | 82.2 ms | 106.9 ms |

**Server-side p95: 2.5 ms.** That is the model path — FAISS over 9,958 items,
feature assembly for 200 candidates, one batched ranker forward pass — and it is
the number that describes the system. Everything else is network.

### The warm phase is *slower*, and that is the finding

Cache hits made no difference: ~98% of warm requests were served from cache, and
end-to-end latency got marginally worse. Both facts have the same cause — at
2.5 ms of server compute against ~70 ms of round trip, **the cache is optimising
3% of the request**. The warm run simply landed on slightly different network
conditions, and that noise is larger than the entire thing the cache saves.

This is worth stating plainly rather than hiding, because it inverts the obvious
conclusion. The TTL cache is not useless — it protects CPU under concurrency and
would matter on a hot key — but as a *latency* optimisation for a remote client
it is invisible. If the goal were a faster feed, the money is in edge proximity
or regional deployment, not in caching. Measuring is what distinguishes those.

### Which number to quote

Both are honest, and they answer different questions:

- **2.5 ms server-side** — how fast the recommender is.
- **74.6 ms end-to-end** — what a client in the same country actually waits,
  network included.

Quoting 74.6 ms as "model latency" would understate the system; quoting 2.5 ms
as "user-perceived latency" would overstate it. Local numbers (cold p95 4.7 ms)
are not comparable to either — a dedicated laptop core with no network.

## What you need before deploying

| requirement | notes |
|---|---|
| `gcloud` CLI | `winget install --id Google.CloudSDK` on Windows. Installed on Windows rather than WSL deliberately: `gcloud auth configure-docker` writes a credential helper into the Docker config the *Windows* docker CLI reads, and a WSL gcloud paired with Docker Desktop mismatches on Artifact Registry pushes. |
| Google account + project | `gcloud projects create <ID>` or use an existing one |
| **Billing enabled** | Required even inside the free tier. Cloud Run will not deploy without it. |
| Docker | already in use here |

Cloud Run's free tier (2M requests, 180k vCPU-seconds, 360k GiB-seconds per
month) comfortably covers a portfolio demo at `--min-instances 0`.

### Windows / Git Bash note

The Cloud SDK ships its own Python. In Git Bash, `gcloud` otherwise picks up
whatever `python` is first on PATH — a Windows Store Python 3.9 here — and
refuses to run: *"no longer supported by gcloud"*. `scripts/deploy_cloudrun.sh`
detects the bundled interpreter automatically. To run gcloud by hand in the
same shell:

```bash
export PATH="$PATH:/c/Users/$USERNAME/AppData/Local/Google/Cloud SDK/google-cloud-sdk/bin"
```

```bash
export CLOUDSDK_PYTHON="$(cygpath -w "/c/Users/$USERNAME/AppData/Local/Google/Cloud SDK/google-cloud-sdk/platform/bundledpython/python.exe")"
```

PowerShell needs neither — the installer wires both up there.

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
