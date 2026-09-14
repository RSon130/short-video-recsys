# Short-Video RecSys

A two-stage recommender for short-video feeds — two-tower retrieval with FAISS
search, followed by a neural ranker — built on **KuaiRec**, a public
interaction dataset from the Kuaishou short-video platform, and deployed as a
containerised API on GCP Cloud Run.

This is a personal project on a public dataset, not a production system. There
is no live traffic; every result is an offline evaluation on a temporal
hold-out. What it does have is the full lifecycle — data, training, statistical
evaluation, serving, deployment — and a written record of what broke along the
way and how each problem was found: **[docs/engineering_log.md](docs/engineering_log.md)**.

```
Request
  └─ Stage 1 · Retrieval   Two-tower, in-batch softmax (InfoNCE) + FAISS exact search → top-200
  └─ Stage 2 · Ranking     MLP ranker, pairwise loss                                  → top-K
  └─ Serving               FastAPI + in-memory TTL cache, Docker, Cloud Run
```

## Current status

> **Results were re-measured on 2026-09-14 and several earlier headline numbers
> are superseded.** The retrieval model they came from had silently collapsed:
> all 7,176 user embeddings were one identical vector, so every user received
> the same list. The fix, the diagnosis, and the re-measurement are in
> [engineering log §11 and §14](docs/engineering_log.md).

| | status |
|---|---|
| Retrieval personalises (7,176 distinct user embeddings) | ✅ fixed and re-measured |
| Retrieval beats popularity | ✅ +132.6% recall@10, 95% CI excludes zero |
| Ranker improves on retrieval | ❌ **currently −26% — the ranker hurts**; under investigation |
| Duration-confounded label | ⚠️ measured, fix designed, not enabled ([label_design.md](docs/label_design.md)) |
| Deployed service | ⚠️ still runs the pre-fix model; not updated while the ranker regression is open |

## Dataset

| | `small_matrix` | `big_matrix` (default) |
|---|---|---|
| Interactions | 4,676,570 | **12,529,113** |
| Users × items | 1,411 × 3,327 | 7,176 × 9,958 |
| Density | **99.6%** | 17.5% |
| Relevance base rate after excluding seen items | 25% | 1.1% |
| Split | temporal 80/10/10 | temporal 80/10/10 |

`small_matrix` is *fully observed* — nearly every (user, item) pair carries a
real `watch_ratio`. That invalidates several standard recipes that assume a
sparse matrix, and it is the root cause of three separate bugs in the log.

Switch `interaction_file` in `config/kuairec.yaml` to change subsets.

## Results — `big_matrix`, 6,873 test users

Relevance is `watch_ratio >= 0.7` in the test split. Every system excludes items
the user already saw in training.

| system | recall@5 | recall@10 | ndcg@10 | recall@20 |
|---|---|---|---|---|
| popularity baseline | 0.0052 | 0.0055 | 0.0052 | 0.0061 |
| **retrieval only** | **0.0143** | **0.0129** | **0.0134** | **0.0121** |
| full pipeline | 0.0097 | 0.0095 | 0.0093 | 0.0102 |

Paired bootstrap on identical users, recall@10:

| comparison | lift | 95% CI | |
|---|---|---|---|
| retrieval vs popularity | **+132.6%** | [+0.0063, +0.0084] | significant |
| full pipeline vs popularity | +72.1% | [+0.0030, +0.0050] | significant |
| full pipeline vs retrieval only | **−26.0%** | [−0.0045, −0.0022] | significant |

What this does and does not show:

- **Retrieval works.** It beats popularity by a wide, statistically clear margin.
- **The ranker currently makes things worse.** Before the collapse fix it added
  +14.3% on top of retrieval; retrained on the new embeddings it subtracts 26%.
  The cause is not yet known. Leading hypothesis and next steps are in
  [engineering log §14](docs/engineering_log.md).
- **Fixing the collapse did not raise recall.** The collapsed model scored
  +136.2% over popularity and the fixed one +132.6% — at this base rate a global
  ordering with per-user seen-item exclusion is already a strong baseline.
  The fix matters because the system now responds to the user at all.
- Retrieval's watch-time AUC among its own candidates is 0.34, below chance.
  That is the most likely lead on both open problems.

### Superseded results

These were real measurements, taken on the collapsed retrieval model. They are
kept so the history is auditable, not as claims:

| comparison | then | now |
|---|---|---|
| retrieval vs popularity | +136.2% | +132.6% |
| full pipeline vs popularity | +169.9% | +72.1% |
| full pipeline vs retrieval only | +14.3% | −26.0% |

The ranker objective comparison (pointwise MSE vs pairwise vs listwise) and the
`small_matrix` results in the engineering log were also measured before the fix
and have not been re-run.

## Experiment statistics

Comparisons are **paired** — every system scored on the same users, then
differenced per user. A live A/B test splits traffic because one person cannot
see two feeds at once; offline that constraint does not exist, and pairing
removes between-user variance, which here is far larger than the effects being
measured.

The harness also reports a sample-ratio check on the hash bucketing (0.504 vs
0.500 expected, z=+0.69) and a power calculation. At 3,436 users per arm the
smallest detectable lift is 26.5%, and detecting a 5% lift would take ~97,000
users per arm.

This exists because of a specific failure: a 150-user sample once showed the
pipeline ahead of popularity by +0.7%, and the full test set reversed that to
−0.9%. A confidence interval says "no signal" immediately; a point estimate
does not.

## Design decisions

**Negatives come from observed low engagement, not unobserved pairs.** At 99.6%
density a uniformly drawn "negative" is an observed interaction with the same
`watch_ratio` distribution as the positives (mean 0.702 either way). Positives
are `watch_ratio >= 0.7`, negatives the *same user's* items at `<= 0.3`, and the
ambiguous middle is excluded. BPR loss went from 0.597 (no signal) to 0.264.

**Retrieval uses in-batch softmax with false-negative masking.** Pairwise BPR
has no term relating one user to another, and every user embedding collapsed
onto the dominant item-quality direction. In-batch softmax adds cross-user
competition, but 13.5%+ of in-batch negatives were the user's own positives,
which held training at chance until known positives were masked out.

**Checkpoints are selected on held-out recall@10, not validation loss** — for
both models. Two identical ranker runs had near-identical loss and a 38% recall
gap; for retrieval, validation loss rose every epoch after the first.

**Training and serving share one feature path.** An earlier version served the
ranker zeros for 47% of its input. Features are normalised (signed `log1p`, then
z-score) at write time, and evaluation refuses to run if model provenance and
config disagree.

**Relevance is engagement, not exposure.** On a matrix where nearly everything
is shown, counting any test row as relevant measures exposure.

**Exact search (`IndexFlatIP`), not IVF or HNSW — measured.** Full results in
[docs/index_benchmark.md](docs/index_benchmark.md).

| index | p50 | recall@200 | size |
|---|---|---|---|
| **flat (exact)** | **0.087 ms** | **1.0000** | 2.55 MB |
| ivfflat | 0.036 ms | 0.9051 | 2.66 MB |
| hnsw | 0.045 ms | 0.7683 | 5.26 MB |

Against a 9.6 ms end-to-end p95, ANN saves 0.5% of a request for 10–23% of the
true neighbours. Exact search is linear (0.087 ms at 10K, 9.19 ms at 1M), so the
crossover is near 300–500K items. At 100K, recall >= 0.9 needs HNSW at
efSearch=256, only 1.3x faster than brute force.

**Side features are excluded from the retrieval towers.** An ablation on
`small_matrix` found they lifted retrieval alone but cut end-to-end recall by a
quarter. That ablation predates the collapse fix and has not been re-run.

## Serving latency

Deployed on **GCP Cloud Run** (`us-central1`, 1 vCPU / 2Gi, scale-to-zero), 500
requests at concurrency 4. Latency depends on model shape (64-dim embeddings,
200 candidates, the same MLP), not on learned weights, so these figures apply to
the retrained model as well.

| vantage | p50 | p95 | p99 | includes |
|---|---|---|---|---|
| app self-reported | — | 2.4 ms | — | handler only |
| **Cloud Run platform** (`request_latencies`) | **5.0 ms** | **9.6 ms** | 10.0 ms | handler + framework + ingress |
| client (laptop) | — | 75–105 ms | — | + internet round trip |

The platform figure is stable while the client figure varies 40% between
sessions: a remote client's wait is dominated by network. That is also why the
TTL cache is invisible end to end — it still protects CPU under concurrency,
but a faster feed would come from regional placement. Deployment is scripted:
[docs/deployment.md](docs/deployment.md).

## Quickstart

Everything runs in Docker — CPU by default, which is also what Cloud Run uses.

```bash
docker compose -f docker-compose.cpu.yml build
```

```bash
docker compose -f docker-compose.cpu.yml run --rm dev python scripts/download_kuairec.py
```

```bash
docker compose -f docker-compose.cpu.yml run --rm dev python features/engineer.py
```

```bash
docker compose -f docker-compose.cpu.yml run --rm dev python training/train_retrieval.py
```

```bash
docker compose -f docker-compose.cpu.yml run --rm dev python scripts/build_index.py
```

```bash
docker compose -f docker-compose.cpu.yml run --rm dev python training/train_ranking.py
```

```bash
docker compose -f docker-compose.cpu.yml run --rm dev python scripts/evaluate.py
```

Serve the API:

```bash
docker compose -f docker-compose.cpu.yml up api
```

```bash
curl -X POST http://localhost:8000/recommend -H "Content-Type: application/json" -d '{"user_id": 0, "top_k": 20}'
```

Tests:

```bash
docker compose -f docker-compose.cpu.yml run --rm dev pytest -q
```

## Project structure

```
config/          base.yaml + kuairec.yaml
config_loader.py deep-merge config loading, shared by every entry point
data/            loaders, canonical schema, factory   (code only)
datastore/       raw + processed data                 (gitignored)
features/        engineer.py, dense_features.py
models/          two_tower.py, ranker.py
training/        train_retrieval.py, train_ranking.py
retrieval/       faiss_index.py
serving/         FastAPI api.py
evaluation/      metrics.py, baselines.py, ab_test.py
scripts/         download, build_index, evaluate, benchmark_index, loadtest, deploy
docs/            engineering_log.md   what broke, how it was found, what it measured
                 label_design.md      the duration confound in the training label
                 index_benchmark.md   exact vs ANN search
                 deployment.md        Cloud Run deploy and latency measurement
```

## Stack

PyTorch · FAISS · FastAPI · Docker · GCP Cloud Run · pandas/NumPy · pytest (196 tests)

## Next

1. **Explain the ranker regression** — check the ranker's input distribution
   under the new embeddings; add logQ correction to retrieval's in-batch softmax.
2. **Duration-debiased label**, as a separate measured change.
3. Re-run the ranker objective comparison under recall-based selection.
4. Redeploy once the full pipeline beats retrieval alone.
