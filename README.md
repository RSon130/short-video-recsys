# Short-Video RecSys

A two-stage recommender for short-video feeds — two-tower retrieval with FAISS
search, followed by a neural ranker — built on **KuaiRec**, a public
interaction dataset from the Kuaishou short-video platform, and deployed as a
containerised API on GCP Cloud Run.

This is a personal project on a public dataset, not a production system. There
is no live traffic; every result is an offline evaluation. What it does have is
the full lifecycle — data, training, evaluation, serving, deployment — and a
written record of what broke, how each problem was found, and what the honest
result turned out to be: **[docs/engineering_log.md](docs/engineering_log.md)**.

```
Request
  └─ Stage 1 · Retrieval   Two-tower, in-batch softmax (InfoNCE) + FAISS exact search → top-200
  └─ Stage 2 · Ranking     MLP ranker, pairwise loss                                  → top-K
  └─ Serving               FastAPI + in-memory TTL cache, Docker, Cloud Run
```

## Headline result

**Under a pre-registered, exposure-unbiased evaluation, neither learned stage
beats simple baselines.** The strongest signal in this data is non-personal
(item quality within a duration band) or temporal (recency).

How the project got there matters more than the number:

1. **The original hold-out measured exposure, not preference.** It only
   contains videos the platform chose to show. KuaiRec ships a near-fully
   observed `small_matrix` that shares every user and video with `big_matrix`
   but no (user, video) pair, and it had gone unused. It is now the test set.
2. **The raw data had 968,005 exact duplicate rows**, clustered on a few days.
   They had inflated the popularity baseline, and with it the earlier
   "+132.6% over popularity" claim, which is withdrawn.
3. **The evaluation protocol was written and reviewed before any model was
   scored:** [docs/evaluation_protocol.md](docs/evaluation_protocol.md). It
   fixes the label, six baselines, Holm-corrected comparisons, a three-cutoff
   sensitivity rule and an item bootstrap. Fresh-context AI-agent reviews (not human reviewers)
   checked the design, the implementation and the results.

Test users: 1,012. Label: within-user top 30% of within-duration-bucket
watch_ratio percentiles.

| system | per-user AUC | NDCG@10 |
|---|---|---|
| item quality within duration (non-personal baseline) | **0.548** | 0.320 |
| item-kNN collaborative filtering | 0.519 | 0.322 |
| ranker alone | 0.516 | 0.424 |
| shortest video first | 0.515 | **0.440** |
| retrieval | 0.512 | 0.240 |
| random | 0.500 | 0.297 |

The protocol's "ranker improves two-stage" test **technically passed**
(NDCG@10 +0.155). The follow-up diagnostics are why it is not reported as a
win:
- retrieval orders its own top-200 worse than random;
- the ranker's score correlates 0.96 with shortest-first;
- a plain shortest-first reorder of the same candidates scores higher.

The label's widest duration bucket (57–315 s) also still carries a duration
gradient. Full analysis: [engineering log §15–§17](docs/engineering_log.md).

| | status |
|---|---|
| Embedding collapse (all users identical) | ✅ fixed: 7,176 distinct user embeddings |
| Exposure-unbiased evaluation with pre-registered protocol | ✅ in place |
| Retrieval or ranker beats fair baselines | ❌ no |
| Label fully duration-controlled | ⚠️ not within the widest bucket; protocol v3 needed |
| Deployed service | ⚠️ still runs an older model; not updated, since no model beats the baselines |

## Phase 2: KuaiRand (in progress)

KuaiRec's only label, watch time, mostly measured video length. **KuaiRand**,
from the same group, adds explicit feedback (like, follow, comment, forward,
hate) and a log of randomly exposed videos. Phase 2 asks whether engagement
carries learnable personal preference once exposure and duration are
controlled. Plan: [docs/phase2_kuairand_plan.md](docs/phase2_kuairand_plan.md).

**Phase 0 gate: NO-GO.** The gate was pre-registered and reviewed before it
ran:
- **Setup:** a personalised LightGBM trained on the recommender-exposed log
  4/09–4/21, evaluated on validation users' random-exposure rows 4/22–5/08.
- **Explicit feedback (like, follow, comment or forward, 875 users):**
  per-user AUC 0.541, vs 0.553 for ranking shorter videos first and 0.541 for
  item impression count.
- **Watch-time thresholds (~6K users):** it beats impression count by
  +0.006 to +0.0095, below the pre-set 0.01 bar.
- **Post hoc:** popularity and tag-familiarity features predict *fewer* likes
  among exposed items but *more* across the catalogue. Models trained on
  exposed logs can learn exposure artefacts.
- **Caught before reporting:** a duration bug, twice, by the review process.

**Phase 2b: exposure bias, run once on 16,046 held-out users**
([pre-registration](docs/phase2_exposure_bias_plan.md)).

- **The sign flip replicates.** How often the recommender showed an item ranks
  explicit engagement *below* chance among the items it chose (0.441) and
  *above* chance under random exposure (0.548), a shift of +0.107.
- **Neither standard correction helped.** Inverse-popularity weighting changed
  nothing (−0.001); dropping exposure-volume features made ranking worse
  (−0.006 to −0.009).
- **Matching the training distribution did.** With users, window, features and
  row count matched, training on randomly exposed rows beat training on
  exposed rows by **+0.055** per-user AUC — with 2.8× fewer positives. But that
  arm does not beat a plain item rate, and the ordering reverses on exposed
  data, so it measures train/evaluation match, not a better model.
- **A Phase 0 result did not replicate.** The +0.0095 valid-play gain over
  impression count is +0.0009 (p 0.59) on held-out users, and is withdrawn.
- **Shipped anyway, and said what shipped.** The pre-registered rule was to
  deploy whichever scorer won, baseline or not, so Cloud Run now serves the
  impression-count scorer at `POST /recommend/kuairand`; `GET /scorer` returns
  the measured numbers behind that choice. Cloud-side p95 **9.56 ms**
  (handler 0.5 ms).

Details: [engineering log §18–§19](docs/engineering_log.md).

## Dataset

| | `small_matrix` (evaluation) | `big_matrix` (training) |
|---|---|---|
| Interactions | 4,676,570 | 11,561,093 after removing 968,005 duplicates |
| Users × items | 1,411 × 3,327 | 7,176 × 9,940 |
| Density | **99.6%** | ~16% |
| Shared (user, video) pairs | 0 | 0 |

### Superseded results

These were real measurements on the `big_matrix` temporal hold-out, which
rewards predicting exposure. They are kept so the history is auditable, not as
claims:

| comparison | collapsed model | after collapse fix | after dedup and protocol selection |
|---|---|---|---|
| retrieval vs popularity (recall@10) | +136.2% | +132.6% | popularity baseline no longer valid (duplicates) |
| full pipeline vs retrieval | +14.3% | −26.0% | −4.9%, not significant |

A post-hoc popularity baseline over only the last 3 days of training reaches
recall@10 0.050 on that hold-out, about 3× the pipeline.

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

PyTorch · FAISS · FastAPI · Docker · GCP Cloud Run · pandas/NumPy · pytest (212 tests)

## Next

1. **Protocol v3**, fixed before any new result: continuous or log-spaced
   duration conditioning checked *within* buckets; random and shortest-first
   reranker controls for the two-stage comparison; a recency baseline.
2. **Retrieval's anti-popularity bias**: test logQ correction in the in-batch
   softmax. Retrieval currently prefers long, unpopular videos.
3. **Ranker**: train it against what it is evaluated on (unwatched candidates)
   and remove its ability to sort by duration alone.
4. Redeploy only a model that beats the baselines under the protocol.
