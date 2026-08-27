# Short-Video RecSys

A two-stage recommender for short-video feeds — two-tower retrieval with ANN
search, followed by a neural ranker — built on KuaiRec, a fully-observed
interaction dataset from the Kuaishou short-video platform. The architecture
follows the retrieval-then-ranking pattern used by YouTube, TikTok, and
Pinterest.

```
Request
  └─ Stage 1 · Retrieval   Two-Tower (BPR) + FAISS IndexFlatIP → top-200 candidates
  └─ Stage 2 · Ranking     MLP ranker (pairwise BPR loss)      → top-K scored items
  └─ Serving               FastAPI + in-memory TTL cache
```

## Dataset

KuaiRec ships two subsets, and the system was measured on both. `small_matrix`
is *fully observed* — nearly every (user, item) pair carries a real
`watch_ratio` — which invalidates several standard recipes and drives most of
the design decisions below.

| | `small_matrix` | `big_matrix` |
|---|---|---|
| Interactions | 4,676,570 | **12,529,113** |
| Users × items | 1,411 × 3,327 | 7,176 × 9,958 |
| Density | **99.6%** | 17.5% |
| Split | temporal 80/10/10 | temporal 80/10/10 |

Default config runs `big_matrix`; switch the `interaction_file` in
`config/kuairec.yaml` for the dense subset.

## Results

The system was evaluated on both KuaiRec subsets, which sit in very different
regimes. The contrast is the most interesting result here: **which stage carries
the system depends on catalogue size and density.**

| | `small_matrix` | `big_matrix` |
|---|---|---|
| Interactions | 4.68M | **12.53M** |
| Users × items | 1,411 × 3,327 | 7,176 × 9,958 |
| Density | 99.6% | 17.5% |
| Eligible items per user | ~676 | ~8,771 |
| Relevance base rate | 25% | 1.1% |

### `big_matrix` — 12.5M interactions, 6,873 test users

| system | recall@5 | recall@10 | ndcg@10 | recall@20 |
|---|---|---|---|---|
| popularity baseline | 0.0052 | 0.0055 | 0.0052 | 0.0061 |
| retrieval only | **0.0191** | 0.0131 | 0.0130 | 0.0134 |
| **full pipeline** | 0.0150 | **0.0149** | **0.0133** | 0.0112 |

Every comparison is a **paired bootstrap** on the same users, with a 95%
confidence interval — see [Experiment statistics](#experiment-statistics).

| comparison | lift | 95% CI | |
|---|---|---|---|
| retrieval vs popularity | +136.2% | [+0.0065, +0.0086] | significant |
| full pipeline vs popularity | **+169.9%** | [+0.0084, +0.0104] | significant |
| full pipeline vs retrieval only | **+14.3%** | [+0.0007, +0.0030] | significant |

At a 1.1% relevance base rate a non-personalised list stops working, and
narrowing 8,771 candidates to 200 has real value — the opposite of
`small_matrix`, where a 25% base rate made popularity unbeatable. Retrieval
still leads at K=5: the ranker only ever sees the shortlist it is handed.

## Experiment statistics

Comparisons are **paired** — every system scored on identical users, then
differenced per user. A live A/B test splits traffic because it must (one person
cannot be shown two feeds at once) and pays for it with between-group variance.
Offline that constraint does not exist, so copying the split would discard half
the data per model for nothing. Pairing also removes between-user variance,
which here dwarfs the effects being measured: the +14.3% two-stage lift is
detectable *because* it is paired.

The harness also reports a sample-ratio check on the hash bucketing (0.504 vs
0.500 expected, z=+0.69) and a power calculation. That last one is sobering: at
3,436 users per arm the smallest detectable lift is ~21%, and detecting a 5%
lift would need ~62,000 users per arm.

This exists because of a specific failure. A 150-user sample once showed the
pipeline beating popularity by +0.7%, and the full 1,411-user set reversed that
to −0.9%. The sample was far below the detectable threshold — it was never a
signal. A confidence interval says so immediately; a point estimate does not.

### The ranker's objective matters more than its architecture

The same network, the same features, the same candidates — only the loss changed.
All three objectives are implemented and selectable via `ranking.objective`:

| objective | recall@5 | recall@10 | recall@20 | watch-AUC |
|---|---|---|---|---|
| pointwise — MSE on watch_ratio | 0.0031 | 0.0070 | 0.0093 | **0.714** |
| **pairwise — BPR** | **0.0112** | **0.0141** | 0.0099 | 0.622 |
| listwise — sampled softmax | 0.0081 | 0.0099 | **0.0100** | 0.609 |

MSE optimises *calibration* — how much of a video someone will watch — and it
wins on watch-time AUC, which is exactly the metric that rewards calibration.
But recall@K rewards *ordering*, and a regression head minimising squared error
is pulled toward the conditional mean, flattening the distinctions that decide
the top slots. Under MSE the full pipeline scored below retrieval alone at every
cutoff; under a ranking loss it overtakes retrieval at K=10 and beats popularity
everywhere.

Retrieval had trained with a ranking loss (BPR) from the start. The ranker was
the only stage optimising something other than the metric it was judged on.

The trade is visible and expected: watch-time AUC falls from 0.714 to 0.622.
A production system wanting both would use a multi-task head — ranking loss for
ordering, regression for calibrated watch-time prediction.

**Listwise did not beat pairwise**, contrary to expectation — sampling more
negatives per step usually helps, which is why large-scale rankers use a sampled
softmax. One explanation was tested and refuted: per-user negative pools are
large (median 233 items; only 1% of users below 10), so drawing 4 with
replacement is not collapsing to duplicates.

The remaining hypothesis is untested: with four *easy* random negatives the
softmax is satisfied as soon as the positive outranks all of them, so gradients
vanish earlier in training than single-pair BPR's. If that is right, the fix is
harder negatives rather than more of them — sampling from retrieval's shortlist
instead of the whole low-watch pool.

Retrieval still leads at K=5 and K=20. The ranker only ever sees what retrieval
passes it, so its ceiling is retrieval's shortlist. Both open threads therefore
point at the same place: **candidate generation, not the ranker.**

### `small_matrix` — 4.68M interactions, all 1,411 test users

Relevance is `watch_ratio >= 0.7`; every system excludes items the user already
consumed in training.

| system | recall@10 | ndcg@10 | recall@20 | ndcg@20 | watch-time AUC |
|---|---|---|---|---|---|
| popularity baseline | **0.4806** | **0.4900** | 0.4629 | 0.4744 | — |
| retrieval only | 0.2824 | 0.3323 | 0.2599 | 0.2987 | 0.542 |
| **full pipeline** | 0.4609 | 0.4310 | 0.4588 | 0.4406 | **0.826** |

**What the ranking stage adds** — the case for two stages:

| metric | retrieval only | full pipeline | lift |
|---|---|---|---|
| recall@10 | 0.2824 | 0.4609 | **+63.2%** |
| recall@20 | 0.2599 | 0.4588 | **+76.5%** |
| ndcg@20 | 0.2987 | 0.4406 | **+47.5%** |
| watch-time AUC | 0.542 | 0.826 | **+52.4%** |

**On this subset the pipeline does not beat the popularity baseline** (−4.1%
recall@10, −0.9% recall@20). That is a real finding, not a tuning failure:

- After excluding seen items, each user has only ~676 eligible items, ~173 of
  which are relevant — a **25% base rate**. On a small, dense catalogue where
  popular items are broadly enjoyed, popularity is a genuinely strong baseline.
- Retrieval's job is *narrowing*. At 3,327 items there is little to narrow, so
  the stage that carries a production system contributes little here. The
  measured retrieval AUC of 0.542 says as much.

A 150-user sample initially showed the pipeline ahead by 0.7% at recall@20. On
the full test set that reverses to −0.9%, so the apparent win was sampling
noise. The full-set number is the one reported.

## Design decisions

**Negatives come from observed low engagement, not from unobserved pairs.**
The standard implicit-feedback recipe — observed item positive, random
unobserved item negative — assumes a sparse matrix, where an unobserved pair is
a fair guess at disinterest. At 99.6% density there are almost no unobserved
pairs, so a uniformly drawn "negative" is an observed interaction with the same
`watch_ratio` distribution as the positives (mean 0.702 either way). Trained
that way, BPR loss sat at 0.597 against a random-init baseline of log(2) ≈
0.693 — the positives and negatives were statistically identical and there was
no signal to learn. Positives are now `watch_ratio >= 0.7`, negatives are the
*same user's* items at `<= 0.3`, and the ambiguous middle band is excluded.
Loss dropped to 0.264.

**Relevance is engagement, not exposure.** Ground truth uses the same
`watch_ratio` threshold as training. Counting any test-split row as relevant
would measure which items a user was *shown* on a matrix where nearly
everything is shown.

**Side features are excluded from the retrieval towers, by measurement.**

| towers | retrieval recall@10 | full-pipeline recall@10 | AUC |
|---|---|---|---|
| ID only | 0.2747 | **0.4793** | **0.854** |
| ID + side features | 0.2887 | 0.3593 | 0.753 |

Side features lift retrieval slightly on its own yet cost the end-to-end system
a quarter of its recall. The item features are dominated by category one-hots
and popularity counts, which cluster the embedding space by category rather
than by affinity, handing the ranker a more homogeneous shortlist. The ranker
consumes those same features directly, where they measurably help.

**Dense features are normalised before use.** Raw item features reach 2.6e11
while tower embeddings are L2-normalised to ~0.1. Fed to the ranker unscaled,
its sigmoid saturated and it emitted 1.0 for every input — validation MSE stuck
at exactly `var + (1-mean)² = 0.1999`. Signed `log1p` then a z-score, applied at
write time so training, evaluation, and serving read identical values.

**Epoch counts come from validation, not from the config.** Both models track
validation loss, restore the best checkpoint, and stop early. The first
retrieval run made the point: training loss bottomed at epoch 4 and drifted
upward for 16 more, exporting embeddings from a measurably worse model.

**`IndexFlatIP`, not `IVFFlat`.** Exact search is ~1 ms at 3,327 items. An
approximate index would add tuning burden for no gain; it earns its place past
~100K items.

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
scripts/         download_kuairec.py, build_index.py, evaluate.py
docs/            system_design.md, engineering_log.md, progress.md, learning_guide.md
```

## Serving latency

Deployed on **GCP Cloud Run** (`us-central1`, 1 vCPU / 2Gi, scale-to-zero).
500 requests at concurrency 4, driven from a laptop; cold and warm reported
separately.

**Measured by Cloud Run**, `run.googleapis.com/request_latencies` — the
platform's own view, independent of any client:

| | p50 | p95 | p99 |
|---|---|---|---|
| **Cloud Run request latency** | **5.0 ms** | **9.6 ms** | 10.0 ms |

Three vantage points measure three different things, and conflating them is the
easy mistake:

| vantage | p95 | what it includes |
|---|---|---|
| app self-reported | 2.4 ms | the handler only |
| **Cloud Run platform** | **9.6 ms** | handler + framework + container ingress |
| client (laptop) | 75–105 ms | all of the above + internet round trip |

The client figure varies 40% between sessions while the platform figure is
stable, which is the whole finding: what a remote client waits for is dominated
by network, and what the service does is small and consistent. **Quote 9.6 ms** —
it is the platform's measurement of the service, not the app grading its own
homework, and not a number that moves with whichever café wifi ran the test.

That gap also explains why the TTL cache is invisible end-to-end: at ~10 ms of
service time against ~70 ms of round trip, the cache optimises a small slice of
what the user actually waits for. It still protects CPU under concurrency; it is
simply not a latency win for a remote client. A faster feed would come from
regional placement.

Locally on a dedicated core the same image serves cold p95 **4.7 ms** (server-side
1.6 ms), cold start **2.0 s**. Deployment is scripted — see
[docs/deployment.md](docs/deployment.md).

## Stack

PyTorch · FAISS · FastAPI · Docker · pandas/NumPy · pytest (157 tests)

## Next

- **Multi-task ranking head** — a ranking loss for ordering plus a regression
  head for calibrated watch-time, recovering the AUC the pairwise objective
  trades away (0.714 → 0.622) without giving back top-K recall.
- **Stronger candidate generation.** The ranker's ceiling is retrieval's
  shortlist; retrieval still leads at K=5. Hard-negative mining and multi-source
  recall (popularity + tag similarity) target that directly.
- Deploy to GCP Cloud Run and measure p50/p95 under load.
- Multi-task ranking (watch + like), then a Transformer ranker.
