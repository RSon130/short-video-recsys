# Exact vs approximate retrieval — measured

This project uses `IndexFlatIP`, exact exhaustive search, and the stated reason
was "the catalogue is only ~10K items, so approximation buys nothing." That was
an assertion. This is the measurement, plus the catalogue size at which it stops
being true.

Reproduce with:

```bash
docker compose -f docker-compose.cpu.yml run --rm dev python scripts/benchmark_index.py --scales 1000000 --sweep-scale 100000
```

`top_k=200` throughout, matching `retrieval.top_k_recall`. Latency is
**single-query**, because that is the serving pattern — one `/recommend` call
has one user. Batched search looks considerably better and would be the wrong
number to quote.

**"Recall" here means agreement with brute force** — the fraction of the exact
top-200 that the approximate index also returned. It is not the recommender
recall@K reported elsewhere in this repo. Two different measurements share a
name, which is worth saying out loud before showing a table of them.

---

## The real catalogue: 9,958 items

| index | build | size | p50 | p95 | recall@200 |
|---|---|---|---|---|---|
| **flat (exact)** | 0.00 s | 2.55 MB | **0.087 ms** | 0.114 ms | **1.0000** |
| ivfflat | 0.03 s | 2.66 MB | 0.036 ms | 0.046 ms | 0.9051 |
| hnsw | 0.07 s | 5.26 MB | 0.045 ms | 0.061 ms | 0.7683 |

ANN is 2.4× faster. **It is also irrelevant**, and that is the finding.

Exact search costs 0.087 ms against a measured end-to-end p95 of 9.6 ms. The
2.4× speedup saves **0.05 ms — about 0.5% of the request** — in exchange for
losing 10% of the true neighbours (IVF) or 23% (HNSW). HNSW additionally costs
2× the storage.

Trading 10% of retrieval quality for half a percent of latency is a bad trade at
any price. The original decision was right, and now it is right *for a measured
reason* rather than an intuition.

---

## Where that flips: 1,000,000 items

| index | build | size | p50 | p95 | recall@200 |
|---|---|---|---|---|---|
| flat (exact) | 0.09 s | 256 MB | **9.19 ms** | 17.78 ms | 1.0000 |
| ivfflat | 4.24 s | 264 MB | 0.21 ms | 0.28 ms | 0.1560 |
| hnsw | **224 s** | 528 MB | 0.35 ms | 0.51 ms | 0.2820 |

Exact search is linear in catalogue size, and the measurements confirm it almost
exactly: **0.087 ms at 10K → 0.884 ms at 100K → 9.19 ms at 1M**, i.e. about
0.0092 ms per thousand items.

So at 1M, exact retrieval alone costs as much as the entire current request. The
crossover — where retrieval stops being free and starts being the budget — is
somewhere around **300K–500K items** for this latency target. Below it,
approximation is premature optimisation; above it, it is mandatory.

Note also the build cost: HNSW takes **224 s at 1M against 0.09 s for flat**,
2,500× slower. That is not a serving cost but it gates how quickly a retrained
model can ship, which matters more than it first appears.

---

## The part that actually matters: recall is a knob

Quoting one (speed, recall) point per index type is close to meaningless, because
both expose a parameter that slides along a curve. Here is the frontier at
100,000 items, where exact search costs 0.884 ms:

| index | setting | p50 | recall@200 | speedup |
|---|---|---|---|---|
| ivfflat | nprobe=1 | 0.034 ms | 0.0362 | 26.3× |
| ivfflat | nprobe=10 | 0.104 ms | 0.2090 | 8.5× |
| ivfflat | nprobe=50 | 0.283 ms | 0.5637 | 3.1× |
| ivfflat | nprobe=100 | 0.522 ms | 0.7718 | 1.7× |
| ivfflat | nprobe=316 (all cells) | 1.486 ms | 1.0000 | **0.6×** |
| hnsw | efSearch=64 | 0.269 ms | 0.5426 | 3.3× |
| hnsw | efSearch=128 | 0.490 ms | 0.7452 | 1.8× |
| hnsw | efSearch=256 | 0.691 ms | 0.9075 | 1.3× |
| hnsw | efSearch=512 | 1.347 ms | 0.9834 | 0.7× |

Three things fall out of this table:

**The headline speedup evaporates when you demand quality.** "26× faster" is
true at recall 0.036, which returns essentially none of the right items. To reach
recall ≥ 0.9 at this scale the best option is HNSW at efSearch=256, and that is
only **1.3× faster than brute force**.

**An exhaustive IVF is slower than flat.** At nprobe=nlist the index scans every
cell and pays the partitioning overhead on top — 0.6× the speed of exact search
for identical results. Approximate structures are not free when you stop
approximating.

**HNSW dominates IVF on the high-recall end.** IVF wins when low recall is
acceptable; above about 0.75 the curves cross and HNSW gives more recall per
millisecond. Which index is "better" depends entirely on the operating point.

---

## Real data is easier than random data

At `nprobe=10`, IVF scores **recall 0.905 on the real catalogue** but only
**0.209 on synthetic vectors of the same dimensionality**. Same algorithm, same
parameters, 4× the quality.

The synthetic vectors are uniform on the unit sphere, which in 64 dimensions is
close to the worst case for any partitioning scheme — every point is roughly
equidistant from every other, so cells carry little information. Real embeddings
cluster, and both IVF and HNSW exploit exactly that structure.

The practical consequence: **ANN benchmarks on synthetic data are pessimistic
and do not transfer.** The numbers above at 100K and 1M establish the *shape* of
the trade-off, not the recall a real catalogue of that size would achieve.

---

## Decision

`IndexFlatIP` stays the default. At 9,958 items it is exact, 0.087 ms, and 0.5%
of a request.

Revisit at roughly 500K items, and when doing so tune to a recall target rather
than a speedup target — pick the operating point first, then choose the index
that reaches it fastest. On the evidence above that is likely to be HNSW, paying
2× storage and a much slower build for the recall.

Both alternatives remain implemented and selectable via `build_index(...,
index_type=...)`, so this is a reproducible comparison rather than a claim.
