"""
Benchmark exact vs approximate FAISS indexes.

Usage:
    python scripts/benchmark_index.py                     # real embeddings only
    python scripts/benchmark_index.py --scales 10000 100000 1000000
    python scripts/benchmark_index.py --out logs/index_benchmark.json

The project uses IndexFlatIP — exact, exhaustive search — and the stated reason
is "the catalogue is only ~10K items, so approximation buys nothing." That is an
assertion. This measures it, and finds the catalogue size where it stops being
true.

Four things are measured per index type:

  build time     one-off, but it gates how fast you can ship a retrained model
  index size     storage, and image size when artifacts are baked in
  query latency  p50/p95/p99 for single-query search, the serving pattern
  recall@K       agreement with exact search — the price of approximation

Recall@K is the metric that matters and the one most easily skipped. An ANN
index that returns results 20x faster while missing a third of the true top-K
has not made the system faster; it has made it worse and faster. Exact search is
the ground truth here by construction, so recall is measurable without labels.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.faiss_index import INDEX_TYPES, build_index, default_nlist, index_size_bytes

PROCESSED = Path("datastore/processed")


def percentile(values, p):
    return float(np.percentile(values, p))


def measure_latency(index, queries, top_k, repeats=3):
    """
    Single-query latency, which is what a request actually pays.

    Batched search is faster per query but is not the serving pattern: a
    /recommend call has one user. Measuring batched throughput and calling it
    latency is a common way to make an index look better than it will feel.
    """
    timings = []
    for _ in range(repeats):
        for q in queries:
            q = np.ascontiguousarray(q.reshape(1, -1).astype("float32"))
            t0 = time.perf_counter()
            index.search(q, top_k)
            timings.append((time.perf_counter() - t0) * 1000)
    return timings


def recall_against_exact(approx_index, exact_index, queries, top_k):
    """
    Fraction of the exact top-K that the approximate index also returns.

    This is recall in the ANN sense — agreement with brute force — not the
    recommender recall@K reported elsewhere in this project. They are different
    measurements that share a name, which is worth stating out loud.
    """
    queries = np.ascontiguousarray(queries.astype("float32"))
    _, exact = exact_index.search(queries, top_k)
    _, approx = approx_index.search(queries, top_k)
    overlaps = [
        len(set(e.tolist()) & set(a.tolist())) / top_k
        for e, a in zip(exact, approx)
    ]
    return float(np.mean(overlaps))


def benchmark_scale(embeddings, queries, top_k, label):
    """Run every index type over one catalogue and return a list of result dicts."""
    n_items, dim = embeddings.shape
    print(f"\n=== {label}: {n_items:,} items x {dim}d, top_k={top_k}, "
          f"{len(queries)} queries ===")

    exact_index = None
    rows = []
    for index_type in INDEX_TYPES:
        t0 = time.perf_counter()
        index = build_index(embeddings, index_type=index_type)
        build_s = time.perf_counter() - t0

        if index_type == "flat":
            exact_index = index

        timings = measure_latency(index, queries, top_k)
        recall = (1.0 if index_type == "flat"
                  else recall_against_exact(index, exact_index, queries, top_k))

        row = {
            "scale": label,
            "n_items": int(n_items),
            "index_type": index_type,
            "build_s": round(build_s, 3),
            "size_mb": round(index_size_bytes(index) / 1e6, 2),
            "p50_ms": round(percentile(timings, 50), 4),
            "p95_ms": round(percentile(timings, 95), 4),
            "p99_ms": round(percentile(timings, 99), 4),
            "qps": round(1000 / percentile(timings, 50), 1),
            "recall_at_k": round(recall, 4),
        }
        if index_type == "ivfflat":
            row["nlist"] = default_nlist(n_items)
            row["nprobe"] = min(10, row["nlist"])
        rows.append(row)

        print(f"  {index_type:8} build {build_s:7.2f}s  {row['size_mb']:7.2f}MB  "
              f"p50 {row['p50_ms']:7.4f}ms  p95 {row['p95_ms']:7.4f}ms  "
              f"recall@{top_k} {recall:.4f}")
    return rows


def sweep_knobs(embeddings, queries, top_k):
    """
    Trace the recall/latency curve for each approximate index.

    A single (speed, recall) point per index type is close to meaningless:
    both IVF and HNSW expose a knob that moves along a curve, and quoting one
    setting invites "well, tune it then". nprobe and efSearch are that knob.
    The useful output is the frontier, not a point on it.
    """
    from retrieval.faiss_index import build_index

    print(f"\n=== recall/latency frontier: {len(embeddings):,} items, "
          f"top_k={top_k} ===")
    exact = build_index(embeddings, index_type="flat")
    flat_p50 = percentile(measure_latency(exact, queries, top_k), 50)
    print(f"  exact (flat)              p50 {flat_p50:8.4f}ms  recall 1.0000")

    rows = []
    nlist = default_nlist(len(embeddings))
    for nprobe in (1, 10, 50, 100, nlist):
        idx = build_index(embeddings, index_type="ivfflat", nprobe=nprobe)
        t = percentile(measure_latency(idx, queries, top_k, repeats=2), 50)
        r = recall_against_exact(idx, exact, queries, top_k)
        rows.append({"index_type": "ivfflat", "knob": f"nprobe={nprobe}",
                     "p50_ms": round(t, 4), "recall_at_k": round(r, 4),
                     "speedup_vs_exact": round(flat_p50 / t, 2)})
        print(f"  ivfflat nprobe={nprobe:<5}      p50 {t:8.4f}ms  recall {r:.4f}"
              f"  ({flat_p50 / t:5.2f}x)")

    for ef in (64, 128, 256, 512):
        idx = build_index(embeddings, index_type="hnsw", ef_search=ef)
        t = percentile(measure_latency(idx, queries, top_k, repeats=2), 50)
        r = recall_against_exact(idx, exact, queries, top_k)
        rows.append({"index_type": "hnsw", "knob": f"efSearch={ef}",
                     "p50_ms": round(t, 4), "recall_at_k": round(r, 4),
                     "speedup_vs_exact": round(flat_p50 / t, 2)})
        print(f"  hnsw efSearch={ef:<5}      p50 {t:8.4f}ms  recall {r:.4f}"
              f"  ({flat_p50 / t:5.2f}x)")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=200,
                        help="matches retrieval.top_k_recall in production")
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--scales", type=int, nargs="*", default=[100_000, 1_000_000],
                        help="synthetic catalogue sizes to test beyond the real one")
    parser.add_argument("--out", default="logs/index_benchmark.json")
    parser.add_argument("--sweep-scale", type=int, default=None,
                        help="also trace the recall/latency frontier at this size")
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    results = []

    # Real embeddings first — the actual catalogue this system serves.
    item_path = PROCESSED / "item_embeddings.npy"
    if item_path.exists():
        items = np.load(item_path)
        users = np.load(PROCESSED / "user_embeddings.npy")
        queries = users[rng.choice(len(users), min(args.queries, len(users)),
                                   replace=False)]
        results += benchmark_scale(items, queries, args.top_k, "real (KuaiRec)")
    else:
        print(f"No {item_path}; skipping the real catalogue.")

    # Synthetic catalogues to find where approximation starts paying. Random
    # unit vectors are harder for ANN than real embeddings — real data clusters,
    # which IVF and HNSW both exploit — so these recall figures are pessimistic.
    dim = 64
    for n in args.scales:
        emb = rng.normal(size=(n, dim)).astype("float32")
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        q = rng.normal(size=(args.queries, dim)).astype("float32")
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        results += benchmark_scale(emb, q, args.top_k, f"synthetic {n:,}")

    sweep = []
    if args.sweep_scale:
        n = args.sweep_scale
        emb = rng.normal(size=(n, dim)).astype("float32")
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        q = rng.normal(size=(args.queries, dim)).astype("float32")
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        sweep = sweep_knobs(emb, q, args.top_k)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"scales": results, "frontier": sweep}, indent=2))
    print(f"\nWrote {out}")

    # The comparison the project actually needs: does approximation pay here?
    print("\n=== speedup vs exact, per scale ===")
    for scale in dict.fromkeys(r["scale"] for r in results):
        rows = [r for r in results if r["scale"] == scale]
        flat = next(r for r in rows if r["index_type"] == "flat")
        for r in rows:
            if r["index_type"] == "flat":
                continue
            speedup = flat["p50_ms"] / r["p50_ms"] if r["p50_ms"] else float("nan")
            print(f"  {scale:20} {r['index_type']:8} "
                  f"{speedup:6.2f}x faster   recall {r['recall_at_k']:.4f}   "
                  f"{r['size_mb'] / flat['size_mb']:5.2f}x the storage")


if __name__ == "__main__":
    main()
