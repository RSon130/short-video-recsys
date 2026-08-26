"""
Latency load test against a running /recommend endpoint.

Usage:
    python scripts/loadtest.py                                   # local compose
    python scripts/loadtest.py --url https://... --requests 500  # Cloud Run

Why this exists rather than timing a single curl: p95 is a property of a
*distribution*, so it needs a population of requests. One request gives you one
sample and no tail. A single fast call is also the easiest number to accidentally
report, and the first question it invites — "how did you measure that?" — has no
good answer.

Two regimes are measured separately, because reporting them together is
misleading:

  cold   every request uses a distinct user, so the TTL cache always misses and
         the full pipeline runs: FAISS query -> feature assembly -> ranker
         forward pass. This is the number that describes the model serving path.

  warm   requests repeat a small user set, so most hit the cache. This measures
         the cache, not the model, and will look dramatically better.

Quoting the warm number alone would overstate performance by whatever the cache
hit rate happens to be, so both are reported with the hit rate stated.
"""
import argparse
import json
import pickle
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_URL = "http://api:8000"
PROCESSED = Path("datastore/processed")


def load_user_ids(limit=None):
    """Raw dataset user ids the API will recognise (it maps raw -> internal)."""
    with open(PROCESSED / "id_maps.pkl", "rb") as f:
        id_maps = pickle.load(f)
    ids = sorted(id_maps["user_id_map"].keys())
    return ids[:limit] if limit else ids


def percentile(values, p):
    """Nearest-rank percentile; no interpolation, no numpy dependency here."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round(p / 100.0 * len(ordered) + 0.5)) - 1))
    return ordered[k]


def one_request(client, url, user_id, top_k):
    """Return (latency_ms measured client-side, server-reported latency_ms)."""
    t0 = time.perf_counter()
    response = client.post(
        f"{url}/recommend",
        json={"user_id": int(user_id), "top_k": top_k},
        timeout=30.0,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    response.raise_for_status()
    return elapsed_ms, response.json().get("latency_ms")


def run_phase(url, user_ids, top_k, concurrency):
    """Fire one request per user id, `concurrency` at a time."""
    latencies, server_latencies = [], []
    with httpx.Client() as client:
        def task(uid):
            return one_request(client, url, uid, top_k)

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            t0 = time.perf_counter()
            for elapsed_ms, server_ms in pool.map(task, user_ids):
                latencies.append(elapsed_ms)
                if server_ms is not None:
                    server_latencies.append(server_ms)
            wall = time.perf_counter() - t0

    return latencies, server_latencies, wall


def summarise(name, latencies, server_latencies, wall):
    stats = {
        "n": len(latencies),
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
        "mean_ms": statistics.fmean(latencies),
        "max_ms": max(latencies),
        "throughput_rps": len(latencies) / wall if wall else float("nan"),
    }
    if server_latencies:
        stats["server_p95_ms"] = percentile(server_latencies, 95)

    print(f"\n{name}")
    print(f"  requests      {stats['n']}")
    print(f"  p50           {stats['p50_ms']:.1f} ms")
    print(f"  p95           {stats['p95_ms']:.1f} ms")
    print(f"  p99           {stats['p99_ms']:.1f} ms")
    print(f"  mean / max    {stats['mean_ms']:.1f} / {stats['max_ms']:.1f} ms")
    print(f"  throughput    {stats['throughput_rps']:.1f} req/s "
          f"(concurrency-dependent)")
    if server_latencies:
        print(f"  server p95    {stats['server_p95_ms']:.1f} ms "
              f"(excludes network + client overhead)")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=1,
                        help="1 measures latency; raise it to measure under load")
    parser.add_argument("--warm-users", type=int, default=10,
                        help="distinct users reused in the warm phase")
    parser.add_argument("--out", default=None, help="write results as JSON")
    args = parser.parse_args()

    user_ids = load_user_ids()
    if args.requests > len(user_ids):
        raise SystemExit(
            f"--requests {args.requests} exceeds the {len(user_ids)} known users; "
            f"the cold phase needs one distinct user per request."
        )

    with httpx.Client() as client:
        client.get(f"{args.url}/health", timeout=30.0).raise_for_status()
    print(f"Target {args.url} — {len(user_ids):,} known users, "
          f"concurrency {args.concurrency}")

    cold_users = user_ids[:args.requests]
    cold = run_phase(args.url, cold_users, args.top_k, args.concurrency)
    cold_stats = summarise("COLD — distinct users, every request misses the cache",
                           *cold)

    warm_pool = user_ids[:args.warm_users]
    warm_users = [warm_pool[i % len(warm_pool)] for i in range(args.requests)]
    warm = run_phase(args.url, warm_users, args.top_k, args.concurrency)
    hit_rate = 1 - (len(warm_pool) / len(warm_users))
    warm_stats = summarise(
        f"WARM — {args.warm_users} users reused, ~{hit_rate:.0%} cache hits", *warm
    )

    print(f"\nCache speedup at p95: "
          f"{cold_stats['p95_ms'] / warm_stats['p95_ms']:.1f}x — "
          f"report the cold number as the serving latency.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"url": args.url, "concurrency": args.concurrency,
             "cold": cold_stats, "warm": warm_stats,
             "warm_cache_hit_rate": hit_rate}, indent=2))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
