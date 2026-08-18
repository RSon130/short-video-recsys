"""
Offline evaluation: compare popularity, retrieval-only, and the full pipeline.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --config config/kuairec.yaml --k 10 20

Every number this project reports is stated against a baseline. Three systems
are evaluated on the same temporal test split:

    popularity      most-engaged items, identical for every user
    retrieval-only  two-tower + FAISS, ranked by cosine similarity
    full pipeline   retrieval candidates re-scored by the MLP ranker

The popularity-vs-retrieval gap shows that personalisation is working; the
retrieval-vs-full gap shows what the second stage buys, which is the whole
argument for a two-stage architecture.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from config_loader import load_config
from data.schema import Cols
from evaluation.baselines import popularity_ranking, popularity_recommendations
from evaluation.metrics import evaluate_at_k_values, watch_time_auc
from features.dense_features import DenseFeatureStore
from models.ranker import build_ranker
from retrieval.faiss_index import load_index, query_index

PROCESSED = Path("datastore/processed")


def build_ground_truth(test_df: pd.DataFrame, positive_threshold: float) -> dict:
    """
    Map each user to the items they genuinely engaged with in the test window.

    Relevance is defined by watch_ratio, not by the presence of a row. On a
    fully-observed matrix nearly every (user, item) pair has a row, so treating
    "appears in the test split" as relevant would measure *exposure* rather than
    preference and would inflate every metric. The threshold matches the one
    used to define positives in training.

    Args:
        test_df:            Test-split interactions.
        positive_threshold: Minimum watch_ratio to count as relevant.

    Returns:
        {user_id: [item_id, ...]} for users with at least one relevant item.
    """
    relevant = test_df.loc[test_df[Cols.WATCH_RATIO] >= positive_threshold]
    return {
        int(uid): group[Cols.ITEM_ID].tolist()
        for uid, group in relevant.groupby(Cols.USER_ID)
    }


def score_systems(cfg, users, ground_truth, top_k_final):
    """
    Generate per-user recommendation lists for all three systems.

    Returns:
        (recommendations, ab_scores) where recommendations maps a system name to
        a list of per-user item lists, and ab_scores carries the raw retrieval
        and ranker scores needed for the A/B comparison.
    """
    positive_threshold = cfg["features"]["positive_watch_ratio"]
    top_k_recall = cfg["retrieval"]["top_k_recall"]

    user_embs = np.load(PROCESSED / "user_embeddings.npy")
    item_embs = np.load(PROCESSED / "item_embeddings.npy")
    index = load_index(cfg["retrieval"]["index_path"])

    features = DenseFeatureStore.load()
    ranker = build_ranker(cfg, features.user_dense_dim, features.item_dense_dim)
    ranker.load_state_dict(
        torch.load(PROCESSED / "ranker_model.pt", map_location="cpu", weights_only=True)
    )
    ranker.eval()

    train_df = pd.read_parquet(PROCESSED / "interactions" / "train.parquet")
    ranking = popularity_ranking(train_df, positive_threshold)
    seen = train_df.groupby(Cols.USER_ID)[Cols.ITEM_ID].apply(set).to_dict()

    # Every system must exclude items the user already consumed in training,
    # or the comparison is meaningless. On this dataset the effect is not
    # marginal: users have seen 2,651 of 3,327 items on average, and the
    # temporal split makes train and test pairs disjoint, so a system that
    # re-recommends seen items cannot score above zero by construction.
    # Filtering only the baseline (as this script first did) handed popularity
    # the entire advantage.
    n_items = len(item_embs)

    recommendations = {
        "popularity": popularity_recommendations(ranking, users, top_k_final, seen),
        "retrieval-only": [],
        "full pipeline": [],
    }
    retrieval_rows, ranker_rows = [], []

    print(f"Scoring {len(users):,} users ...")
    for uid in users:
        u_emb = user_embs[uid]
        # Over-fetch, then drop seen items and keep the top_k_recall survivors.
        # Exact search over the whole catalogue is ~1 ms at this item count; a
        # production system would over-fetch by a fixed factor instead.
        all_scores, all_candidates = query_index(index, u_emb, top_k=n_items)
        excluded = seen.get(uid, set())
        kept = [(i, s) for i, s in zip(all_candidates, all_scores)
                if int(i) not in excluded][:top_k_recall]
        candidates = np.array([i for i, _ in kept], dtype=np.int64)
        ret_scores = np.array([s for _, s in kept], dtype=np.float32)

        x = torch.from_numpy(features.build_batch(u_emb, item_embs, uid, candidates))
        with torch.no_grad():
            rank_scores = ranker.predict(x).squeeze(-1).numpy()

        by_retrieval = sorted(
            zip(candidates, ret_scores), key=lambda t: t[1], reverse=True
        )
        by_ranker = sorted(
            zip(candidates, rank_scores), key=lambda t: t[1], reverse=True
        )
        recommendations["retrieval-only"].append(
            [int(i) for i, _ in by_retrieval[:top_k_final]]
        )
        recommendations["full pipeline"].append(
            [int(i) for i, _ in by_ranker[:top_k_final]]
        )

        for iid, ret_s, rank_s in zip(candidates, ret_scores, rank_scores):
            retrieval_rows.append((uid, int(iid), float(ret_s)))
            ranker_rows.append((uid, int(iid), float(rank_s)))

    return recommendations, (retrieval_rows, ranker_rows)


def report(results: dict, k_values: list) -> None:
    """Print a comparison table and the lift of each system over popularity."""
    systems = list(results.keys())
    metrics = [f"{m}@{k}" for k in k_values for m in ("recall", "ndcg")]

    width = max(len(s) for s in systems) + 2
    header = "system".ljust(width) + "".join(m.rjust(12) for m in metrics)
    print("\n" + header)
    print("-" * len(header))
    for system in systems:
        row = system.ljust(width)
        row += "".join(f"{results[system][m]:12.4f}" for m in metrics)
        print(row)

    base = results.get("popularity")
    if not base:
        return
    print("\nLift over popularity baseline:")
    for system in systems:
        if system == "popularity":
            continue
        parts = []
        for m in metrics:
            if base[m] > 0:
                parts.append(f"{m} {100 * (results[system][m] / base[m] - 1):+.1f}%")
        print(f"  {system}: " + ", ".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/kuairec.yaml")
    parser.add_argument("--k", nargs="+", type=int, default=None)
    parser.add_argument("--limit-users", type=int, default=None,
                        help="evaluate on a sample of users (for a fast check)")
    parser.add_argument("--top-k-recall", type=int, default=None,
                        help="override how many candidates retrieval passes on")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.top_k_recall:
        cfg["retrieval"]["top_k_recall"] = args.top_k_recall
    k_values = args.k or cfg["evaluation"]["k_values"]
    positive_threshold = cfg["features"]["positive_watch_ratio"]

    test_df = pd.read_parquet(PROCESSED / "interactions" / "test.parquet")
    ground_truth = build_ground_truth(test_df, positive_threshold)
    users = sorted(ground_truth.keys())
    if args.limit_users:
        users = users[: args.limit_users]
        ground_truth = {u: ground_truth[u] for u in users}

    print(f"Test users with >=1 relevant item (watch_ratio >= {positive_threshold}): "
          f"{len(users):,}")
    print(f"Mean relevant items per user: "
          f"{np.mean([len(v) for v in ground_truth.values()]):.1f}")

    top_k_final = max(k_values)
    recommendations, (retrieval_rows, ranker_rows) = score_systems(
        cfg, users, ground_truth, top_k_final
    )

    gt_lists = [ground_truth[u] for u in users]
    results = {
        system: evaluate_at_k_values(recs, gt_lists, k_values)
        for system, recs in recommendations.items()
    }
    report(results, k_values)

    # Watch-time AUC: does each scorer order candidates by actual engagement?
    truth = test_df.set_index([Cols.USER_ID, Cols.ITEM_ID])[Cols.WATCH_RATIO]
    auc = {}
    for name, rows in (("retrieval-only", retrieval_rows), ("full pipeline", ranker_rows)):
        pairs = pd.DataFrame(rows, columns=[Cols.USER_ID, Cols.ITEM_ID, "score"])
        pairs = pairs.join(truth, on=[Cols.USER_ID, Cols.ITEM_ID], how="inner").dropna()
        if len(pairs):
            auc[name] = watch_time_auc(
                pairs[Cols.WATCH_RATIO].to_numpy(), pairs["score"].to_numpy()
            )
    if auc:
        print("\nWatch-time AUC (candidates present in the test split):")
        for name, value in auc.items():
            print(f"  {name}: {value:.4f}")

    out = PROCESSED / "evaluation_results.json"
    out.write_text(json.dumps(
        {"metrics": results, "watch_time_auc": auc, "n_users": len(users)}, indent=2
    ))
    print(f"\nWrote {out}")
    return results


if __name__ == "__main__":
    main()
