"""
Offline evaluation: Recall@K, NDCG@K on test split + A/B test.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --config config/kuairec.yaml --k 10 20
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.schema import Cols
from evaluation.metrics import evaluate_at_k_values, watch_time_auc
from evaluation.ab_test import run_ab_test
from features.engineer import load_config
from models.ranker import build_ranker
from retrieval.faiss_index import load_index, query_index


def build_ground_truth(test_df: pd.DataFrame) -> dict:
    """Group test items by user -> list of item_ids."""
    gt = {}
    for uid, group in test_df.groupby(Cols.USER_ID):
        gt[int(uid)] = group[Cols.ITEM_ID].tolist()
    return gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/kuairec.yaml")
    parser.add_argument("--k", nargs="+", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    k_values = args.k or cfg["evaluation"]["k_values"]

    # ------------------------------------------------------------------
    # Load artifacts
    # ------------------------------------------------------------------
    with open("data/processed/id_maps.pkl", "rb") as f:
        id_maps = pickle.load(f)

    user_embs = np.load("data/processed/user_embeddings.npy")
    item_embs = np.load("data/processed/item_embeddings.npy")
    index = load_index(cfg["retrieval"]["index_path"])

    user_dense_dim = id_maps.get("user_dense_dim", cfg["features"]["user_dense_dim"])
    item_dense_dim = id_maps.get("item_dense_dim", cfg["features"]["item_dense_dim"])
    ranker = build_ranker(cfg, user_dense_dim, item_dense_dim)
    ranker.load_state_dict(
        torch.load("data/processed/ranker_model.pt", map_location="cpu", weights_only=True)
    )
    ranker.eval()

    test_df = pd.read_parquet("data/processed/interactions/test.parquet")
    ground_truth = build_ground_truth(test_df)
    users = list(ground_truth.keys())

    top_k_recall = cfg["retrieval"]["top_k_recall"]
    top_k_final = max(k_values)

    # ------------------------------------------------------------------
    # Per-user retrieval + ranking
    # ------------------------------------------------------------------
    user_dense = np.zeros(user_dense_dim, dtype=np.float32)

    recommended_lists = []
    retrieval_score_rows = []   # for A/B: (uid, iid, retrieval_score)
    ranker_score_rows = []      # for A/B: (uid, iid, ranker_score)

    print(f"Evaluating {len(users)} users ...")
    for uid in users:
        u_emb = user_embs[uid]
        scores, indices = query_index(index, u_emb, top_k=top_k_recall)

        # Ranker re-scoring
        ranked = []
        for iid, ret_score in zip(indices, scores):
            i_emb = item_embs[iid]
            item_dense = np.zeros(item_dense_dim, dtype=np.float32)
            x = torch.from_numpy(
                np.concatenate([u_emb, i_emb, user_dense, item_dense]).astype(np.float32)
            ).unsqueeze(0)
            with torch.no_grad():
                rank_score = float(ranker(x).item())
            ranked.append((iid, ret_score, rank_score))

        ranked.sort(key=lambda t: t[2], reverse=True)
        recommended_lists.append([r[0] for r in ranked[:top_k_final]])

        for iid, ret_s, rank_s in ranked:
            retrieval_score_rows.append((uid, iid, ret_s))
            ranker_score_rows.append((uid, iid, rank_s))

    # ------------------------------------------------------------------
    # Recall@K / NDCG@K
    # ------------------------------------------------------------------
    gt_lists = [ground_truth[uid] for uid in users]
    metrics = evaluate_at_k_values(recommended_lists, gt_lists, k_values)

    print("\n=== Offline Metrics (test split) ===")
    for key, val in sorted(metrics.items()):
        print(f"  {key}: {val:.4f}")

    # ------------------------------------------------------------------
    # A/B test — group A: retrieval score, group B: ranker score
    # ------------------------------------------------------------------
    ret_df = pd.DataFrame(retrieval_score_rows, columns=[Cols.USER_ID, Cols.ITEM_ID, "score"])
    rank_df = pd.DataFrame(ranker_score_rows, columns=[Cols.USER_ID, Cols.ITEM_ID, "score"])

    test_merged = test_df[[Cols.USER_ID, Cols.ITEM_ID, Cols.WATCH_RATIO]]
    test_with_ret = test_merged.merge(ret_df, on=[Cols.USER_ID, Cols.ITEM_ID], how="left").fillna(0)
    test_with_rank = test_merged.merge(rank_df, on=[Cols.USER_ID, Cols.ITEM_ID], how="left").fillna(0)

    ab_interactions = test_with_ret[[Cols.USER_ID, Cols.WATCH_RATIO]].copy()
    model_a_scores = test_with_ret["score"].values
    model_b_scores = test_with_rank["score"].values

    ab_results = run_ab_test(ab_interactions, model_a_scores, model_b_scores, cfg)
    print("\n=== A/B Test (Group A = retrieval | Group B = ranker) ===")
    for group, stats in ab_results.items():
        print(f"  {group}: n_users={stats['n_users']}, watch_time_auc={stats['watch_time_auc']:.4f}")

    print("\nDone.")
    return metrics, ab_results


if __name__ == "__main__":
    main()
