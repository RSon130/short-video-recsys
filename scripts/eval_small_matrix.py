"""
Evaluate the current models on KuaiRec's fully-observed small_matrix.

Why: the big_matrix temporal hold-out only contains items the platform chose to
show, so offline recall rewards predicting exposure. small_matrix is (nearly)
fully observed — every one of its 1,411 users watched essentially all 3,327 of
its videos — and shares no (user, video) pair with big_matrix, while all its
users and videos appear in big_matrix. Models trained on big_matrix can be
scored on it with no exposure selection.

Read-only; trains nothing. Reports two relevance definitions:
    raw       watch_ratio >= 0.7 (the training label)
    debiased  top 30% of watch_ratio within the item's duration decile
              (removes "short videos get finished"; see docs/label_design.md)
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import load_config
from data.schema import Cols
from evaluation.ab_test import paired_bootstrap
from features.dense_features import DenseFeatureStore
from models.ranker import build_ranker

P = Path("datastore/processed")


def ndcg_at(order_rel, k=10):
    gains = order_rel[:k]
    dcg = (gains / np.log2(np.arange(2, len(gains) + 2))).sum()
    ideal = np.sort(order_rel)[::-1][:k]
    idcg = (ideal / np.log2(np.arange(2, len(ideal) + 2))).sum()
    return dcg / idcg if idcg > 0 else 0.0


def main():
    cfg = load_config("config/kuairec.yaml")
    maps = pickle.load(open(P / "id_maps.pkl", "rb"))
    umap, imap = maps["user_id_map"], maps["item_id_map"]
    raw = pd.read_csv("datastore/raw/kuairec/small_matrix.csv",
                      usecols=["user_id", "video_id", "watch_ratio", "video_duration"])
    raw = raw.groupby(["user_id", "video_id"], as_index=False).agg(
        watch_ratio=("watch_ratio", "max"), video_duration=("video_duration", "first"))
    raw["u"] = raw["user_id"].map(umap)
    raw["i"] = raw["video_id"].map(imap)
    print(f"small_matrix pairs {len(raw):,}; users mapped {raw['u'].notna().mean():.1%}, "
          f"items mapped {raw['i'].notna().mean():.1%}")
    df = raw.dropna(subset=["u", "i"]).astype({"u": int, "i": int})

    # Debiased relevance: rank within duration decile (deciles over items).
    item_dur = df.groupby("i")["video_duration"].first()
    dec = pd.qcut(item_dur.rank(method="first"), 10, labels=False)
    df["dec"] = df["i"].map(dec)
    df["q"] = df.groupby("dec")["watch_ratio"].rank(pct=True)
    df["rel_raw"] = (df["watch_ratio"] >= 0.7).astype(int)
    df["rel_deb"] = (df["q"] >= 0.7).astype(int)
    print(f"positive rate raw {df['rel_raw'].mean():.1%}, debiased {df['rel_deb'].mean():.1%}; "
          f"corr(duration, rel) raw {np.corrcoef(df['video_duration'], df['rel_raw'])[0, 1]:+.3f}, "
          f"debiased {np.corrcoef(df['video_duration'], df['rel_deb'])[0, 1]:+.3f}")

    U, V = np.load(P / "user_embeddings.npy"), np.load(P / "item_embeddings.npy")
    feats = DenseFeatureStore.load()
    ranker = build_ranker(cfg, feats.user_dense_dim, feats.item_dense_dim)
    ranker.load_state_dict(torch.load(P / "ranker_model.pt", map_location="cpu", weights_only=True))
    ranker.eval()
    train = pd.read_parquet(P / "interactions/train.parquet", columns=[Cols.ITEM_ID, Cols.WATCH_RATIO])
    pop = np.bincount(train.loc[train[Cols.WATCH_RATIO] >= 0.7, Cols.ITEM_ID], minlength=len(V)).astype(float)
    dur = np.full(len(V), np.nan); dur[item_dur.index] = item_dur.values

    systems = ["popularity", "shortest-first", "retrieval (all items)", "ranker (all items)",
               "two-stage: retrieval top-200 -> ranker", "retrieval top-200 order"]
    res = {lab: {s: {"p10": [], "ndcg10": [], "auc": []} for s in systems} for lab in ("raw", "deb")}

    for u, g in df.groupby("u"):
        items = g["i"].to_numpy()
        s_ret = V[items] @ U[u]
        x = torch.from_numpy(feats.build_batch(U[u], V, u, items))
        with torch.no_grad():
            s_rnk = ranker(x).squeeze(1).numpy()
        top200 = np.argsort(-s_ret)[:200]
        two_stage = np.full(len(items), -np.inf); two_stage[top200] = s_rnk[top200]
        ret200 = np.full(len(items), -np.inf); ret200[top200] = s_ret[top200]
        scores = {"popularity": pop[items], "shortest-first": -dur[items],
                  "retrieval (all items)": s_ret, "ranker (all items)": s_rnk,
                  "two-stage: retrieval top-200 -> ranker": two_stage,
                  "retrieval top-200 order": ret200}
        for lab, col in (("raw", "rel_raw"), ("deb", "rel_deb")):
            rel = g[col].to_numpy()
            if rel.sum() == 0 or rel.sum() == len(rel):
                continue
            for s, sc in scores.items():
                order = np.argsort(-sc, kind="stable")
                res[lab][s]["p10"].append(rel[order[:10]].mean())
                res[lab][s]["ndcg10"].append(ndcg_at(rel[order].astype(float)))
                finite = np.isfinite(sc)
                res[lab][s]["auc"].append(roc_auc_score(rel, np.where(finite, sc, sc[finite].min() - 1)))

    for lab, title in (("raw", "RAW relevance (watch_ratio >= 0.7)"),
                       ("deb", "DURATION-DEBIASED relevance (top 30% within duration decile)")):
        n = len(res[lab]["popularity"]["p10"])
        print(f"\n=== {title}: {n:,} users ===")
        print("system".ljust(42) + "precision@10   ndcg@10   per-user AUC")
        for s in systems:
            r = res[lab][s]
            print(s.ljust(42) + f"{np.mean(r['p10']):12.4f}{np.mean(r['ndcg10']):10.4f}{np.mean(r['auc']):14.4f}")
        for a, b in (("retrieval (all items)", "ranker (all items)"),
                     ("retrieval top-200 order", "two-stage: retrieval top-200 -> ranker"),
                     ("popularity", "retrieval (all items)"),
                     ("shortest-first", "ranker (all items)")):
            bt = paired_bootstrap(np.array(res[lab][a]["p10"]), np.array(res[lab][b]["p10"]))
            print(f"  P@10 {b} vs {a}: {bt['relative_lift']:+.1%} "
                  f"[{bt['ci_low']:+.4f}, {bt['ci_high']:+.4f}] {'significant' if bt['significant'] else 'NOT significant'}")


if __name__ == "__main__":
    main()
