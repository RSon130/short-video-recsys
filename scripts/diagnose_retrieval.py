"""
Diagnose the retrieval stage — the ceiling on everything the ranker can do.

Read-only: uses the artifacts on disk, trains nothing.

Usage:
    python scripts/diagnose_retrieval.py [--users N]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from config_loader import load_config
from data.schema import Cols

P = Path("datastore/processed")
RNG = np.random.default_rng(0)


def section(title):
    print(f"\n=== {title} ===", flush=True)


def topk_unseen(scores, seen, k):
    order = np.argsort(-scores)
    return np.array([i for i in order[: k + len(seen) + 1] if i not in seen][:k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=3000)
    args = ap.parse_args()
    cfg = load_config("config/kuairec.yaml")
    pos_t = cfg["features"]["positive_watch_ratio"]
    cols = [Cols.USER_ID, Cols.ITEM_ID, Cols.WATCH_RATIO]
    train = pd.read_parquet(P / "interactions/train.parquet", columns=cols)
    val = pd.read_parquet(P / "interactions/val.parquet", columns=cols)
    test = pd.read_parquet(P / "interactions/test.parquet", columns=cols)
    U = np.load(P / "user_embeddings.npy")
    V = np.load(P / "item_embeddings.npy")
    n_users, n_items = len(U), len(V)

    seen_tr = train.groupby(Cols.USER_ID)[Cols.ITEM_ID].apply(set).to_dict()
    pop = np.bincount(train.loc[train[Cols.WATCH_RATIO] >= pos_t, Cols.ITEM_ID], minlength=n_items).astype(float)

    # ------------------------------------------------------------------
    section("R0. Embedding geometry")
    mean_u = U.mean(0)
    print(f"user vectors: norm of mean {np.linalg.norm(mean_u):.3f} (0 = spread, 1 = collapsed)")
    sub = U[RNG.choice(n_users, 1000, replace=False)]
    cos = sub @ sub.T
    print(f"user-user cosine: mean {cos[np.triu_indices(1000, 1)].mean():.3f}")
    sv = np.linalg.svd(U - mean_u, compute_uv=False)
    print(f"user effective rank (entropy of sv^2): {np.exp(-(p := sv**2 / (sv**2).sum()) @ np.log(p + 1e-12)):.1f} of {U.shape[1]}")
    item_align = V @ (mean_u / np.linalg.norm(mean_u))
    print(f"corr(item alignment with mean user, log popularity) = {np.corrcoef(item_align, np.log1p(pop))[0, 1]:+.3f}")

    # ------------------------------------------------------------------
    section("R1. Ground-truth reachability (seen-item exclusion)")
    for name, df in (("val", val), ("test", test)):
        pos = df.loc[df[Cols.WATCH_RATIO] >= pos_t, [Cols.USER_ID, Cols.ITEM_ID]]
        rewatch = np.array([i in seen_tr.get(u, ()) for u, i in zip(pos[Cols.USER_ID], pos[Cols.ITEM_ID])])
        print(f"{name}: {len(pos):,} positive rows, {rewatch.mean():.1%} are items the user already saw in train "
              f"(excluded from every system's candidates, still in the denominator)")

    # ------------------------------------------------------------------
    def evaluate(split_df, label, users_n):
        gt = split_df.loc[split_df[Cols.WATCH_RATIO] >= pos_t].groupby(Cols.USER_ID)[Cols.ITEM_ID].apply(set).to_dict()
        users = np.sort(RNG.choice(np.array(sorted(gt)), min(users_n, len(gt)), replace=False))
        systems = {"retrieval": lambda u: V @ U[u],
                   "global mean user (non-personal)": lambda u: V @ mean_u,
                   "popularity": lambda u: pop}
        ks = (10, 50, 200, 1000)
        res = {s: {k: [] for k in ks} for s in systems}
        aucs = {s: [] for s in systems}
        tops = {s: [] for s in systems}
        for u in users:
            seen = seen_tr.get(u, set())
            truth = gt[u] - seen
            n_rel_all = len(gt[u])
            unseen = np.array([i for i in range(n_items) if i not in seen])
            y = np.isin(unseen, list(truth))
            for s, f in systems.items():
                sc = f(u)
                order = unseen[np.argsort(-sc[unseen])]
                for k in ks:
                    res[s][k].append(len(set(order[:k]) & gt[u]) / min(k, n_rel_all))
                if 0 < y.sum() < len(y):
                    aucs[s].append(roc_auc_score(y, sc[unseen]))
                tops[s].append(set(order[:200]))
        section(f"R2. {label}: {len(users):,} users — recall@K over the full unseen catalogue")
        print("system".ljust(34) + "".join(f"recall@{k}".rjust(12) for k in ks) + "  full-catalogue AUC")
        for s in systems:
            print(s.ljust(34) + "".join(f"{np.mean(res[s][k]):12.4f}" for k in ks) + f"  {np.mean(aucs[s]):.4f}")
        idx = RNG.choice(len(users), 300, replace=False)
        jac = [len(tops["retrieval"][a] & tops["retrieval"][b]) / len(tops["retrieval"][a] | tops["retrieval"][b])
               for a, b in zip(idx[:150], idx[150:])]
        distinct = len(set().union(*[set(list(t)[:200]) for t in tops["retrieval"]]))
        print(f"retrieval top-200 Jaccard between random user pairs: {np.mean(jac):.3f}; "
              f"distinct items across all users' top-200: {distinct:,} of {n_items:,}")
        top_pop = [np.log1p(pop[list(t)]).mean() for t in tops["retrieval"]]
        print(f"mean log-popularity: retrieval top-200 {np.mean(top_pop):.2f} vs catalogue {np.log1p(pop).mean():.2f}")

    evaluate(val, "VALIDATION period", args.users)
    evaluate(test, "TEST period", args.users)

    # ------------------------------------------------------------------
    section("R3. Temporal drift in item popularity")
    pop_val = np.bincount(val.loc[val[Cols.WATCH_RATIO] >= pos_t, Cols.ITEM_ID], minlength=n_items)
    pop_test = np.bincount(test.loc[test[Cols.WATCH_RATIO] >= pos_t, Cols.ITEM_ID], minlength=n_items)
    from scipy.stats import spearmanr
    print(f"Spearman(train pop, val pop) {spearmanr(pop, pop_val).correlation:.3f} | "
          f"(train pop, test pop) {spearmanr(pop, pop_test).correlation:.3f}")
    new_in_test = (pop == 0) & (pop_test > 0)
    print(f"items with test positives but zero train positives: {new_in_test.sum():,} "
          f"({pop_test[new_in_test].sum() / pop_test.sum():.1%} of test positives)")
    print(f"users in test: {test[Cols.USER_ID].nunique():,}; test positives per user mean "
          f"{test.loc[test[Cols.WATCH_RATIO] >= pos_t].groupby(Cols.USER_ID).size().mean():.1f}")


if __name__ == "__main__":
    main()
