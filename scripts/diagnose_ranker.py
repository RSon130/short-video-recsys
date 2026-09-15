"""
Diagnose why the full pipeline scores below retrieval alone (engineering log §14).

Read-only: uses the artifacts on disk, trains nothing. Every check is framed
as a hypothesis with the measurement that would support or rule it out.

Usage:
    python scripts/diagnose_ranker.py [--users N]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from config_loader import load_config
from data.schema import Cols
from features.dense_features import DenseFeatureStore
from models.ranker import build_ranker

P = Path("datastore/processed")
RNG = np.random.default_rng(0)
OUT = {}


def section(title):
    print(f"\n=== {title} ===", flush=True)


def recall_at(lists, truth, k=10):
    vals = [len(set(l[:k]) & t) / min(k, len(t)) for l, t in zip(lists, truth)]
    return float(np.mean(vals)), np.array(vals)


def per_user_spearman(a_rows, b_rows):
    vals = []
    for a, b in zip(a_rows, b_rows):
        if np.std(a) > 0 and np.std(b) > 0:
            vals.append(spearmanr(a, b).correlation)
    return float(np.nanmean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=3000)
    args = ap.parse_args()
    cfg = load_config("config/kuairec.yaml")
    pos_t = cfg["features"]["positive_watch_ratio"]
    neg_t = cfg["features"]["negative_watch_ratio"]
    top_k_recall = cfg["retrieval"]["top_k_recall"]

    train = pd.read_parquet(P / "interactions/train.parquet",
                            columns=[Cols.USER_ID, Cols.ITEM_ID, Cols.WATCH_RATIO, "video_duration", "timestamp"])
    val = pd.read_parquet(P / "interactions/val.parquet",
                          columns=[Cols.USER_ID, Cols.ITEM_ID, Cols.WATCH_RATIO, "timestamp"])
    test = pd.read_parquet(P / "interactions/test.parquet",
                           columns=[Cols.USER_ID, Cols.ITEM_ID, Cols.WATCH_RATIO, "timestamp"])
    U = np.load(P / "user_embeddings.npy")
    V = np.load(P / "item_embeddings.npy")
    n_users, n_items = len(U), len(V)
    feats = DenseFeatureStore.load()
    ranker = build_ranker(cfg, feats.user_dense_dim, feats.item_dense_dim)
    ranker.load_state_dict(torch.load(P / "ranker_model.pt", map_location="cpu", weights_only=True))
    ranker.eval()

    def rank_score(u_embs, v_embs, uids, iids):
        x = torch.from_numpy(feats.build_matrix(u_embs, v_embs, uids, iids))
        with torch.no_grad():
            return ranker(x).squeeze(1).numpy()

    # ------------------------------------------------------------------
    section("0. Data and feature-table sanity")
    for name, df in (("train", train), ("val", val), ("test", test)):
        print(f"{name}: {len(df):,} rows, ts {df['timestamp'].min():.0f}..{df['timestamp'].max():.0f}, "
              f"dup (user,item) pairs {df.duplicated([Cols.USER_ID, Cols.ITEM_ID]).sum():,}")
    key = lambda df: set(zip(df[Cols.USER_ID].to_numpy(), df[Cols.ITEM_ID].to_numpy()))
    k_tr, k_va, k_te = key(train), key(val), key(test)
    print(f"pair overlap train&test {len(k_tr & k_te):,}, val&test {len(k_va & k_te):,}, train&val {len(k_tr & k_va):,}")
    itf = pd.read_parquet(P / "item_features.parquet")
    uf = pd.read_parquet(P / "user_features.parquet")
    print(f"item_features rows {len(itf):,}, unique items {itf[Cols.ITEM_ID].nunique():,}; "
          f"user_features rows {len(uf):,}, unique users {uf[Cols.USER_ID].nunique():,}")
    print(f"ranker input: 2x{U.shape[1]} emb + {feats.user_dense_dim} user dense + {feats.item_dense_dim} item dense")
    OUT["pair_overlap"] = {"train_test": len(k_tr & k_te), "val_test": len(k_va & k_te)}

    # ------------------------------------------------------------------
    truth_df = test.loc[test[Cols.WATCH_RATIO] >= pos_t]
    gt = truth_df.groupby(Cols.USER_ID)[Cols.ITEM_ID].apply(set).to_dict()
    users = np.array(sorted(gt))
    users = np.sort(RNG.choice(users, min(args.users, len(users)), replace=False))
    seen_tr = train.groupby(Cols.USER_ID)[Cols.ITEM_ID].apply(set).to_dict()
    seen_va = val.groupby(Cols.USER_ID)[Cols.ITEM_ID].apply(set).to_dict()
    pop_count = np.bincount(train.loc[train[Cols.WATCH_RATIO] >= pos_t, Cols.ITEM_ID], minlength=n_items)
    dur = train.groupby(Cols.ITEM_ID)["video_duration"].median().reindex(range(n_items)).to_numpy()
    test_wr = test.groupby([Cols.USER_ID, Cols.ITEM_ID])[Cols.WATCH_RATIO].max().to_dict()  # repeats: keep max

    cands, ret_s, rnk_s, truth = [], [], [], []
    for u in users:
        s = V @ U[u]
        order = np.argsort(-s)
        ex = seen_tr.get(u, set())
        c = np.array([i for i in order if i not in ex][:top_k_recall])
        cands.append(c)
        ret_s.append(s[c])
        rnk_s.append(rank_score(np.broadcast_to(U[u], (len(c), U.shape[1])), V[c], np.full(len(c), u), c))
        truth.append(gt[u])

    def lists_by(scores):
        return [list(c[np.argsort(-sc)]) for c, sc in zip(cands, scores)]

    section(f"1. Headline reproduced on {len(users):,} sampled test users")
    r_ret, pu_ret = recall_at(lists_by(ret_s), truth)
    r_rnk, pu_rnk = recall_at(lists_by(rnk_s), truth)
    print(f"recall@10 retrieval {r_ret:.4f} | full pipeline {r_rnk:.4f} | ratio {r_rnk / r_ret - 1:+.1%}")
    hits200 = np.array([len(set(c) & t) for c, t in zip(cands, truth)])
    print(f"candidate ceiling: mean relevant items inside top-200 = {hits200.mean():.2f}; "
          f"users with >=1 = {np.mean(hits200 > 0):.1%}; oracle recall@10 = "
          f"{np.mean([min(10, h) / min(10, len(t)) for h, t in zip(hits200, truth)]):.4f}")
    OUT["headline"] = {"retrieval": r_ret, "full": r_rnk}

    section("2. Evaluation protocol: slots spent on items the user already watched in the val period")
    for name, sc in (("retrieval", ret_s), ("full pipeline", rnk_s)):
        top = lists_by(sc)
        frac = np.mean([len(set(l[:10]) & seen_va.get(u, set())) / 10 for l, u in zip(top, users)])
        # re-rank excluding val-seen too
        top_ex = [[i for i in l if i not in seen_va.get(u, set())] for l, u in zip(top, users)]
        r_ex, _ = recall_at(top_ex, truth)
        print(f"{name:14s} top-10 share already watched in val: {frac:.1%} | recall@10 excluding val-seen: {r_ex:.4f}")
        OUT[f"valseen_{name}"] = {"share": float(frac), "recall_excl": r_ex}

    section("3. Does the ranker agree with retrieval, and is it personalised?")
    print(f"per-user Spearman(retrieval score, ranker score) over candidates: {per_user_spearman(ret_s, rnk_s):+.3f}")
    # Item-only ranker: average score of each candidate item across 200 reference users.
    ref = RNG.choice(n_users, 200, replace=False)
    all_c = np.unique(np.concatenate(cands))
    item_mean = np.zeros(n_items)
    acc = np.zeros(len(all_c))
    for r in ref:
        acc += rank_score(np.broadcast_to(U[r], (len(all_c), U.shape[1])), V[all_c], np.full(len(all_c), r), all_c)
    item_mean[all_c] = acc / len(ref)
    item_only = [item_mean[c] for c in cands]
    r_io, _ = recall_at(lists_by(item_only), truth)
    print(f"per-user Spearman(ranker score, item-only mean score): {per_user_spearman(rnk_s, item_only):+.3f}")
    print(f"recall@10 ranking candidates by item-only ranker score: {r_io:.4f}")
    # Permute user embedding (keep user dense) and permute user dense (keep embedding).
    perm = RNG.permutation(len(users))
    r_perm_emb, r_perm_dense = [], []
    for j, (u, c) in enumerate(zip(users, cands)):
        v = users[perm[j]]
        r_perm_emb.append(rank_score(np.broadcast_to(U[v], (len(c), U.shape[1])), V[c], np.full(len(c), u), c))
        r_perm_dense.append(rank_score(np.broadcast_to(U[u], (len(c), U.shape[1])), V[c], np.full(len(c), v), c))
    print(f"recall@10 ranker with user EMBEDDING from another user: {recall_at(lists_by(r_perm_emb), truth)[0]:.4f}")
    print(f"recall@10 ranker with user DENSE features from another user: {recall_at(lists_by(r_perm_dense), truth)[0]:.4f}")
    # Retrieval itself with a random other user's embedding (candidate set fixed).
    r_ret_perm = [V[c] @ U[users[perm[j]]] for j, c in enumerate(cands)]
    print(f"recall@10 retrieval order with another user's embedding (same candidates): {recall_at(lists_by(r_ret_perm), truth)[0]:.4f}")
    OUT["personalisation"] = {"item_only": r_io}

    section("4. What does each scorer prefer? (per-user Spearman over candidates)")
    for name, sc in (("retrieval", ret_s), ("ranker", rnk_s)):
        sp_pop = per_user_spearman(sc, [np.log1p(pop_count[c]) for c in cands])
        sp_dur = per_user_spearman(sc, [dur[c] for c in cands])
        print(f"{name:10s} vs item popularity {sp_pop:+.3f} | vs duration {sp_dur:+.3f}")
    rel_in = [np.array([i in t for i in c]) for c, t in zip(cands, truth)]
    rel_pop = np.mean([np.log1p(pop_count[c][m]).mean() for c, m in zip(cands, rel_in) if m.any()])
    irr_pop = np.mean([np.log1p(pop_count[c][~m]).mean() for c, m in zip(cands, rel_in)])
    rel_dur = np.nanmean([np.nanmedian(dur[c][m]) for c, m in zip(cands, rel_in) if m.any()])
    irr_dur = np.nanmean([np.nanmedian(dur[c][~m]) for c, m in zip(cands, rel_in)])
    print(f"within candidates: relevant items log-pop {rel_pop:.2f} vs non-relevant {irr_pop:.2f}; "
          f"median duration {rel_dur / 1000:.1f}s vs {irr_dur / 1000:.1f}s")
    for name, sc in (("retrieval", ret_s), ("full pipeline", rnk_s)):
        top = [c[np.argsort(-s)[:10]] for c, s in zip(cands, sc)]
        print(f"{name:14s} top-10 mean log-pop {np.mean([np.log1p(pop_count[t]).mean() for t in top]):.2f}, "
              f"median duration {np.nanmedian(np.concatenate([dur[t] for t in top])) / 1000:.1f}s")
    pop_rank_lists = [list(c[np.argsort(-pop_count[c])]) for c in cands]
    print(f"recall@10 re-ranking candidates by popularity: {recall_at(pop_rank_lists, truth)[0]:.4f}")
    short_lists = [list(c[np.argsort(dur[c])]) for c in cands]
    print(f"recall@10 re-ranking candidates shortest-first: {recall_at(short_lists, truth)[0]:.4f}")

    section("5. Watch-time AUC: pooled (as reported) vs per user")
    for name, sc in (("retrieval", ret_s), ("ranker", rnk_s)):
        ys, ss, per = [], [], []
        for u, c, s in zip(users, cands, sc):
            wr = np.array([test_wr.get((u, i), np.nan) for i in c])
            m = ~np.isnan(wr)
            if m.sum() == 0:
                continue
            y = (wr[m] >= 0.5).astype(int)
            ys.append(y); ss.append(s[m])
            if 0 < y.sum() < len(y):
                per.append(roc_auc_score(y, s[m]))
        y_all, s_all = np.concatenate(ys), np.concatenate(ss)
        print(f"{name:10s} pooled AUC {roc_auc_score(y_all, s_all):.4f} | mean per-user AUC {np.mean(per):.4f} "
              f"({len(per):,} users) | candidate rows with a test label {len(y_all):,}")

    section("6. Generalisation: pairwise accuracy (same user, pos >= 0.7 vs neg <= 0.3)")
    def pairs(df, n=40000):
        p = df.loc[df[Cols.WATCH_RATIO] >= pos_t, [Cols.USER_ID, Cols.ITEM_ID]]
        q = df.loc[df[Cols.WATCH_RATIO] <= neg_t, [Cols.USER_ID, Cols.ITEM_ID]]
        pools = q.groupby(Cols.USER_ID)[Cols.ITEM_ID].apply(np.array).to_dict()
        p = p[p[Cols.USER_ID].isin(pools.keys())].sample(n, random_state=0)
        uu = p[Cols.USER_ID].to_numpy(); ip = p[Cols.ITEM_ID].to_numpy()
        ineg = np.array([RNG.choice(pools[u]) for u in uu])
        return uu, ip, ineg
    for name, df in (("train", train), ("val", val), ("test", test)):
        uu, ip, ineg = pairs(df)
        ret_acc = np.mean((U[uu] * V[ip]).sum(1) > (U[uu] * V[ineg]).sum(1))
        rp = rank_score(U[uu], V[ip], uu, ip); rn = rank_score(U[uu], V[ineg], uu, ineg)
        dur_acc = np.nanmean(dur[ip] < dur[ineg])
        pop_acc = np.mean(pop_count[ip] > pop_count[ineg])
        print(f"{name:5s} retrieval {ret_acc:.3f} | ranker {np.mean(rp > rn):.3f} | "
              f"'shorter wins' {dur_acc:.3f} | 'more popular wins' {pop_acc:.3f}")

    section("7. Complementarity: does the ranker carry information retrieval lacks?")
    def z(a):
        return (a - a.mean()) / (a.std() + 1e-9)
    for w in (0.25, 0.5):
        blend = [z(r) * (1 - w) + z(k) * w for r, k in zip(ret_s, rnk_s)]
        print(f"recall@10 z-blend, ranker weight {w}: {recall_at(lists_by(blend), truth)[0]:.4f}")

    Path("logs").mkdir(exist_ok=True)
    Path("logs/diagnose_ranker.json").write_text(json.dumps(OUT, indent=2, default=float))


if __name__ == "__main__":
    main()
