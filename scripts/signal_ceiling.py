"""
Is there learnable *personal* preference in KuaiRec's watch_ratio at all?

Two measurements, independent of this project's models:

1. Test-retest: (user, video) pairs watched more than once in big_matrix. If
   the user-specific part of watch_ratio (after removing user and item means)
   does not repeat, it is noise, and no model can learn it.
2. Low-rank ceiling on small_matrix (fully observed): hide a random 50% of each
   user's pairs, fit user/item means and a truncated SVD of the residual on the
   rest, and score the hidden half with per-user AUC. Item means capture
   duration and item quality; the SVD residual is the personal part.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RNG = np.random.default_rng(0)
RAW = "datastore/raw/kuairec"


def test_retest():
    b = pd.read_csv(f"{RAW}/big_matrix.csv", usecols=["user_id", "video_id", "timestamp", "watch_ratio"])
    b["wr"] = b["watch_ratio"].clip(0, 5)
    gm = b["wr"].mean()
    b["resid"] = b["wr"] - b.groupby("video_id")["wr"].transform("mean") \
        - b.groupby("user_id")["wr"].transform("mean") + gm
    b = b.sort_values("timestamp")
    b["n"] = b.groupby(["user_id", "video_id"]).cumcount()
    first = b[b["n"] == 0].set_index(["user_id", "video_id"])
    second = b[b["n"] == 1].set_index(["user_id", "video_id"])
    j = first.join(second, lsuffix="_1", rsuffix="_2", how="inner")
    print(f"[test-retest] pairs watched twice: {len(j):,} "
          f"({j.index.get_level_values(0).nunique():,} users, {j.index.get_level_values(1).nunique():,} videos)")
    print(f"  corr(watch_ratio view1, view2)                 {np.corrcoef(j['wr_1'], j['wr_2'])[0, 1]:+.3f}")
    print(f"  corr(user-item residual view1, view2)          {np.corrcoef(j['resid_1'], j['resid_2'])[0, 1]:+.3f}")
    pos1, pos2 = j["wr_1"] >= 0.7, j["wr_2"] >= 0.7
    print(f"  label agreement (>=0.7) view1 vs view2         {np.mean(pos1 == pos2):.3f} "
          f"(chance at these rates {np.mean(pos1) * np.mean(pos2) + (1 - np.mean(pos1)) * (1 - np.mean(pos2)):.3f})")


def low_rank_ceiling():
    s = pd.read_csv(f"{RAW}/small_matrix.csv", usecols=["user_id", "video_id", "watch_ratio", "video_duration"])
    s = s.groupby(["user_id", "video_id"], as_index=False).agg(wr=("watch_ratio", "max"), dur=("video_duration", "first"))
    s["wr"] = s["wr"].clip(0, 5)
    uid = {u: k for k, u in enumerate(s["user_id"].unique())}
    iid = {v: k for k, v in enumerate(s["video_id"].unique())}
    s["u"], s["i"] = s["user_id"].map(uid), s["video_id"].map(iid)
    dur = s.groupby("i")["dur"].first()
    s["dec"] = s["i"].map(pd.qcut(dur.rank(method="first"), 50, labels=False))
    s["rel_raw"] = (s["wr"] >= 0.7).astype(int)
    s["rel_deb"] = (s.groupby("dec")["wr"].rank(pct=True) >= 0.7).astype(int)
    s["hold"] = RNG.random(len(s)) < 0.5

    tr = s[~s["hold"]]
    gm = tr["wr"].mean()
    ib = tr.groupby("i")["wr"].mean() - gm
    ub = tr.groupby("u")["wr"].mean() - gm
    M = np.zeros((len(uid), len(iid)), dtype=np.float32)
    W = np.zeros_like(M)
    r = tr["wr"] - gm - tr["i"].map(ib) - tr["u"].map(ub)
    M[tr["u"], tr["i"]] = r
    W[tr["u"], tr["i"]] = 1
    print(f"[low-rank] small_matrix {len(uid):,} x {len(iid):,}, train cells {W.mean():.1%}; "
          f"duration deciles 50 for debiased label")
    U_, S_, Vt = np.linalg.svd(M / 0.5, full_matrices=False)   # rescale for 50% missing

    te = s[s["hold"]].copy()
    te["item"] = te["i"].map(ib).fillna(0)
    rng_scores = RNG.random(len(te))
    models = {"random": rng_scores, "item mean (duration + quality)": te["item"].to_numpy()}
    for k in (4, 16, 64):
        P = (U_[:, :k] * S_[:k]) @ Vt[:k]
        models[f"SVD-{k} residual only (personal)"] = P[te["u"], te["i"]]
        models[f"item mean + SVD-{k}"] = te["item"].to_numpy() + P[te["u"], te["i"]]
    te["dur"] = te["dur"].astype(float)
    models["shortest-first"] = -te["dur"].to_numpy()

    for lab in ("rel_raw", "rel_deb"):
        print(f"  per-user AUC, {'RAW' if lab == 'rel_raw' else 'DURATION-DEBIASED'} relevance on hidden half:")
        for name, sc in models.items():
            te["_s"] = sc
            aucs = [roc_auc_score(g[lab], g["_s"]) for _, g in te.groupby("u") if 0 < g[lab].sum() < len(g)]
            print(f"    {name:36s} {np.mean(aucs):.4f}")


if __name__ == "__main__":
    test_retest()
    low_rank_ceiling()
