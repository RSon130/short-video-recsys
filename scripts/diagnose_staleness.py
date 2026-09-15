"""
Verify the item-staleness explanation for the ranker regression.

Hypothesis: the ranker (trained on items users watched) promotes items the
platform had stopped showing, which cannot be relevant in the test window.
Read-only. Train-window information only, except where marked DIAGNOSTIC.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import load_config
from data.schema import Cols
from evaluation.ab_test import paired_bootstrap
from features.dense_features import DenseFeatureStore
from models.ranker import build_ranker

P = Path("datastore/processed")
RNG = np.random.default_rng(1)


def main(n_users=3000):
    cfg = load_config("config/kuairec.yaml")
    c = [Cols.USER_ID, Cols.ITEM_ID, Cols.WATCH_RATIO, "timestamp", "video_duration"]
    train = pd.read_parquet(P / "interactions/train.parquet", columns=c)
    test = pd.read_parquet(P / "interactions/test.parquet", columns=c)
    U, V = np.load(P / "user_embeddings.npy"), np.load(P / "item_embeddings.npy")
    n_items = len(V)
    feats = DenseFeatureStore.load()
    ranker = build_ranker(cfg, feats.user_dense_dim, feats.item_dense_dim)
    ranker.load_state_dict(torch.load(P / "ranker_model.pt", map_location="cpu", weights_only=True))
    ranker.eval()

    last_seen = train.groupby(Cols.ITEM_ID)["timestamp"].max().reindex(range(n_items)).to_numpy()
    t_cut = np.quantile(train["timestamp"], 0.9)
    recent = np.nan_to_num(last_seen, nan=0) >= t_cut           # train-window only
    dur_any = pd.concat([train, test]).groupby(Cols.ITEM_ID)["video_duration"].median() \
        .reindex(range(n_items)).to_numpy()
    in_test = np.zeros(n_items, bool); in_test[test[Cols.ITEM_ID].unique()] = True   # DIAGNOSTIC
    shown = test.groupby(Cols.USER_ID)[Cols.ITEM_ID].apply(set).to_dict()             # DIAGNOSTIC
    gt = test.loc[test[Cols.WATCH_RATIO] >= 0.7].groupby(Cols.USER_ID)[Cols.ITEM_ID].apply(set).to_dict()
    seen = train.groupby(Cols.USER_ID)[Cols.ITEM_ID].apply(set).to_dict()
    users = np.sort(RNG.choice(np.array(sorted(gt)), n_users, replace=False))

    m = ~np.isnan(dur_any) & ~np.isnan(last_seen)
    print(f"items: {recent.mean():.1%} active in last 10% of train; "
          f"Spearman(duration, last-seen in train) {spearmanr(dur_any[m], last_seen[m]).correlation:+.3f}; "
          f"median duration recent {np.nanmedian(dur_any[recent]) / 1000:.1f}s vs stale {np.nanmedian(dur_any[~recent]) / 1000:.1f}s")

    rows = {k: [] for k in ("ret", "rnk", "ret_recent", "rnk_recent")}
    stats = {k: [] for k in ("ret_intest", "rnk_intest", "ret_recent", "rnk_recent", "ret_shown", "rnk_shown",
                             "ret_prec_shown", "rnk_prec_shown", "cand_intest", "cand_recent")}

    def rec(lst, truth):
        return len(set(lst[:10]) & truth) / min(10, len(truth))

    for u in users:
        s = V @ U[u]
        ex = seen.get(u, set())
        cand = np.array([i for i in np.argsort(-s) if i not in ex][:200])
        x = torch.from_numpy(feats.build_batch(U[u], V, u, cand))
        with torch.no_grad():
            r = ranker(x).squeeze(1).numpy()
        top_ret, top_rnk = cand[np.argsort(-s[cand])], cand[np.argsort(-r)]
        truth = gt[u]
        rows["ret"].append(rec(top_ret, truth)); rows["rnk"].append(rec(top_rnk, truth))
        rows["ret_recent"].append(rec(top_ret[recent[top_ret]], truth))
        rows["rnk_recent"].append(rec(top_rnk[recent[top_rnk]], truth))
        sh = shown.get(u, set())
        for name, top in (("ret", top_ret[:10]), ("rnk", top_rnk[:10])):
            stats[f"{name}_intest"].append(in_test[top].mean())
            stats[f"{name}_recent"].append(recent[top].mean())
            n_sh = len(set(top) & sh)
            stats[f"{name}_shown"].append(n_sh / 10)
            if n_sh:
                stats[f"{name}_prec_shown"].append(len(set(top) & truth) / n_sh)
        stats["cand_intest"].append(in_test[cand].mean()); stats["cand_recent"].append(recent[cand].mean())

    print(f"\n{len(users):,} users. candidates: {np.mean(stats['cand_intest']):.1%} appear anywhere in test (DIAGNOSTIC), "
          f"{np.mean(stats['cand_recent']):.1%} active in last 10% of train")
    for name, label in (("ret", "retrieval"), ("rnk", "ranker")):
        print(f"{label:9s} top-10: in test {np.mean(stats[name + '_intest']):.1%} | recent {np.mean(stats[name + '_recent']):.1%} | "
              f"shown to this user in test {np.mean(stats[name + '_shown']):.2%} | hit rate when shown "
              f"{np.mean(stats[name + '_prec_shown']):.3f}")

    def cmp(a, b, label):
        res = paired_bootstrap(np.array(rows[a]), np.array(rows[b]))
        print(f"{label}: {np.mean(rows[a]):.4f} -> {np.mean(rows[b]):.4f} ({res['relative_lift']:+.1%}, "
              f"95% CI [{res['ci_low']:+.5f}, {res['ci_high']:+.5f}], {'significant' if res['significant'] else 'NOT significant'})")

    print()
    cmp("ret", "rnk", "all candidates      retrieval -> ranker")
    cmp("ret_recent", "rnk_recent", "recent-only (train) retrieval -> ranker")
    cmp("ret", "ret_recent", "retrieval: all -> recent-only filter")
    cmp("rnk", "rnk_recent", "ranker:    all -> recent-only filter")


if __name__ == "__main__":
    main()
