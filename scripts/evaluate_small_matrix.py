"""
Evaluate the pipeline under docs/evaluation_protocol.md (v2).

Usage:
    python scripts/evaluate_small_matrix.py [--skip-item-bootstrap]

Order of operations follows the protocol: label check first, then systems and
baselines on validation and test users, then the pre-registered comparisons on
test users with Holm correction, the 20/30/50% direction check, and an item
bootstrap for comparisons that pass.
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
from evaluation.small_matrix import (FRACS, PRIMARY_FRAC, Baselines, DurationModel, auc,
                                     bootstrap_diff, build_labels, holm, label_check,
                                     load_small_matrix, ndcg_at, frozen_split, tiebreak_order,
                                     top_candidates, two_stage_scores, user_metrics)
from features.dense_features import DenseFeatureStore
from models.ranker import build_ranker

P = Path("datastore/processed")
LABELS = [f"rel_{int(f * 100)}" for f in FRACS] + ["rel_raw"]
PRIMARY = f"rel_{int(PRIMARY_FRAC * 100)}"
SYSTEMS = ["retrieval", "ranker alone", "two-stage"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/kuairec.yaml")
    ap.add_argument("--skip-item-bootstrap", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = cfg["project"]["seed"]
    report = {"protocol": "docs/evaluation_protocol.md v2"}

    maps = pickle.load(open(P / "id_maps.pkl", "rb"))
    n_users, n_items = maps["n_users"], maps["n_items"]
    train = pd.read_parquet(P / "interactions/train.parquet",
                            columns=[Cols.USER_ID, Cols.ITEM_ID, Cols.WATCH_RATIO, "video_duration"])
    U, V = np.load(P / "user_embeddings.npy"), np.load(P / "item_embeddings.npy")
    feats = DenseFeatureStore.load()
    ranker = build_ranker(cfg, feats.user_dense_dim, feats.item_dense_dim)
    ranker.load_state_dict(torch.load(P / "ranker_model.pt", map_location="cpu", weights_only=True))
    ranker.eval()
    for name in ("retrieval_meta.json", "ranker_meta.json"):
        if (P / name).exists():
            meta = json.loads((P / name).read_text())
            report[name] = {k: meta.get(k) for k in ("selected_on", "best_epoch", "trained_at")}
            print(f"{name}: {report[name]}")

    raw_dir = Path(cfg["data"]["kuairec"]["raw_dir"])
    pairs, info = load_small_matrix(raw_dir, maps["user_id_map"], maps["item_id_map"])
    model = DurationModel.fit(train, n_items)
    df, drop = build_labels(pairs, model)
    info.update(drop)
    report["data"] = info
    print(f"data: {info}")

    # --- 1. pre-registered label check, before any model is scored ---------
    checks = {f: label_check(df, f) for f in FRACS}
    report["label_check"] = {str(f): c for f, c in checks.items()}
    for f, c in checks.items():
        print(f"label check @{int(f * 100)}%: bucket positive rate {c['min_bucket_rate']:.3f}..{c['max_bucket_rate']:.3f} "
              f"corr(bucket, label) {c['corr_duration_label']:+.3f}"
              + (f" | primary range {c['range']} -> {'PASS' if c['passed'] else 'FAIL'}" if f == PRIMARY_FRAC else ""))

    val_users, test_users = frozen_split(df["u"].unique(), maps["user_id_map"], seed)
    report["users"] = {"validation": len(val_users), "test": len(test_users)}
    for name in ("retrieval_meta.json", "ranker_meta.json"):
        if (P / name).exists():
            hist = json.loads((P / name).read_text()).get("history")
            report.setdefault("validation_scorings", {})[name] = len(hist) if hist else "not recorded"
    print(f"users: validation {len(val_users)}, test {len(test_users)}")

    # Weighted by play_cnt: days with no plays report play_progress 0, and an
    # unweighted mean would score items by how many empty days they had.
    daily = pd.read_csv(raw_dir / "item_daily_features.csv", usecols=["video_id", "play_progress", "play_cnt"])
    daily = daily[daily["play_cnt"] > 0]
    daily["w"] = daily["play_progress"] * daily["play_cnt"]
    agg = daily.groupby("video_id")[["w", "play_cnt"]].sum()
    pp = agg["w"] / agg["play_cnt"]
    play_progress = np.full(n_items, np.nan)
    known = pp.index.map(maps["item_id_map"])
    keep = ~pd.isna(known)
    play_progress[known[keep].astype(int)] = pp.to_numpy()[keep]
    all_users = np.sort(df["u"].unique())
    target_items = np.sort(df["i"].unique())
    print("fitting baselines (item-kNN takes a minute) ...", flush=True)
    base = Baselines(train, model, n_users, n_items, play_progress, all_users, target_items,
                     pos_threshold=cfg["features"]["positive_watch_ratio"], seed=seed)
    names = SYSTEMS + list(Baselines.NAMES)

    # --- 2. score every system for every evaluation user ------------------
    rng = np.random.default_rng(seed)
    boot_users = set(rng.choice(test_users, min(300, len(test_users)), replace=False).tolist())
    per_user = {lab: {s: {"auc": [], "ndcg10": [], "p10": []} for s in names} for lab in LABELS}
    extra = {lab: {"within_candidate_auc_ranker": [], "within_candidate_auc_retrieval": [],
                   "candidate_recall": []} for lab in LABELS}
    order_users, stored = [], {}
    groups = df.groupby("u")
    print(f"scoring {len(all_users):,} users ...", flush=True)
    for n, (u, g) in enumerate(groups):
        u = int(u)
        items = g["i"].to_numpy()
        tb = np.random.default_rng(seed + 7 * u).random(len(items))
        ret = V[items] @ U[u]
        with torch.no_grad():
            rk = ranker(torch.from_numpy(feats.build_batch(U[u], V, u, items))).squeeze(1).numpy()
        cand = top_candidates(ret)
        scores = {"retrieval": ret, "ranker alone": rk, "two-stage": two_stage_scores(ret, rk[cand], cand)}
        for b in Baselines.NAMES:
            scores[b] = base.scores(b, u, items)
        order_users.append(u)
        for lab in LABELS:
            y = g[lab].to_numpy()
            for s, sc in scores.items():
                m = user_metrics(y, sc, tb)
                for k in m:
                    per_user[lab][s][k].append(m[k])
            extra[lab]["within_candidate_auc_ranker"].append(auc(y[cand], rk[cand]))
            extra[lab]["within_candidate_auc_retrieval"].append(auc(y[cand], ret[cand]))
            extra[lab]["candidate_recall"].append(y[cand].sum() / max(y.sum(), 1))
        if u in boot_users:
            stored[u] = {"items": items, "tb": tb, "cand": cand, "rk": rk, "ret": ret,
                         "labels": {lab: g[lab].to_numpy() for lab in LABELS},
                         "scores": {s: scores[s] for s in names}}
        if (n + 1) % 200 == 0:
            print(f"  {n + 1:,} users", flush=True)

    order_users = np.array(order_users)
    is_test = np.isin(order_users, test_users)
    is_val = np.isin(order_users, val_users)

    def arr(lab, s, metric, mask):
        return np.array(per_user[lab][s][metric])[mask]

    def xarr(lab, key, mask):
        return np.array(extra[lab][key])[mask]

    # --- 3. tables ---------------------------------------------------------
    report["tables"] = {}
    for split_name, mask in (("validation", is_val), ("test", is_test)):
        for lab in LABELS:
            key = f"{split_name}/{lab}"
            table = {s: {m: float(np.nanmean(arr(lab, s, m, mask))) for m in ("auc", "ndcg10", "p10")} for s in names}
            table["_two_stage_diagnostics"] = {k: float(np.nanmean(xarr(lab, k, mask))) for k in extra[lab]}
            report["tables"][key] = table
            print(f"\n=== {split_name.upper()} users ({mask.sum()}) — label {lab} ===")
            print("system".ljust(42) + "per-user AUC   NDCG@10     P@10")
            for s in names:
                t = table[s]
                print(s.ljust(42) + f"{t['auc']:12.4f}{t['ndcg10']:10.4f}{t['p10']:9.4f}")
            d = table["_two_stage_diagnostics"]
            print(f"  within-candidate AUC: ranker {d['within_candidate_auc_ranker']:.4f} vs retrieval "
                  f"{d['within_candidate_auc_retrieval']:.4f} | candidate recall (share of relevant in top-200) "
                  f"{d['candidate_recall']:.4f}")

    # --- 4. pre-registered comparisons on test users -----------------------
    comps = {}
    for s in ("retrieval", "ranker alone"):
        for b in Baselines.NAMES[1:]:
            comps[f"{s} vs {b} [auc]"] = ("metric", b, s, "auc")
    comps["two-stage vs retrieval [ndcg10]"] = ("metric", "retrieval", "two-stage", "ndcg10")
    comps["two-stage vs retrieval [within-candidate auc]"] = ("wc", None, None, None)

    def diff_for(lab, spec, mask):
        kind, a, b, metric = spec
        if kind == "wc":
            return xarr(lab, "within_candidate_auc_retrieval", mask), xarr(lab, "within_candidate_auc_ranker", mask)
        return arr(lab, a, metric, mask), arr(lab, b, metric, mask)

    results = {}
    for name, spec in comps.items():
        a, b = diff_for(PRIMARY, spec, is_test)
        r = bootstrap_diff(a, b, seed=seed)
        r["direction_by_frac"] = {}
        for lab in LABELS[:3]:
            a2, b2 = diff_for(lab, spec, is_test)
            r["direction_by_frac"][lab] = float(np.nanmean(b2 - a2))
        results[name] = r
    rejected = holm({k: v["p"] for k, v in results.items()})
    for k in results:
        r = results[k]
        signs = {np.sign(v) for v in r["direction_by_frac"].values()}
        r["holm_significant"] = rejected[k]
        r["direction_consistent"] = len(signs) == 1 and 0 not in signs
        r["passes"] = bool(r["holm_significant"] and r["direction_consistent"] and r["diff"] > 0)

    # --- 5. item bootstrap for passing comparisons -------------------------
    if not args.skip_item_bootstrap:
        brng = np.random.default_rng(seed + 1)
        n_t = len(target_items)
        for name, spec in comps.items():
            if not results[name]["passes"]:
                continue
            means = []
            for _ in range(200):
                w = np.bincount(brng.integers(0, n_t, n_t), minlength=n_t)
                diffs = []
                for u, st in stored.items():
                    reps = w[np.searchsorted(target_items, st["items"])]
                    idx = np.repeat(np.arange(len(st["items"])), reps)
                    y = st["labels"][PRIMARY][idx]
                    if spec[0] == "wc":
                        in_c = np.zeros(len(st["items"]), bool); in_c[st["cand"]] = True
                        ci = idx[in_c[idx]]
                        diffs.append(auc(st["labels"][PRIMARY][ci], st["rk"][ci]) - auc(st["labels"][PRIMARY][ci], st["ret"][ci]))
                    else:
                        _, a, b, metric = spec
                        if metric == "auc":
                            diffs.append(auc(y, st["scores"][b][idx]) - auc(y, st["scores"][a][idx]))
                        else:
                            def nd(sc):
                                o = tiebreak_order(sc[idx], st["tb"][idx])
                                return ndcg_at(y[o], int(y.sum()))
                            diffs.append(nd(st["scores"][b]) - nd(st["scores"][a]))
                means.append(np.nanmean(diffs))
            lo, hi = np.quantile(means, [0.025, 0.975])
            results[name]["item_bootstrap_ci"] = [float(lo), float(hi)]
            results[name]["passes"] = bool(results[name]["passes"] and lo > 0)

    report["comparisons_test"] = results
    print("\n=== Pre-registered comparisons (test users, primary label, Holm over all 14) ===")
    for k, r in results.items():
        ib = r.get("item_bootstrap_ci")
        print(f"{k:62s} diff {r['diff']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] p={r['p']:.4f} "
              f"holm={'Y' if r['holm_significant'] else 'N'} dir20/30/50={'Y' if r['direction_consistent'] else 'N'}"
              + (f" itemCI [{ib[0]:+.4f}, {ib[1]:+.4f}]" if ib else "") + f" -> {'PASS' if r['passes'] else 'no'}")

    verdict = {}
    for s in ("retrieval", "ranker alone"):
        verdict[s] = all(results[f"{s} vs {b} [auc]"]["passes"] for b in Baselines.NAMES[1:])
    verdict["ranker adds value in two-stage"] = (results["two-stage vs retrieval [ndcg10]"]["passes"]
                                                 and results["two-stage vs retrieval [within-candidate auc]"]["passes"])
    report["verdict"] = verdict
    print(f"\nVERDICT (adds value per protocol §7): {verdict}")

    out = P / "small_matrix_eval.json"
    out.write_text(json.dumps(report, indent=2, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
