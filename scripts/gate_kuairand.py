"""
Phase 0 signal-ceiling gate on KuaiRand-Pure (docs/phase2_kuairand_plan.md,
Phase 0 step 5). Settings below were frozen before this script touched
validation users' random-log rows. v2 applies the fresh-context AI-agent review
of 2026-09-15 (see "Amendments"), also before any validation run.

Question: on the exposure-unbiased random log, does a personalised LightGBM
rank a user's items better than the best non-personal scorer?

Data (raw ids; one row per user, video, time_ms)
  fit rows        standard log, each label day 4/13-4/20, single-column tabs;
                  features for a day's rows use only dates before that day
  diagnostic rows standard log 4/21, single-column tabs, features from <= 4/20
  evaluation      features from the whole standard log 4/09-4/21; rows are the
                  random log 4/22-5/08 of validation users
                  (config/kuairand_user_split.json), all tabs. Sensitivity:
                  tab 1 only.

  Item and author statistics (and the non-personal baselines) use single-column
  history rows only, so an item's rate does not depend on which UI showed it.
  User statistics use all history rows. Features are shares and smoothed rates,
  not raw counts: the log is front-loaded, so count scales differ between fit
  history (4-11 days) and evaluation history (13 days).
  video_features_statistic is not used (its window overlaps evaluation), nor
  are post-impression fields or upload date (only three distinct values).

Scorers per label
  personal      LightGBM: item + author + user + user x tag / author / duration band
  non-personal  LightGBM on item + author features; item rate; item rate
                shrunk to its duration band; shortest-first; longest-first;
                item impressions. The best on validation is the comparator,
                which favours the baseline. Personal vs the non-personal
                LightGBM (matched ablation) is also reported.

Gate (pre-registered)
  family   preference labels with status "primary" in kuairand_audit.json:
           only explicit_positive. It is there because the rare-label rule
           always keeps the composite; its own item-rate half-width is 0.0188.
  test     paired user bootstrap (2000) of GAUC(personal) - GAUC(best
           non-personal); two-sided p; Holm across the family (one label, so
           p <= 0.05).
  GO       Holm-rejected AND point estimate >= 0.01.
  Power    per-user AUC sd ~0.28 over ~875 users gives a paired SE of about
           0.007-0.010. A significant result needs roughly delta >= 0.014-0.019,
           so NO-GO means "no gain above about 0.02 was detected", not "no
           personal signal".

LightGBM (fixed, no tuning, no early stopping): binary, lr 0.05, 300 rounds,
num_leaves 31, min_data_in_leaf 200, feature_fraction 0.8, bagging 0.8 every
iteration, lambda_l2 10, seed 42. GAUC on the 4/21 rows at 50/100/200/300
rounds is logged as a diagnostic only.

Amendments (2026-09-15, before any validation run)
  * v1 early-stopped on pooled AUC, then on GAUC of 4/21 rows with 6-210 users
    per label; best rounds 2-38 were noise. Now fixed at 300 rounds.
  * v1 fit on 94K rows from 4/17-4/20; now expanding daily cutoffs over
    4/13-4/20 (~344K rows).
  * v1 crashed in the duration-band baseline (duplicate merge column).
  * Raw-count features replaced by shares; item statistics on single-column
    rows; newest-first and item age dropped; dedup on (user, video, time_ms).

Bug-fix re-runs (after run 1, disclosed; verdict rule unchanged)
  Run 1 (design commit 7a43943): NO-GO. A fresh-context AI-agent review of the
  result surfaced that video_features_basic has NaN video_duration for 239
  items (3.1% of random rows). NaN scores corrupted the shortest- and
  longest-first per-user AUCs.
  Run 2 filled those durations from the logs, but every log row for those
  items has duration_ms == 0, so it filled 0 and made them "shortest".
  Caught by a second AI-agent review. NO-GO.
  Run 3 (this code): duration is unknown for those items. NaN for LightGBM,
  own duration band (-1), median item duration in the duration baselines,
  play ratio NaN where duration_ms == 0. per_user_auc rejects NaN.
  All runs are kept: kuairand_gate_run1_nan_duration.json,
  kuairand_gate_run2_zero_duration.json, kuairand_gate.json.

Usage
  python scripts/gate_kuairand.py --fit-only   # train + 4/21 diagnostic; no validation rows
  python scripts/gate_kuairand.py              # full gate
"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from data.kuairand import read_log, read_user_features, read_video_basic
from evaluation.gauc import paired_bootstrap, per_user_auc, users_with_both_classes
from evaluation.small_matrix import holm

AUDIT = Path("datastore/processed/kuairand_audit.json")
SPLIT = Path("config/kuairand_user_split.json")
OUT = Path("datastore/processed/kuairand_gate.json")

LABEL_DAYS = list(range(20220413, 20220421))
DIAG_DAY = 20220421
RATE_LABELS = ["is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate",
               "long_view", "explicit_positive"]
MODEL_LABELS = ["explicit_positive", "is_like", "is_hate", "is_follow", "is_comment", "is_forward",
                "is_click", "long_view"]
MIN_GAIN = 0.01
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
              seed=42, verbose=-1, num_threads=0)
ROUNDS, DIAG_AT, N_BOOT = 300, [50, 100, 200, 300], 2000
DROP = ("user_id", "video_id", "author_id", "dur_band20")


def smooth(pos, cnt, prior, m):
    return (pos + m * prior) / (cnt + m)


def item_table(basic: pd.DataFrame) -> pd.DataFrame:
    b = basic.copy()
    b["duration_s"] = b["video_duration"] / 1000
    b["log_dur"] = np.log1p(b["duration_s"])
    tags = b["tag"].fillna("").astype(str)
    b["tag_first"] = pd.to_numeric(tags.str.split(",").str[0], errors="coerce").fillna(-1).astype(int)
    b["n_tags"] = tags.str.count(",") + (tags != "").astype(int)
    b["video_type_c"] = b["video_type"].astype("category").cat.codes
    b["upload_type_c"] = b["upload_type"].astype("category").cat.codes
    # Unknown duration (239 items) stays NaN for LightGBM and gets its own band, -1.
    known = b["log_dur"].notna()
    b["dur_band"], b["dur_band20"] = -1, -1
    b.loc[known, "dur_band"] = pd.qcut(b.loc[known, "log_dur"].rank(method="first"), 5, labels=False)
    b.loc[known, "dur_band20"] = pd.qcut(b.loc[known, "log_dur"].rank(method="first"), 20, labels=False)
    return b[["video_id", "author_id", "duration_s", "log_dur", "tag_first", "n_tags", "video_type_c",
              "upload_type_c", "music_type", "dur_band", "dur_band20"]]


def user_table(uf: pd.DataFrame) -> pd.DataFrame:
    u = uf.copy()
    u["user_active_degree_c"] = u["user_active_degree"].astype("category").cat.codes
    keep = ["user_id", "user_active_degree_c", "is_lowactive_period", "is_live_streamer", "is_video_author",
            "follow_user_num", "fans_user_num", "friend_user_num", "register_days"] + \
           [f"onehot_feat{i}" for i in range(18)]
    return u[keep]


def build_features(hist: pd.DataFrame, rows: pd.DataFrame, items: pd.DataFrame, users: pd.DataFrame,
                   single_col: list) -> tuple[pd.DataFrame, list, list]:
    """Features for `rows` from `hist` only. Returns (X, nonpersonal_cols, personal_cols), row-aligned."""
    h = hist.merge(items[["video_id", "author_id", "tag_first", "dur_band", "log_dur"]], on="video_id", how="left")
    h["play_ratio"] = (h["play_time_ms"] / h["duration_ms"].where(h["duration_ms"] > 0)).clip(0, 5)
    h1 = h[h["tab"].isin(single_col)]
    priors = {c: h1[c].mean() for c in RATE_LABELS}
    upriors = {c: h[c].mean() for c in RATE_LABELS}

    X = rows[["user_id", "video_id"]].merge(items, on="video_id", how="left")

    # item and author statistics, single-column history only
    gi = h1.groupby("video_id")
    it = pd.DataFrame({"item_impr_share": gi.size() / len(h1), "item_play_ratio": gi["play_ratio"].mean()})
    for c in RATE_LABELS:
        it[f"item_{c}"] = smooth(gi[c].sum(), gi.size(), priors[c], 20)
    ga = h1.groupby("author_id")
    au = pd.DataFrame({"author_impr_share": ga.size() / len(h1)})
    for c in ["is_click", "is_like", "is_follow", "long_view", "explicit_positive"]:
        au[f"author_{c}"] = smooth(ga[c].sum(), ga.size(), priors[c], 20)
    X = X.merge(it, left_on="video_id", right_index=True, how="left") \
         .merge(au, left_on="author_id", right_index=True, how="left")
    nonpersonal = [c for c in X.columns if c not in DROP]

    # user statistics, all history rows
    X = X.merge(users, on="user_id", how="left")
    gu = h.groupby("user_id")
    us = pd.DataFrame({"user_play_ratio": gu["play_ratio"].mean(),
                       "user_single_col_share": gu["tab"].apply(lambda t: t.isin(single_col).mean())})
    for c in RATE_LABELS:
        us[f"user_{c}"] = smooth(gu[c].sum(), gu.size(), upriors[c], 10)
    lv = h[h["long_view"] == 1].groupby("user_id")["log_dur"].mean().rename("user_pref_log_dur")
    X = X.merge(us, left_on="user_id", right_index=True, how="left") \
         .merge(lv, left_on="user_id", right_index=True, how="left")
    X["dur_gap"] = X["log_dur"] - X["user_pref_log_dur"]

    # user x attribute crosses, shrunk toward the user's own rate
    user_n = gu.size()
    for key, name in [("tag_first", "utag"), ("author_id", "uauth"), ("dur_band", "uband")]:
        g = h.groupby(["user_id", key])
        n = g.size()
        cr = pd.DataFrame({"_n": n,
                           f"{name}_share": n / user_n.reindex(n.index.get_level_values(0)).to_numpy(),
                           f"{name}_play_ratio": g["play_ratio"].mean(),
                           "_long_view": g["long_view"].sum(),
                           "_explicit_positive": g["explicit_positive"].sum(),
                           "_is_like": g["is_like"].sum()}).reset_index()
        X = X.merge(cr, on=["user_id", key], how="left")
        cnt = X["_n"].fillna(0)
        for lab in ["long_view", "explicit_positive", "is_like"]:
            X[f"{name}_{lab}"] = smooth(X[f"_{lab}"].fillna(0), cnt, X[f"user_{lab}"].fillna(upriors[lab]), 5)
        X = X.drop(columns=["_n", "_long_view", "_explicit_positive", "_is_like"])
        if key == "author_id":
            X["uauth_followed"] = X[["user_id", "author_id"]].merge(
                h[h["is_follow"] == 1][["user_id", "author_id"]].drop_duplicates().assign(f=1),
                on=["user_id", "author_id"], how="left")["f"].fillna(0).to_numpy()
    personal = [c for c in X.columns if c not in DROP]
    assert len(X) == len(rows) and (X["video_id"].to_numpy() == rows["video_id"].to_numpy()).all() \
        and (X["user_id"].to_numpy() == rows["user_id"].to_numpy()).all()
    return X, nonpersonal, personal


def fit(X, y, Xd, yd, cols):
    booster = lgb.train(PARAMS, lgb.Dataset(X[cols], y), ROUNDS)
    du = Xd["user_id"].to_numpy()
    diag = {r: round(float(per_user_auc(du, yd, booster.predict(Xd[cols], num_iteration=r)).mean()), 4)
            for r in DIAG_AT}
    return booster, diag


def baseline_scores(hist1: pd.DataFrame, X: pd.DataFrame, label: str) -> dict:
    """Non-personal scores from single-column history rows that carry dur_band20."""
    prior = hist1[label].mean()
    hi = hist1.groupby("video_id")[label].agg(["sum", "count"])
    items = X["video_id"]
    cnt = items.map(hi["count"]).fillna(0).to_numpy()
    pos = items.map(hi["sum"]).fillna(0).to_numpy()
    band_prior = X["dur_band20"].map(hist1.groupby("dur_band20")[label].mean()).fillna(prior).to_numpy()
    # unknown duration gets a neutral value: the median known item duration
    dur = X["duration_s"].fillna(X["duration_s"].median()).to_numpy()
    return {"item_rate": (pos + 20 * prior) / (cnt + 20),
            "item_rate_duration_band": (pos + 20 * band_prior) / (cnt + 20),
            "shortest_first": -dur,
            "longest_first": dur,
            "item_impressions": cnt}


def evaluate(val, scores, labels, family):
    u = val["user_id"].to_numpy()
    results = {}
    for lab in labels:
        y = val[lab].to_numpy()
        pua = {k: per_user_auc(u, y, s) for k, s in scores[lab].items()}
        gauc = {k: round(float(v.mean()), 4) for k, v in pua.items()}
        best = max((k for k in gauc if k != "lgbm_personal"), key=lambda k: gauc[k])
        p = scores[lab]["lgbm_personal"]
        centred = p - pd.Series(p).groupby(u).transform("mean").to_numpy()
        results[lab] = {"users_both_classes": users_with_both_classes(u, y), "positives": int(y.sum()),
                        "gauc": gauc, "best_nonpersonal": best,
                        "personal_vs_best": paired_bootstrap(pua[best], pua["lgbm_personal"], n_boot=N_BOOT),
                        "personal_vs_lgbm_nonpersonal": paired_bootstrap(pua["lgbm_nonpersonal"], pua["lgbm_personal"],
                                                                         n_boot=N_BOOT),
                        "pooled_user_centred_auc_personal": round(float(roc_auc_score(y, centred)), 4)
                        if 0 < y.sum() < len(y) else None}
    rejected = holm({lab: results[lab]["personal_vs_best"]["p"] for lab in family})
    go = {lab: bool(rejected[lab] and results[lab]["personal_vs_best"]["diff"] >= MIN_GAIN) for lab in family}
    return results, rejected, go


def print_table(title, results, family):
    print(f"\n[{title}]")
    print(f"{'label':18s} {'users':>6s} {'best non-personal':>30s} {'personal':>9s} {'diff':>8s} {'95% CI':>19s} {'p':>7s}"
          f" | {'vs lgbm-np':>9s}")
    for lab, r in results.items():
        c, m = r["personal_vs_best"], r["personal_vs_lgbm_nonpersonal"]
        print(f"{lab:18s} {r['users_both_classes']:6d} {r['best_nonpersonal']:>23s} {r['gauc'][r['best_nonpersonal']]:.4f} "
              f"{r['gauc']['lgbm_personal']:9.4f} {c['diff']:+8.4f} [{c['ci_low']:+.4f},{c['ci_high']:+.4f}] {c['p']:7.4f}"
              f" | {m['diff']:+.4f}{'  <- gate' if lab in family else ''}")
    for lab, r in results.items():
        print(f"  {lab}: " + ", ".join(f"{k} {v:.4f}" for k, v in r["gauc"].items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-only", action="store_true")
    args = ap.parse_args()

    audit = json.loads(AUDIT.read_text())
    single_col = audit["single_column_tabs_inferred"]
    family = [c for c, v in audit["validation_tasks"].items() if v.get("status") == "primary"]
    print(f"gate family (from audit): {family}; single-column tabs {single_col}")

    tr = read_log("standard_early", dedup=True)
    items, users = item_table(read_video_basic()), user_table(read_user_features())

    parts, labs = [], []
    for day in LABEL_DAYS:
        rows = tr[(tr["date"] == day) & tr["tab"].isin(single_col)].reset_index(drop=True)
        X, np_cols, p_cols = build_features(tr[tr["date"] < day], rows, items, users, single_col)
        parts.append(X)
        labs.append(rows[MODEL_LABELS])
    Xfit, yfit = pd.concat(parts, ignore_index=True), pd.concat(labs, ignore_index=True)
    drows = tr[(tr["date"] == DIAG_DAY) & tr["tab"].isin(single_col)].reset_index(drop=True)
    Xd, _, _ = build_features(tr[tr["date"] < DIAG_DAY], drows, items, users, single_col)
    print(f"fit rows {len(Xfit):,} (explicit positives {int(yfit['explicit_positive'].sum()):,}), "
          f"diagnostic rows {len(Xd):,}, non-personal features {len(np_cols)}, personal features {len(p_cols)}")

    models, diag = {}, {}
    for lab in MODEL_LABELS:
        for kind, cols in [("personal", p_cols), ("nonpersonal", np_cols)]:
            m, d = fit(Xfit, yfit[lab], Xd, drows[lab].to_numpy(), cols)
            models[(lab, kind)], diag[f"{lab}/{kind}"] = (m, cols), d
            print(f"  {lab:18s} {kind:12s} 4/21 GAUC by rounds {d}  "
                  f"(users both classes {users_with_both_classes(drows['user_id'].to_numpy(), drows[lab].to_numpy())})")
    if args.fit_only:
        return

    split = json.loads(SPLIT.read_text())
    rnd = read_log("random", dedup=True)
    val = rnd[rnd["user_id"].isin(set(split["validation"]))].reset_index(drop=True)
    Xval, _, _ = build_features(tr, val, items, users, single_col)
    hist1 = tr[tr["tab"].isin(single_col)].merge(items[["video_id", "dur_band20"]], on="video_id", how="left")

    scores = {}
    for lab in MODEL_LABELS:
        scores[lab] = baseline_scores(hist1, Xval, lab)
        for kind in ["nonpersonal", "personal"]:
            m, cols = models[(lab, kind)]
            scores[lab][f"lgbm_{kind}"] = m.predict(Xval[cols])
    results, rejected, go = evaluate(val, scores, MODEL_LABELS, family)
    verdict = "GO" if any(go.values()) else "NO-GO"

    tab1 = (val["tab"] == 1).to_numpy()
    sens, _, sens_go = evaluate(val[tab1].reset_index(drop=True),
                               {lab: {k: s[tab1] for k, s in d.items()} for lab, d in scores.items()},
                               MODEL_LABELS, family)

    OUT.write_text(json.dumps({"family": family, "holm_rejected": rejected, "go_by_label": go, "verdict": verdict,
                               "diagnostic_4_21_gauc": diag, "results": results,
                               "sensitivity_tab1_only": {"results": sens, "go_by_label": sens_go},
                               "fit_rows": len(Xfit),
                               "features": {"nonpersonal": np_cols, "personal": p_cols}}, indent=1))
    print_table("validation users, random log, all tabs (PRIMARY)", results, family)
    print_table("sensitivity: tab 1 only", sens, family)
    print(f"\nHolm rejected: {rejected}\nGO by label: {go}\nVERDICT: {verdict}   (tab-1 sensitivity: {sens_go})")


if __name__ == "__main__":
    main()
