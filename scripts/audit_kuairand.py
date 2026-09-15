"""
Phase 0 audit of KuaiRand-Pure (docs/phase2_kuairand_plan.md, Phase 0 steps 1-4).

Descriptive only; no model is trained here. Writes
datastore/processed/kuairand_audit.json and freezes the validation/test user
split in config/kuairand_user_split.json (raw user ids, 30/70 hash split of
random-log users).

Pre-registered rules applied here (fixed before the audit was first run):
  * Two-column tab (ORIGINAL rule, failed): a tab whose is_click agrees with
    the valid_play rule on fewer than 99% of its rows, on the premise that
    single-column tabs agree ~100%. The premise is false: no tab reaches 99%
    (the main feed tab 1 agrees 96.7%), so the rule labelled every tab
    two-column. It is still computed and reported.
  * Two-column tab (AMENDMENT 2026-09-15, after seeing per-tab agreement and
    zero-play rates, before any model or gate run): a tab where at least 90%
    of unclicked impressions have play_time_ms == 0 in the train window. In a
    two-column grid nothing plays until tapped; in an autoplay feed almost
    every impression plays. Tabs with < 1000 train rows are left unclassified.
  * Rare-label rule: on validation users' random rows, the user-bootstrap 95%
    CI half-width of the item-rate baseline's GAUC. If > 0.01 the label is
    demoted to secondary; the composite explicit_positive is always kept.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from data.kuairand import LABELS, long_view_rule, read_log, read_video_basic, valid_play
from evaluation.gauc import bootstrap_mean, per_user_auc, users_with_both_classes
from evaluation.small_matrix import split_users

SPLIT_FILE = Path("config/kuairand_user_split.json")
OUT = Path("datastore/processed/kuairand_audit.json")
SEED = 42
TWO_COL_AGREEMENT = 0.99
TWO_COL_ZERO_PLAY = 0.90
MIN_TAB_ROWS = 1000
HALF_WIDTH_MAX = 0.01
PREFERENCE = ["is_like", "is_follow", "is_comment", "is_forward", "is_hate", "explicit_positive"]
ALL_LABELS = LABELS + ["explicit_positive"]


def frozen_user_split(random_users) -> tuple[set, set]:
    if SPLIT_FILE.exists():
        s = json.loads(SPLIT_FILE.read_text())
        return set(s["validation"]), set(s["test"])
    val, test = split_users(random_users, SEED)
    SPLIT_FILE.write_text(json.dumps({
        "seed": SEED, "note": "raw KuaiRand user_id of random-log users; see docs/phase2_kuairand_plan.md",
        "validation": [int(u) for u in val], "test": [int(u) for u in test]}, indent=1))
    return set(int(u) for u in val), set(int(u) for u in test)


def file_audit(name, df):
    dup_all = int(df.duplicated().sum())
    dup_key = int(df.duplicated(["user_id", "video_id", "time_ms"]).sum())
    ts_date = pd.to_datetime(df["time_ms"], unit="ms", utc=True).dt.tz_convert("Asia/Shanghai")
    date_mismatch = float((ts_date.dt.strftime("%Y%m%d").astype(int) != df["date"]).mean())
    return {
        "rows": len(df), "users": int(df["user_id"].nunique()), "items": int(df["video_id"].nunique()),
        "date_min": int(df["date"].min()), "date_max": int(df["date"].max()),
        "date_vs_time_ms_mismatch_frac_utc8": round(date_mismatch, 5),
        "exact_duplicate_rows": dup_all, "duplicate_user_item_time": dup_key,
        "repeat_user_item_pairs": int(df.duplicated(["user_id", "video_id"]).sum()),
        "is_rand_values": {int(k): int(v) for k, v in df["is_rand"].value_counts().items()},
        "play_time_zero_frac": round(float((df["play_time_ms"] == 0).mean()), 4),
    }


def tab_audit(df):
    vp = valid_play(df["play_time_ms"], df["duration_ms"])
    lv = long_view_rule(df["play_time_ms"], df["duration_ms"])
    t = df.assign(vp_agree=(df["is_click"] == vp), lv_agree=(df["long_view"] == lv),
                  zero_play_unclicked=((df["play_time_ms"] == 0) & (df["is_click"] == 0)),
                  unclicked=(df["is_click"] == 0))
    g = t.groupby("tab")
    out = pd.DataFrame({
        "rows": g.size(),
        "click_agrees_valid_play": g["vp_agree"].mean(),
        "long_view_agrees_rule": g["lv_agree"].mean(),
        "zero_play_given_unclicked": g["zero_play_unclicked"].sum() / g["unclicked"].sum().clip(lower=1),
        **{f"rate_{c}": g[c].mean() for c in ALL_LABELS},
    })
    return out


def duration_audit(df, n_buckets=20):
    """Positive rate by duration bucket, and the gradient left *within* buckets."""
    d = df.assign(bucket=pd.qcut(df["duration_ms"].rank(method="first"), n_buckets, labels=False))
    res = {}
    for c in ALL_LABELS:
        by_bucket = d.groupby("bucket").agg(dur_s_mid=("duration_ms", "median"), rate=(c, "mean"))
        # within-bucket: pooled AUC of shortest-first inside each bucket, weighted
        # by positives. Scale-free, unlike a correlation, which is ~0 for rare labels.
        within = []
        for _, b in d.groupby("bucket"):
            if b[c].nunique() > 1:
                within.append((b[c].sum(), roc_auc_score(b[c], -b["duration_ms"])))
        w = np.array(within)
        # per-user AUC of shortest-first: how much of the label duration alone ranks
        pua = per_user_auc(df["user_id"].to_numpy(), df[c].to_numpy(), -df["duration_ms"].to_numpy())
        res[c] = {
            "rate_by_bucket": [[round(r.dur_s_mid / 1000, 1), round(r.rate, 5)] for r in by_bucket.itertuples()],
            "within_bucket_shortest_first_auc": round(float((w[:, 0] * w[:, 1]).sum() / w[:, 0].sum()), 4) if len(w) else None,
            "shortest_first_gauc": round(float(pua.mean()), 4), "shortest_first_gauc_users": int(len(pua)),
        }
    return res


def item_rate_scores(train, target, label, m=20):
    prior = train[label].mean()
    g = train.groupby("video_id")[label].agg(["sum", "count"])
    rate = (g["sum"] + m * prior) / (g["count"] + m)
    return target["video_id"].map(rate).fillna(prior).to_numpy()


def main():
    logs = {k: read_log(k) for k in ["standard_early", "standard_late", "random"]}
    basic = read_video_basic()
    report = {"files": {k: file_audit(k, v) for k, v in logs.items()}}

    tr, rnd, late = logs["standard_early"], logs["random"], logs["standard_late"]
    items = {k: set(v["video_id"]) for k, v in logs.items()}
    users = {k: set(v["user_id"]) for k, v in logs.items()}
    report["overlap"] = {
        "random_items_not_in_train_window": len(items["random"] - items["standard_early"]),
        "random_items_not_in_any_standard": len(items["random"] - items["standard_early"] - items["standard_late"]),
        "random_rows_on_items_not_in_train_window": int((~rnd["video_id"].isin(items["standard_early"])).sum()),
        "random_users_not_in_train_window": len(users["random"] - users["standard_early"]),
        "items_in_basic_features": len(basic),
        "random_items_missing_basic_features": len(items["random"] - set(basic["video_id"])),
        "random_rows_also_in_standard_late_same_time": int(rnd.merge(late[["user_id", "video_id", "time_ms"]],
                                                                   on=["user_id", "video_id", "time_ms"]).shape[0]),
        "duration_ms_vs_basic_mismatch_frac": round(float(
            (rnd.merge(basic[["video_id", "video_duration"]], on="video_id")
             .pipe(lambda x: (x["duration_ms"] - x["video_duration"]).abs() > 1000)).mean()), 4),
    }
    per_user = rnd.groupby("user_id").size()
    report["random_items_per_user"] = {q: float(per_user.quantile(q)) for q in [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]}
    report["random_items_per_user"]["mean"] = float(per_user.mean())

    tabs = {k: tab_audit(v) for k, v in logs.items()}
    report["tabs"] = {k: v.round(5).reset_index().to_dict(orient="records") for k, v in tabs.items()}
    agree = pd.concat([v[["rows", "click_agrees_valid_play"]] for v in tabs.values()]).groupby(level=0) \
        .apply(lambda x: (x["rows"] * x["click_agrees_valid_play"]).sum() / x["rows"].sum())
    report["two_column_tabs_original_rule"] = sorted(int(t) for t, a in agree.items() if a < TWO_COL_AGREEMENT)
    te = tabs["standard_early"]
    two_col = sorted(int(t) for t, r in te.iterrows()
                     if r["rows"] >= MIN_TAB_ROWS and r["zero_play_given_unclicked"] >= TWO_COL_ZERO_PLAY)
    single_col = sorted(int(t) for t, r in te.iterrows()
                        if r["rows"] >= MIN_TAB_ROWS and r["zero_play_given_unclicked"] < TWO_COL_ZERO_PLAY)
    report["two_column_tabs_inferred"] = two_col
    report["single_column_tabs_inferred"] = single_col
    report["two_column_rule"] = (f"amended: train-window zero-play share of unclicked rows >= {TWO_COL_ZERO_PLAY}, "
                                 f"tabs with >= {MIN_TAB_ROWS} rows; others unclassified")

    rates = {}
    for k, v in logs.items():
        two = v["tab"].isin(two_col)
        rates[k] = {c: round(float(v[c].mean()), 5) for c in ALL_LABELS}
        rates[k]["click_two_column"] = round(float(v.loc[two, "is_click"].mean()), 5) if two.any() else None
        one = v["tab"].isin(single_col)
        rates[k]["click_single_column"] = round(float(v.loc[one, "is_click"].mean()), 5) if one.any() else None
        rates[k]["two_column_row_frac"] = round(float(two.mean()), 5)
    report["label_rates"] = rates

    report["duration_random_log"] = duration_audit(rnd)
    report["duration_random_log_single_column"] = {
        c: v for c, v in duration_audit(rnd[rnd["tab"].isin(single_col)]).items() if c in ["is_click", "long_view"]}

    val_users, test_users = frozen_user_split(rnd["user_id"].unique())
    report["split"] = {"validation_users": len(val_users), "test_users": len(test_users)}
    val = rnd[rnd["user_id"].isin(val_users)].copy()
    val["click_two_column"] = np.where(val["tab"].isin(two_col), val["is_click"], np.nan)

    tasks = {}
    for c in PREFERENCE + ["click_two_column", "is_click", "long_view"]:
        v = val.dropna(subset=[c]) if c == "click_two_column" else val
        train_col = "is_click" if c == "click_two_column" else c
        t = tr[tr["tab"].isin(two_col)] if c == "click_two_column" else tr
        if v["user_id"].nunique() < 2:
            tasks[c] = {"rows": 0}
            continue
        t = t.drop_duplicates()
        y = v[c].to_numpy().astype(int)
        pua = per_user_auc(v["user_id"].to_numpy(), y, item_rate_scores(t, v, train_col))
        hw = bootstrap_mean(pua.to_numpy()) if len(pua) > 1 else {"half_width": float("nan"), "mean": float("nan"), "n": len(pua)}
        tasks[c] = {
            "val_rows": int(len(v)), "val_positives": int(y.sum()),
            "val_users_both_classes": users_with_both_classes(v["user_id"].to_numpy(), y),
            "item_rate_gauc": round(hw["mean"], 4), "item_rate_gauc_half_width": round(hw["half_width"], 4),
            "status": ("primary" if (c == "explicit_positive" or hw["half_width"] <= HALF_WIDTH_MAX)
                       else "secondary (half-width > 0.01)") if c not in ["is_click", "long_view"]
            else "watch-time label (not in gate)",
        }
    report["validation_tasks"] = tasks

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: report[k] for k in ["files", "overlap", "random_items_per_user", "two_column_tabs_original_rule", "two_column_tabs_inferred", "single_column_tabs_inferred",
                                             "label_rates", "split", "validation_tasks"]}, indent=1, default=float))
    for k, v in tabs.items():
        print(f"\n[tabs: {k}]")
        print(v[["rows", "click_agrees_valid_play", "long_view_agrees_rule", "zero_play_given_unclicked",
                 "rate_is_click", "rate_is_like", "rate_long_view"]].round(4).to_string())
    print("\n[duration, random log]")
    for c, v in report["duration_random_log"].items():
        rb = v["rate_by_bucket"]
        print(f"  {c:18s} shortest-first GAUC {v['shortest_first_gauc']:.4f} (users {v['shortest_first_gauc_users']:>6}) "
              f"within-bucket AUC {v['within_bucket_shortest_first_auc']}  rate shortest {rb[0][1]:.4f} longest {rb[-1][1]:.4f}")
    for c, v in report["duration_random_log_single_column"].items():
        print(f"  single-col {c:10s} shortest-first GAUC {v['shortest_first_gauc']:.4f} within-bucket AUC {v['within_bucket_shortest_first_auc']}")


if __name__ == "__main__":
    main()
