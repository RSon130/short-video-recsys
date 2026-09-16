"""
The same model, scored two ways: the standard offline evaluation everyone runs,
and the exposure-unbiased one.

Descriptive, not a decision. docs/phase2_exposure_bias_plan.md reserved the
standard log 4/22-5/08 as a contrast set "never used for decisions"; this is
that contrast. No threshold, gate or claim depends on it, and the pre-registered
verdicts in engineering log section 19 stand as they are.

  exposed    standard log 4/22-5/08, test users, tabs 1-2 — what the
             recommender chose to show. Training and evaluation share this
             distribution, which is the setup most offline reports use.
  unbiased   random log 4/22-5/08, test users, tabs 1-2 — uniformly exposed
             videos from the platform's candidate pool.

Same model (A0: personal LightGBM, seeds 42/43/44 averaged), same features
(standard log 4/09-4/21), same comparator (N1: item impression count), same
metric (per-user AUC).

RESULT AND CORRECTION (engineering log section 20). Against N1 the logged gap
looks like +0.08 to +0.25 AUC. A fresh-context AI-agent review showed 78-96% of
that is N1 failing, not the model winning: N1 scores below chance on logged
explicit labels (the section 19 sign flip). Against a fair comparator - the item
positive rate from the training window - the logged gap is +0.017 on watch
labels and ~0 on explicit labels once duration is demeaned. Read this script's
output with that in mind; `logs/verify_fair_comparator.py` computes the fair
comparison.

Caveats: `positive_rate` below is over all rows, not the evaluable users
actually scored; p values of 0.0001 are the 1/n_boot floor and mean "< 1e-4";
N1 gives never-impressed items a tied floor score, so on the unbiased side it
is partly an "ever shown" indicator.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.kuairand import read_log, read_user_features, read_video_basic
from evaluation.gauc import paired_bootstrap, per_user_auc, users_with_both_classes
from scripts.exposure_bias_kuairand import (FIT_DAYS, LABELS, LATE_START, PRIMARY_TABS,
                                            SEEDS, arm_scores, band_demean, train)
from scripts.gate_kuairand import AUDIT, SPLIT, build_features, item_table, smooth, user_table

OUT = Path("datastore/processed/kuairand_exposed_vs_unbiased.json")
N_BOOT = 10_000


def evaluate(name, rows, X, scores, labels):
    u = rows["user_id"].to_numpy()
    bands = X["dur_band20"].to_numpy()
    out = {}
    for lab in labels:
        y = rows[lab].to_numpy()
        pua = {k: per_user_auc(u, y, v) for k, v in scores.items()}
        cmp_ = paired_bootstrap(pua["N1"], pua["A0"], n_boot=N_BOOT)
        # same duration guard as the main experiment: scores demeaned within 20
        # duration bands, so a gap that is really a duration sort shows up
        dm = {k: per_user_auc(u, y, band_demean(scores[k], bands)) for k in ("N1", "A0")}
        j = pd.concat([dm["N1"].rename("a"), dm["A0"].rename("b")], axis=1).dropna()
        cmp_["duration_demeaned_diff"] = round(float((j["b"] - j["a"]).mean()), 4)
        out[lab] = {"users_both_classes": users_with_both_classes(u, y), "positives": int(y.sum()),
                    "rows": int(len(rows)), "positive_rate": round(float(y.mean()), 4),
                    "gauc": {k: round(float(v.mean()), 4) for k, v in pua.items()},
                    "A0_minus_N1": cmp_,
                    "gauc_duration_demeaned": {k: round(float(v.mean()), 4) for k, v in dm.items()}}
    return out


def main():
    single_col = json.loads(AUDIT.read_text())["single_column_tabs_inferred"]
    test_users = set(json.loads(SPLIT.read_text())["test"])
    tr = read_log("standard_early", dedup=True)
    late = read_log("standard_late", dedup=True)
    rnd = read_log("random", dedup=True)
    items, users = item_table(read_video_basic()), user_table(read_user_features())

    parts, labs = [], []
    for day in FIT_DAYS:
        rows = tr[(tr["date"] == day) & tr["tab"].isin(single_col)].reset_index(drop=True)
        X, _, p_cols = build_features(tr[tr["date"] < day], rows, items, users, single_col)
        parts.append(X)
        labs.append(rows[LABELS])
    Xfit, yfit = pd.concat(parts, ignore_index=True), pd.concat(labs, ignore_index=True)
    print(f"A0 fit rows {len(Xfit):,}")

    arms = {lab: {"cols": p_cols, "boosters": [train(Xfit, yfit[lab], p_cols, s) for s in SEEDS]}
            for lab in LABELS}
    print("trained", list(arms))

    hist1 = tr[tr["tab"].isin(single_col)]
    impressions = hist1["video_id"].value_counts()
    report = {"note": __doc__.strip(), "evaluations": {}}
    for name, df in [("exposed_standard_log", late), ("unbiased_random_log", rnd[rnd["date"] >= LATE_START])]:
        rows = df[df["user_id"].isin(test_users) & df["tab"].isin(PRIMARY_TABS)].reset_index(drop=True)
        X, _, _ = build_features(tr, rows, items, users, single_col)
        scores = {"N1": rows["video_id"].map(impressions).fillna(0).to_numpy(float),
                  "A0": None, "shortest_first": -X["duration_s"].fillna(X["duration_s"].median()).to_numpy()}
        report["evaluations"][name] = {}
        saved = {}
        for lab in LABELS:
            scores["A0"] = arm_scores(arms[lab], X)
            saved[f"A0_{lab}"] = scores["A0"]
            report["evaluations"][name].update(evaluate(name, rows, X, scores, [lab]))
        np.savez_compressed(f"datastore/processed/exposed_vs_unbiased_{name}.npz",
                            user_id=rows["user_id"].to_numpy(), video_id=rows["video_id"].to_numpy(),
                            dur_band20=X["dur_band20"].to_numpy(),
                            duration_s=X["duration_s"].to_numpy(), N1=scores["N1"],
                            shortest_first=scores["shortest_first"],
                            **{f"y_{lab}": rows[lab].to_numpy() for lab in LABELS}, **saved)
        print(f"\n[{name}] rows {len(rows):,}, users {rows['user_id'].nunique():,}")

    OUT.write_text(json.dumps(report, indent=1, default=float))

    print(f"\n{'label':18s} {'evaluation':22s} {'users':>7s} {'pos rate':>9s} {'N1':>7s} {'A0':>7s} "
          f"{'A0-N1':>8s} {'95% CI':>19s} {'p':>7s}")
    for lab in LABELS:
        for name in report["evaluations"]:
            r = report["evaluations"][name][lab]
            c = r["A0_minus_N1"]
            print(f"{lab:18s} {name:22s} {r['users_both_classes']:7,} {r['positive_rate']:9.4f} "
                  f"{r['gauc']['N1']:7.4f} {r['gauc']['A0']:7.4f} {c['diff']:+8.4f} "
                  f"[{c['ci_low']:+.4f},{c['ci_high']:+.4f}] {c['p']:7.4f} "
                  f"dur-demeaned {c['duration_demeaned_diff']:+.4f} "
                  f"(N1 {r['gauc_duration_demeaned']['N1']:.4f} A0 {r['gauc_duration_demeaned']['A0']:.4f})")


if __name__ == "__main__":
    main()
