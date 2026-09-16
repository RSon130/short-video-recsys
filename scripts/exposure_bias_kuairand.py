"""
Phase 2b: exposure bias on KuaiRand. Pre-registered in
docs/phase2_exposure_bias_plan.md (v2); every threshold here follows it.

Deliverable 1  does the sign flip replicate on test users? Model-free.
               (computed after the model scores are persisted, so a crash
               there never costs a re-score of test users)
Deliverable 2  do three standard corrections improve per-user ranking under
               random exposure? Arms A0, A1, A2, A3, A0', A3s vs comparators
               N1, N3.

Test users are scored once. Run order: --bug-check (validation invariants
only, no arm-vs-arm random-log differences), review, commit, then the real run.

Usage
  python scripts/exposure_bias_kuairand.py --bug-check
  python scripts/exposure_bias_kuairand.py
"""
import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from data.kuairand import read_log, read_user_features, read_video_basic
from evaluation.gauc import paired_bootstrap, per_user_auc, users_with_both_classes
from evaluation.small_matrix import holm
from scripts.gate_kuairand import (AUDIT, PARAMS, ROUNDS, SPLIT, build_features, item_table,
                                   smooth, user_table)

OUT = Path("datastore/processed/kuairand_exposure_bias.json")
FEATURE_LISTS = Path("datastore/processed/kuairand_exposure_bias_features.json")
GATE_RESULT = Path("datastore/processed/kuairand_gate.json")

SEEDS = [42, 43, 44]
N_BOOT = 10_000
MIN_EFFECT = 0.005
PRIMARY = ["is_click", "long_view"]
SECONDARY = ["explicit_positive", "is_like"]
LABELS = PRIMARY + SECONDARY
PRIMARY_TABS = [1, 2]
FIT_DAYS = list(range(20220413, 20220421))
LATE_START = 20220422
IPS_EXPONENT, IPS_CLIP_Q = 0.5, 0.99
A2_DROP = ("item_impr_share", "author_impr_share", "utag_share",
           "uauth_share", "uband_share", "user_single_col_share")
FLIP_FEATURES = ["item_impr_share", "author_impr_share", "utag_share"]
DECISIONS = ["H1 A1-A0", "H2 A2-A0", "H3s A3s-A0'", "H4 A3-N3", "H5 A0(seed42)-N1"]
DESCRIPTIVE = ["A3-A0", "N1-N3", "A1-N1", "A2-N1", "A3-N1"]


def fit_matrix_hash(X, cols):
    return hashlib.md5(np.ascontiguousarray(X[cols].to_numpy(dtype="float64")).tobytes()).hexdigest()[:12]


def ips_weights(hist_single, rows):
    """w = p_i^-0.5 from the row's own history window, clipped at that day's q99
    and normalised to mean 1. Returns (weights, clipped_mask); the clip is per
    label day, not global."""
    n_i = hist_single["video_id"].value_counts()
    p = (rows["video_id"].map(n_i).fillna(0) + 1) / (len(hist_single) + n_i.index.size)
    w = p.to_numpy() ** (-IPS_EXPONENT)
    cap = np.quantile(w, IPS_CLIP_Q)
    clipped = w > cap
    w = np.minimum(w, cap)
    return w / w.mean(), clipped


def train(X, y, cols, seed, weight=None):
    # Only `seed` is set: LightGBM derives bagging/feature/data seeds from it, so
    # seed 42 reproduces gate run 3 exactly (an invariant checked in --bug-check).
    params = {**PARAMS, "seed": seed, "num_threads": 4}
    return lgb.train(params, lgb.Dataset(X[cols], y, weight=weight), ROUNDS)


def arm_scores(arm, Xeval, per_seed=False):
    """Mean prediction over seeds; optionally each seed's own predictions."""
    preds = [b.predict(Xeval[arm["cols"]]) for b in arm["boosters"]]
    return (np.mean(preds, axis=0), preds) if per_seed else np.mean(preds, axis=0)


def duration_terciles(duration_s: pd.Series) -> np.ndarray:
    """Three duration groups; unknown duration (NaN) is its own group, -1."""
    t = pd.Series(-1, index=duration_s.index)
    known = duration_s.notna()
    t[known] = pd.qcut(duration_s[known].rank(method="first"), 3, labels=False)
    return t.to_numpy()


def band_demean(scores, bands):
    s = pd.Series(scores)
    return (s - s.groupby(np.asarray(bands)).transform("mean")).to_numpy()


def single_feature_gauc(X, rows, feature, label, tabs=None):
    m = rows["tab"].isin(tabs).to_numpy() if tabs else np.ones(len(rows), bool)
    v = X.loc[m, feature]
    keep = v.notna().to_numpy()
    r = rows[m][keep]
    return per_user_auc(r["user_id"].to_numpy(), r[label].to_numpy(), v[keep].to_numpy())


def boot_mean_ci(values, seed=7):
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    out = np.empty(N_BOOT)
    for i in range(0, N_BOOT, 500):
        k = min(500, N_BOOT - i)
        out[i:i + k] = v[rng.integers(0, len(v), size=(k, len(v)))].mean(axis=1)
    return float(v.mean()), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bug-check", action="store_true",
                    help="validation invariants only; prints no arm-vs-arm random-log difference")
    args = ap.parse_args()

    audit = json.loads(AUDIT.read_text())
    single_col = audit["single_column_tabs_inferred"]
    split = json.loads(SPLIT.read_text())
    val_users, test_users = set(split["validation"]), set(split["test"])

    tr = read_log("standard_early", dedup=True)
    late = read_log("standard_late", dedup=True)
    rnd = read_log("random", dedup=True)
    items, users = item_table(read_video_basic()), user_table(read_user_features())
    inv = {"lightgbm": lgb.__version__, "seeds": SEEDS, "rounds": ROUNDS,
           "params": {**PARAMS, "num_threads": 4}}

    # ---- fit sets -------------------------------------------------------
    parts, labs, weights, clips, meta = [], [], [], [], []
    for day in FIT_DAYS:
        rows = tr[(tr["date"] == day) & tr["tab"].isin(single_col)].reset_index(drop=True)
        hist = tr[tr["date"] < day]
        X, np_cols, p_cols = build_features(hist, rows, items, users, single_col)
        w, clipped = ips_weights(hist[hist["tab"].isin(single_col)], rows)
        parts.append(X)
        labs.append(rows[LABELS])
        weights.append(w)
        clips.append(clipped)
        meta.append(rows[["user_id", "tab"]])
    X_a0, y_a0 = pd.concat(parts, ignore_index=True), pd.concat(labs, ignore_index=True)
    w_a1 = np.concatenate(weights)
    w_a1 = w_a1 / w_a1.mean()
    clipped_frac = float(np.concatenate(clips).mean())
    fit_meta = pd.concat(meta, ignore_index=True)

    def val_rows(df):
        return df[df["user_id"].isin(val_users) & df["tab"].isin(single_col)].reset_index(drop=True)

    rows_a3 = val_rows(rnd[rnd["date"] >= LATE_START])
    rows_a0p = val_rows(late)
    X_a3, _, _ = build_features(tr, rows_a3, items, users, single_col)
    X_a0p, _, _ = build_features(tr, rows_a0p, items, users, single_col)
    sub = np.random.default_rng(42).choice(len(rows_a3), size=len(rows_a0p), replace=False)
    X_a3s, rows_a3s = X_a3.iloc[sub].reset_index(drop=True), rows_a3.iloc[sub].reset_index(drop=True)

    a2_cols = [c for c in p_cols if c not in A2_DROP]
    FEATURE_LISTS.write_text(json.dumps({"personal": p_cols, "nonpersonal": np_cols, "a2": a2_cols}, indent=1))

    assert not (set(rows_a3["user_id"]) & test_users) and not (set(rows_a0p["user_id"]) & test_users)
    assert len(rows_a3s) == len(rows_a0p) and len(set(sub)) == len(sub) and sub.max() < len(rows_a3)
    assert (rows_a3s["time_ms"].to_numpy() == rows_a3["time_ms"].to_numpy()[sub]).all()   # A3s subset of A3
    assert set(a2_cols) < set(p_cols) and len(p_cols) - len(a2_cols) == len(A2_DROP)

    specs = {
        "A0": dict(X=X_a0, y=y_a0, cols=p_cols, w=None),
        "A1": dict(X=X_a0, y=y_a0, cols=p_cols, w=w_a1),
        "A2": dict(X=X_a0, y=y_a0, cols=a2_cols, w=None),
        "A3": dict(X=X_a3, y=rows_a3[LABELS], cols=p_cols, w=None),
        "A0'": dict(X=X_a0p, y=rows_a0p[LABELS], cols=p_cols, w=None),
        "A3s": dict(X=X_a3s, y=rows_a3s[LABELS], cols=p_cols, w=None),
    }
    inv["fit_sets"] = {k: {"rows": len(v["X"]), "features": len(v["cols"]),
                           "hash": fit_matrix_hash(v["X"], v["cols"]),
                           "positives": {c: int(v["y"][c].sum()) for c in LABELS}}
                       for k, v in specs.items()}
    inv["ips"] = {"exponent": IPS_EXPONENT, "clip_q": IPS_CLIP_Q, "mean": float(w_a1.mean()),
                  "max": float(w_a1.max()), "ess_frac": float(w_a1.sum() ** 2 / (w_a1 ** 2).sum() / len(w_a1)),
                  "clipped_frac": clipped_frac, "clip_note": "q99 per label day, not global"}
    inv["fit_rows_include_test_users_exposed_rows"] = {
        "A0/A1/A2": True, "A3": False, "A0'": False,
        "note": "A0-A2 fit on the standard log's rows for all users, as gate run 3 did; "
                "test users' random-log rows and labels are never used for fitting"}
    inv["h3s_positive_rate_note"] = (
        "A3s and A0' have the same row count but different base rates by construction "
        "(random vs exposed exposure). Any A3s-A0' difference conflates the exposure "
        "mechanism with the number of training positives.")
    inv["nan_frac_fit"] = {k: round(float(v["X"][v["cols"]].isna().to_numpy().mean()), 4) for k, v in specs.items()}
    print(json.dumps(inv, indent=1, default=str))
    Path("datastore/processed/kuairand_exposure_bias_invariants.json").write_text(
        json.dumps(inv, indent=1, default=str))

    # ---- train ----------------------------------------------------------
    arms = {}
    for name, sp in specs.items():
        for lab in LABELS:
            arms[(name, lab)] = {"cols": sp["cols"],
                                 "boosters": [train(sp["X"], sp["y"][lab], sp["cols"], s, sp["w"]) for s in SEEDS]}
            print(f"  trained {name:4s} {lab:18s} rows {len(sp['X']):>7,} feats {len(sp['cols'])}")

    # ---- evaluation rows ------------------------------------------------
    if args.bug_check:
        ev = rnd[rnd["user_id"].isin(val_users)].reset_index(drop=True)      # invariants only
        diag = tr[(tr["date"] == 20220421) & tr["tab"].isin(single_col)].reset_index(drop=True)
        Xdiag, _, _ = build_features(tr[tr["date"] < 20220421], diag, items, users, single_col)
        Xev, _, _ = build_features(tr, ev, items, users, single_col)
        gate = json.loads(GATE_RESULT.read_text())
        out = {"invariants": inv, "prediction_hashes": {}, "diag_4_21_gauc": {}}
        for (name, lab), arm in arms.items():
            p = arm_scores(arm, Xev)
            out["prediction_hashes"][f"{name}/{lab}"] = hashlib.md5(p.tobytes()).hexdigest()[:12]
            d = arm_scores(arm, Xdiag)
            out["diag_4_21_gauc"][f"{name}/{lab}"] = round(
                float(per_user_auc(diag["user_id"].to_numpy(), diag[lab].to_numpy(), d).mean()), 4)
        # A0 seed 42 must reproduce gate run 3 exactly on validation rows
        for lab in LABELS:
            b = arms[("A0", lab)]["boosters"][0]
            g = float(per_user_auc(ev["user_id"].to_numpy(), ev[lab].to_numpy(),
                                   b.predict(Xev[p_cols])).mean())
            out.setdefault("A0_seed42_vs_gate_run3", {})[lab] = {
                "now": round(g, 4), "gate": gate["results"][lab]["gauc"]["lgbm_personal"],
                "match": abs(g - gate["results"][lab]["gauc"]["lgbm_personal"]) < 5e-4}
        print(json.dumps(out, indent=1))
        Path("datastore/processed/kuairand_exposure_bias_bugcheck.json").write_text(json.dumps(out, indent=1))
        return

    ev_all = rnd[rnd["user_id"].isin(test_users)].reset_index(drop=True)
    Xev_all, _, _ = build_features(tr, ev_all, items, users, single_col)
    prim = ev_all["tab"].isin(PRIMARY_TABS).to_numpy()

    # ---- deliverable 2: arms -------------------------------------------
    hist1 = tr[tr["tab"].isin(single_col)]
    n1 = ev_all["video_id"].map(hist1["video_id"].value_counts()).fillna(0).to_numpy()
    scores, per_seed = {}, {}
    for lab in LABELS:
        s = {}
        for name in specs:
            s[name], ps = arm_scores(arms[(name, lab)], Xev_all, per_seed=True)
            per_seed[f"{name}/{lab}"] = ps
        s["A0_seed42"] = per_seed[f"A0/{lab}"][0]
        s["N1"] = n1
        prior = rows_a3[lab].mean()
        g = rows_a3.groupby("video_id")[lab].agg(["sum", "count"])
        mapped = ev_all["video_id"].map(smooth(g["sum"], g["count"], prior, 20))
        s["N3"] = mapped.fillna(prior).to_numpy()
        scores[lab] = s
    inv["comparator_coverage"] = {"n3_prior_frac": round(float(mapped.isna().mean()), 4),
                                  "n1_zero_frac": round(float((n1 == 0).mean()), 4)}
    # persist scores before the analysis, so a crash never costs a re-score of test users
    np.savez_compressed("datastore/processed/kuairand_exposure_bias_scores.npz",
                        user_id=ev_all["user_id"].to_numpy(), video_id=ev_all["video_id"].to_numpy(),
                        tab=ev_all["tab"].to_numpy(), dur_band20=Xev_all["dur_band20"].to_numpy(),
                        duration_s=Xev_all["duration_s"].to_numpy(),
                        **{f"y_{lab}": ev_all[lab].to_numpy() for lab in LABELS},
                        **{f"s_{lab}_{k}": v for lab in LABELS for k, v in scores[lab].items()},
                        **{f"seed_{k}_{lab}_{i}": p for k in specs for lab in LABELS
                           for i, p in enumerate(per_seed[f"{k}/{lab}"])})

    pairs = {"H1 A1-A0": ("A0", "A1"), "H2 A2-A0": ("A0", "A2"), "H3s A3s-A0'": ("A0'", "A3s"),
             "H4 A3-N3": ("N3", "A3"), "H5 A0(seed42)-N1": ("N1", "A0_seed42"),
             "A3-A0": ("A0", "A3"), "N1-N3": ("N3", "N1"), "A1-N1": ("N1", "A1"),
             "A2-N1": ("N1", "A2"), "A3-N1": ("N1", "A3")}

    # ---- deliverable 1: sign-flip replication ---------------------------
    flip = {}
    std_rows = fit_meta.assign(**{c: y_a0[c].to_numpy() for c in LABELS})
    std_prim = std_rows["tab"].isin(PRIMARY_TABS).to_numpy()
    std_no_test = ~std_rows["user_id"].isin(test_users).to_numpy()
    for feat in FLIP_FEATURES:
        for lab in LABELS:
            # both sides restricted to the primary tabs, so the two AUCs are comparable
            a = single_feature_gauc(X_a0, std_rows, feat, lab, PRIMARY_TABS)
            b = single_feature_gauc(Xev_all, ev_all, feat, lab, PRIMARY_TABS)
            a_nt = single_feature_gauc(X_a0[std_no_test].reset_index(drop=True),
                                       std_rows[std_no_test].reset_index(drop=True), feat, lab, PRIMARY_TABS)
            ma, ba = boot_mean_ci(a.to_numpy(), seed=11)
            mb, bb = boot_mean_ci(b.to_numpy(), seed=12)
            d = bb - ba
            lo, hi = np.quantile(d, [0.025, 0.975])
            flip[f"{feat}/{lab}"] = {
                "standard_fit_gauc": round(ma, 4), "test_random_gauc": round(mb, 4),
                "diff": round(mb - ma, 4), "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4),
                "users_standard": int(len(a)), "users_test_random": int(len(b)),
                "rows_kept_frac_standard": round(float(X_a0.loc[std_prim, feat].notna().mean()), 4),
                "rows_kept_frac_random": round(float(Xev_all.loc[prim, feat].notna().mean()), 4),
                # the standard side includes test users' *exposed* rows, so the two user
                # sets overlap and the unpaired bootstrap is conservative
                "standard_gauc_excluding_test_users": round(float(a_nt.mean()), 4),
                "replicated_flip": bool(lo > 0 and ma < 0.5 < mb) if lab in SECONDARY else None,
                "note": "extension beyond engineering log section 18, which reported explicit feedback only"
                        if lab == "is_like" else None}

    terc_all = duration_terciles(Xev_all["duration_s"])

    def compare(lab, rows_mask):
        r = ev_all[rows_mask]
        u, y = r["user_id"].to_numpy(), r[lab].to_numpy()
        bands = Xev_all.loc[rows_mask, "dur_band20"].to_numpy()
        terc = terc_all[rows_mask]
        pua = {k: per_user_auc(u, y, v[rows_mask]) for k, v in scores[lab].items()}

        def seed_gauc(name):
            if name not in specs:
                return None
            return [float(per_user_auc(u, y, ps[rows_mask]).mean()) for ps in per_seed[f"{name}/{lab}"]]

        res = {"users_both_classes": users_with_both_classes(u, y), "positives": int(y.sum()),
               "gauc": {k: round(float(v.mean()), 4) for k, v in pua.items()}, "comparisons": {}}
        for name, (a, b) in pairs.items():
            c = paired_bootstrap(pua[a], pua[b], n_boot=N_BOOT)
            dm = {k: per_user_auc(u, y, band_demean(scores[lab][k][rows_mask], bands)) for k in (a, b)}
            dd = float(pd.concat([dm[a].rename("a"), dm[b].rename("b")], axis=1).dropna()
                       .pipe(lambda j: (j["b"] - j["a"]).mean()))
            c["duration_demeaned_diff"] = round(dd, 4)
            # pre-registered guard: with scores demeaned within 20 duration bands, a
            # claim must keep its sign and at least half its magnitude
            c["duration_guard_pass"] = bool(np.sign(dd) == np.sign(c["diff"])
                                            and abs(dd) >= 0.5 * abs(c["diff"]))
            c["duration_terciles"] = {}
            for t in sorted(set(terc.tolist())):
                m = terc == t
                if m.sum() and len(set(y[m].tolist())) > 1:
                    j = pd.concat([per_user_auc(u[m], y[m], scores[lab][a][rows_mask][m]).rename("a"),
                                   per_user_auc(u[m], y[m], scores[lab][b][rows_mask][m]).rename("b")],
                                  axis=1).dropna()
                    c["duration_terciles"][int(t)] = {"users": int(len(j)),
                                                      "diff": round(float((j["b"] - j["a"]).mean()), 4)}
            c["equivalent_within_0.005"] = bool(c["ci_low"] > -MIN_EFFECT and c["ci_high"] < MIN_EFFECT)
            if name in DECISIONS:
                sa, sb = seed_gauc(a), seed_gauc(b)
                if sa and sb:
                    c["per_seed_diff"] = [round(sb[i] - sa[i], 4) for i in range(len(SEEDS))]
                elif sb:
                    c["per_seed_diff"] = [round(sb[i] - float(pua[a].mean()), 4) for i in range(len(SEEDS))]
                elif sa:
                    c["per_seed_diff"] = [round(float(pua[b].mean()) - sa[i], 4) for i in range(len(SEEDS))]
            res["comparisons"][name] = c
        return res

    results = {lab: compare(lab, prim) for lab in LABELS}
    sens = {lab: compare(lab, np.ones(len(ev_all), bool)) for lab in PRIMARY}

    pv = {f"{lab} | {n}": results[lab]["comparisons"][n]["p"] for lab in PRIMARY for n in DECISIONS}
    rejected = holm(pv)
    def claim(key):
        lab, name = key.split(" | ")
        c = results[lab]["comparisons"][name]
        return bool(rejected[key] and c["diff"] >= MIN_EFFECT and c["duration_guard_pass"])

    claims = {k: claim(k) for k in pv}

    OUT.write_text(json.dumps({"invariants": inv, "sign_flip_replication": flip, "results_primary_tabs": results,
                               "sensitivity_all_tabs": sens, "holm_rejected": rejected, "claims": claims,
                               "h3s_positive_rate_note": inv["h3s_positive_rate_note"]},
                              indent=1, default=float))

    print("\n[Deliverable 1] sign-flip replication: single-feature per-user AUC")
    print(f"{'feature / label':40s} {'standard fit':>12s} {'test random':>12s} {'diff':>8s} {'95% CI':>19s} {'flip?':>6s}")
    for k, v in flip.items():
        print(f"{k:40s} {v['standard_fit_gauc']:12.4f} {v['test_random_gauc']:12.4f} {v['diff']:+8.4f} "
              f"[{v['ci_low']:+.4f},{v['ci_high']:+.4f}] {str(v['replicated_flip']):>6s}")

    for title, block, labs_ in [("primary: test users, random log, tabs 1-2", results, LABELS),
                                ("sensitivity: all tabs", sens, PRIMARY)]:
        print(f"\n[{title}]")
        for lab in labs_:
            r = block[lab]
            print(f"  {lab} ({r['users_both_classes']:,} users): " +
                  ", ".join(f"{k} {v:.4f}" for k, v in r["gauc"].items()))
            for n in DECISIONS + DESCRIPTIVE:
                c = r["comparisons"][n]
                mark = ""
                if lab in PRIMARY and n in DECISIONS:
                    key = f"{lab} | {n}"
                    if claims[key]:
                        mark = "  CLAIM"
                    elif rejected[key] and c["diff"] >= MIN_EFFECT:
                        mark = "  (significant, fails the duration guard)"
                    elif rejected[key]:
                        mark = "  (significant, below 0.005)"
                    if c["equivalent_within_0.005"]:
                        mark += "  [no material difference]"
                print(f"    {n:20s} {c['diff']:+.4f} [{c['ci_low']:+.4f},{c['ci_high']:+.4f}] p {c['p']:.4f} "
                      f"dur-demeaned {c['duration_demeaned_diff']:+.4f}"
                      f"{' seeds ' + str(c['per_seed_diff']) if 'per_seed_diff' in c else ''}{mark}")
    print(f"\nHolm rejected: {rejected}\nClaims: {claims}")


if __name__ == "__main__":
    main()
