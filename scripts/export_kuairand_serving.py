"""
Export the artifacts the KuaiRand scorer serves (docs/phase2_exposure_bias_plan.md).

The deployed scorer is the **item impression count** from the recommender-exposed
standard log. Phase 2 deploys whichever scorer wins under the protocol, even a
non-personal one: on held-out users no model arm beat this count by a material
margin (valid play +0.0009, p 0.59; long view +0.0021, p 0.32), so the simpler
scorer ships and the API says so.

Writes to datastore/serving/:
    kuairand_items.parquet   item_id, impressions, duration_s, rank
    kuairand_seen.npz        CSR-style (user -> items already shown in training)
    kuairand_meta.json       scorer identity, measured numbers, provenance
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from data.kuairand import read_log, read_video_basic

OUT = Path("datastore/serving")
AUDIT = Path("datastore/processed/kuairand_audit.json")
RESULT = Path("datastore/processed/kuairand_exposure_bias.json")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    single_col = json.loads(AUDIT.read_text())["single_column_tabs_inferred"]
    tr = read_log("standard_early", dedup=True)
    basic = read_video_basic()

    impressions = tr[tr["tab"].isin(single_col)]["video_id"].value_counts()
    items = basic[["video_id", "video_duration"]].rename(columns={"video_id": "item_id"})
    items["impressions"] = items["item_id"].map(impressions).fillna(0).astype(int)
    items["duration_s"] = items["video_duration"] / 1000     # NaN for 239 unknown-duration items
    # Ties broken by item_id so the ordering is stable across rebuilds.
    items = items.sort_values(["impressions", "item_id"], ascending=[False, True]).reset_index(drop=True)
    items["rank"] = np.arange(1, len(items) + 1)
    items[["item_id", "impressions", "duration_s", "rank"]].to_parquet(OUT / "kuairand_items.parquet", index=False)

    seen = tr[["user_id", "video_id"]].drop_duplicates().sort_values(["user_id", "video_id"])
    users, counts = np.unique(seen["user_id"].to_numpy(), return_counts=True)
    np.savez_compressed(OUT / "kuairand_seen.npz", users=users.astype(np.int32),
                        offsets=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
                        items=seen["video_id"].to_numpy().astype(np.int32))

    res = json.loads(RESULT.read_text())["results_primary_tabs"]
    # git is not installed in the container; the caller passes the commit in.
    commit = os.environ.get("GIT_COMMIT", "unknown")
    (OUT / "kuairand_meta.json").write_text(json.dumps({
        "scorer": "item_impressions",
        "description": "Item impression count in the KuaiRand standard log 4/09-4/21, "
                       "single-column tabs. Non-personal: the ranking is the same for every "
                       "user; only the filter of already-seen items is per user.",
        "why_this_scorer": "Pre-registered rule: deploy whichever scorer wins under the "
                           "protocol, even a baseline. On held-out users the personalised "
                           "LightGBM did not beat it by a material margin.",
        "measured_on_held_out_users": {
            lab: {"item_impressions_gauc": res[lab]["gauc"]["N1"],
                  "personal_model_gauc": res[lab]["gauc"]["A0"],
                  "difference": res[lab]["comparisons"]["H5 A0(seed42)-N1"]["diff"],
                  "p": res[lab]["comparisons"]["H5 A0(seed42)-N1"]["p"],
                  "users": res[lab]["users_both_classes"]}
            for lab in ["is_click", "long_view"]},
        "label_note": "is_click is KuaiRand's valid-play threshold, not a click.",
        "metric": "per-user AUC on randomly exposed impressions, test users, scored once",
        "protocol": "docs/phase2_exposure_bias_plan.md",
        "catalogue_size": int(len(items)),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "built_from_commit": commit,
    }, indent=1))

    print(f"items {len(items):,}  impressions max {items['impressions'].max():,}  "
          f"zero-impression items {(items['impressions'] == 0).sum():,}")
    print(f"seen pairs {len(seen):,} for {len(users):,} users")
    print((OUT / "kuairand_meta.json").read_text())


if __name__ == "__main__":
    main()
