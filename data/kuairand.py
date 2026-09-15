"""
KuaiRand-Pure readers (phase 2).

Source: https://github.com/chongminggao/KuaiRand (Zenodo record 10439422,
KuaiRand-Pure.tar.gz, md5 0820331067a3784d9691136f772b35a7). Extract into
datastore/raw/kuairand/; files are located by name, so the nesting inside the
archive does not matter.

Phase 0 only needs raw logs and side tables, so this module exposes plain
readers rather than a BaseDataLoader subclass. BaseDataLoader builds id maps
from the interactions it loads and splits on a global time percentile; both
are wrong for KuaiRand (see docs/phase2_kuairand_plan.md, "Code changes the
loader needs"). Raw ids are kept throughout.

Label semantics that matter (README):
  is_click   two-column UI: a real click. Single-column UI: *valid_play*, a
             watch-time threshold (see valid_play()).
  long_view  a watch-time threshold (see long_view_rule()).
  profile_stay_time, comment_stay_time, is_profile_enter are recorded after
  the impression and must never be features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("datastore/raw/kuairand")

LOG_FILES = {
    "standard_early": "log_standard_4_08_to_4_21_pure.csv",   # train window
    "standard_late": "log_standard_4_22_to_5_08_pure.csv",    # contrast only
    "random": "log_random_4_22_to_5_08_pure.csv",             # evaluation
}

LABELS = ["is_click", "is_like", "is_follow", "is_comment", "is_forward", "is_hate", "long_view"]
EXPLICIT = ["is_like", "is_follow", "is_comment", "is_forward"]
POST_IMPRESSION = ["profile_stay_time", "comment_stay_time", "is_profile_enter"]

_LOG_DTYPES = {
    "user_id": "int32", "video_id": "int32", "date": "int32", "hourmin": "int16",
    "time_ms": "int64", "play_time_ms": "int64", "duration_ms": "int64",
    "profile_stay_time": "int64", "comment_stay_time": "int64",
    "is_profile_enter": "int8", "is_rand": "int8", "tab": "int8",
    **{c: "int8" for c in LABELS},
}


def find_file(name: str, raw_dir: Path = RAW_DIR) -> Path:
    hits = sorted(Path(raw_dir).rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {raw_dir}; download KuaiRand-Pure first")
    return hits[0]


def read_log(which: str, raw_dir: Path = RAW_DIR, dedup: bool = False) -> pd.DataFrame:
    """One log file. dedup=True keeps the first row per (user, video, time_ms):
    15,609 exact duplicate rows in the 4/09-4/21 standard log plus ~1,000 more
    that share the key but differ in date, click or play time."""
    df = pd.read_csv(find_file(LOG_FILES[which], raw_dir), dtype=_LOG_DTYPES)
    if dedup:
        df = df.drop_duplicates(["user_id", "video_id", "time_ms"]).reset_index(drop=True)
    df["explicit_positive"] = df[EXPLICIT].max(axis=1).astype("int8")
    return df


def read_user_features(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return pd.read_csv(find_file("user_features_pure.csv", raw_dir))


def read_video_basic(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """video_features_basic. video_duration is NaN for 239 items, and every log
    row for those items has duration_ms == 0 (no other item does), so their
    duration is unknown and stays NaN. Callers must handle it explicitly;
    per_user_auc rejects NaN scores."""
    return pd.read_csv(find_file("video_features_basic_pure.csv", raw_dir))


def valid_play(play_time_ms: np.ndarray, duration_ms: np.ndarray) -> np.ndarray:
    """README's valid_play: full play if duration <= 7 s, else play > 7 s."""
    play, dur = np.asarray(play_time_ms), np.asarray(duration_ms)
    return np.where(dur <= 7_000, play >= dur, play > 7_000)


def long_view_rule(play_time_ms: np.ndarray, duration_ms: np.ndarray) -> np.ndarray:
    """README's long_view: full play if duration <= 18 s, else play >= 18 s."""
    play, dur = np.asarray(play_time_ms), np.asarray(duration_ms)
    return np.where(dur <= 18_000, play >= dur, play >= 18_000)
