"""
Exposure-unbiased evaluation on KuaiRec's small_matrix.

Implements docs/evaluation_protocol.md (v2). Read that first: every choice here
— the label, the baselines, the metrics, the decision rule — was fixed before
any model was scored, and changing one is a logged protocol change.

Nothing from small_matrix feeds training, features, label statistics, or
baselines. Everything fitted here is fitted on the big_matrix train split.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata

from data.schema import Cols

N_BUCKETS = 50
N_BANDS = 5
CANDIDATES = 200
FRACS = (0.2, 0.3, 0.5)
PRIMARY_FRAC = 0.3
LABEL_CHECK_RANGE = (0.20, 0.45)
KNN_NEIGHBOURS = 50
KNN_SHRINK = 100.0


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def load_small_matrix(raw_dir, user_id_map: dict, item_id_map: dict) -> tuple[pd.DataFrame, dict]:
    """small_matrix pairs mapped to model ids, exact duplicates removed."""
    raw = pd.read_csv(Path(raw_dir) / "small_matrix.csv",
                      usecols=["user_id", "video_id", "timestamp", "watch_ratio"])
    n_raw = len(raw)
    raw = raw.drop_duplicates(subset=["user_id", "video_id", "timestamp"])
    n_dedup = len(raw)
    if raw.duplicated(subset=["user_id", "video_id"]).any():
        raise ValueError("small_matrix has repeated (user, video) pairs; the protocol assumes none")
    raw["u"] = raw["user_id"].map(user_id_map)
    raw["i"] = raw["video_id"].map(item_id_map)
    mapped = raw.dropna(subset=["u", "i"])
    info = {"rows_raw": n_raw, "rows_after_dedup": n_dedup, "rows_mapped": len(mapped),
            "users": int(mapped["u"].nunique()), "items": int(mapped["i"].nunique())}
    out = mapped[["u", "i", "watch_ratio"]].astype({"u": np.int64, "i": np.int64})
    return out.reset_index(drop=True), info


SPLIT_FILE = Path("config/small_matrix_user_split.json")


def split_users(user_ids, seed: int, val_frac: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic hash split of user ids into (validation, test)."""
    ids = np.asarray(sorted(set(int(u) for u in user_ids)))
    h = np.array([int(hashlib.md5(f"{seed}:{u}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF for u in ids])
    return ids[h < val_frac], ids[h >= val_frac]


def frozen_split(model_user_ids, user_id_map: dict, seed: int,
                 path: Path = SPLIT_FILE) -> tuple[np.ndarray, np.ndarray]:
    """Validation/test users, frozen by *raw* KuaiRec user id.

    Model ids are ranks among users that survive filtering, so any change to
    deduplication or cold-start thresholds renumbers them. Hashing model ids
    would then silently move users between validation and test. The split is
    computed once and stored as raw ids; every later run reads the file.
    """
    import json
    inverse = {int(v): int(k) for k, v in user_id_map.items()}
    model_ids = np.asarray(sorted(set(int(u) for u in model_user_ids)))
    if path.exists():
        stored = json.loads(path.read_text())
        val_raw, test_raw = set(stored["validation"]), set(stored["test"])
        val = np.array([u for u in model_ids if inverse[u] in val_raw])
        test = np.array([u for u in model_ids if inverse[u] in test_raw])
        missing = len(model_ids) - len(val) - len(test)
        if missing:
            raise ValueError(f"{missing} evaluation users are not in the frozen split file {path}")
        return val, test
    val, test = split_users(model_ids, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seed": seed, "note": "raw KuaiRec user_id; see docs/evaluation_protocol.md",
                                "validation": sorted(inverse[int(u)] for u in val),
                                "test": sorted(inverse[int(u)] for u in test)}, indent=1))
    return val, test


# ----------------------------------------------------------------------
# Label: within-user top fraction of within-duration-bucket percentiles
# ----------------------------------------------------------------------

@dataclass
class DurationModel:
    """Duration buckets and per-bucket watch_ratio distributions, fitted on train."""
    edges: np.ndarray            # interior bucket edges (N_BUCKETS - 1)
    item_duration: np.ndarray    # per item id; NaN if the item has no train rows
    sorted_wr: list              # per bucket, sorted train watch_ratio

    @classmethod
    def fit(cls, train: pd.DataFrame, n_items: int) -> "DurationModel":
        dur = train.groupby(Cols.ITEM_ID)["video_duration"].median()
        item_duration = np.full(n_items, np.nan)
        item_duration[dur.index.to_numpy()] = dur.to_numpy()
        edges = np.quantile(dur.to_numpy(), np.linspace(0, 1, N_BUCKETS + 1)[1:-1])
        model = cls(edges=edges, item_duration=item_duration, sorted_wr=[])
        b = model.bucket(train[Cols.ITEM_ID].to_numpy())
        wr = train[Cols.WATCH_RATIO].to_numpy(dtype=float)
        model.sorted_wr = [np.sort(wr[b == k]) for k in range(N_BUCKETS)]
        return model

    def bucket(self, items: np.ndarray) -> np.ndarray:
        d = self.item_duration[np.asarray(items)]
        out = np.searchsorted(self.edges, d, side="right")
        return np.where(np.isnan(d), -1, out)

    def percentile(self, items: np.ndarray, watch_ratio: np.ndarray) -> np.ndarray:
        """Mid-rank percentile of each watch_ratio within its item's bucket."""
        b = self.bucket(items)
        wr = np.asarray(watch_ratio, dtype=float)
        p = np.full(len(wr), np.nan)
        for k in range(N_BUCKETS):
            m = b == k
            if not m.any():
                continue
            ref = self.sorted_wr[k]
            lo = np.searchsorted(ref, wr[m], side="left")
            hi = np.searchsorted(ref, wr[m], side="right")
            p[m] = (lo + hi) / 2.0 / len(ref)
        return p


def within_user_top(users: np.ndarray, values: np.ndarray, frac: float) -> np.ndarray:
    """1 if the value is in the top `frac` of that user's values (average ranks)."""
    s = pd.Series(values).groupby(pd.Series(users)).rank(pct=True, method="average")
    return (s.to_numpy() > 1.0 - frac).astype(np.int8)


def build_labels(pairs: pd.DataFrame, model: DurationModel) -> tuple[pd.DataFrame, dict]:
    """Add bucket, percentile, primary/sensitivity labels and the raw label."""
    df = pairs.copy()
    df["bucket"] = model.bucket(df["i"].to_numpy())
    n_before = len(df)
    df = df[df["bucket"] >= 0].reset_index(drop=True)
    df["p"] = model.percentile(df["i"].to_numpy(), df["watch_ratio"].to_numpy())
    for f in FRACS:
        df[f"rel_{int(f * 100)}"] = within_user_top(df["u"].to_numpy(), df["p"].to_numpy(), f)
    df["rel_raw"] = (df["watch_ratio"] >= 0.7).astype(np.int8)
    return df, {"pairs_dropped_no_train_duration": n_before - len(df)}


def label_check(df: pd.DataFrame, frac: float = PRIMARY_FRAC) -> dict:
    rates = df.groupby("bucket")[f"rel_{int(frac * 100)}"].mean()
    lo, hi = LABEL_CHECK_RANGE
    return {"min_bucket_rate": float(rates.min()), "max_bucket_rate": float(rates.max()),
            "range": [lo, hi], "passed": bool(rates.min() >= lo and rates.max() <= hi),
            "corr_duration_label": float(np.corrcoef(
                np.nan_to_num(df["bucket"].to_numpy(float)), df[f"rel_{int(frac * 100)}"])[0, 1])}


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def tiebreak_order(scores: np.ndarray, tiebreak: np.ndarray) -> np.ndarray:
    """Indices sorted by descending score; ties broken by a fixed random key.

    Ties are common (popularity counts, durations). Breaking them by index
    order would inject item-id order, which tracks upload time.
    """
    return np.lexsort((tiebreak, -scores))


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    pos = labels.astype(bool)
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def ndcg_at(labels_in_order: np.ndarray, n_relevant: int, k: int = 10) -> float:
    gains = labels_in_order[:k].astype(float)
    dcg = (gains / np.log2(np.arange(2, len(gains) + 2))).sum()
    ideal = np.ones(min(k, n_relevant))
    idcg = (ideal / np.log2(np.arange(2, len(ideal) + 2))).sum()
    return float(dcg / idcg) if idcg > 0 else float("nan")


def two_stage_scores(retrieval: np.ndarray, ranker_on_candidates: np.ndarray,
                     candidates: np.ndarray) -> np.ndarray:
    """Full-list scores for the deployed shape.

    `candidates` indexes retrieval's top items; they are ordered by the ranker
    and placed above every other item, which keep retrieval order.
    """
    n = len(retrieval)
    out = np.empty(n)
    rest = np.setdiff1d(np.arange(n), candidates, assume_unique=True)
    rest_order = rest[np.argsort(-retrieval[rest], kind="stable")]
    out[rest_order] = -np.arange(len(rest_order), dtype=float) - len(candidates) - 1
    cand_order = candidates[np.argsort(-ranker_on_candidates, kind="stable")]
    out[cand_order] = -np.arange(len(cand_order), dtype=float)
    return out


def top_candidates(retrieval: np.ndarray, k: int = CANDIDATES) -> np.ndarray:
    k = min(k, len(retrieval))
    return np.argpartition(-retrieval, k - 1)[:k]


def user_metrics(labels: np.ndarray, scores: np.ndarray, tiebreak: np.ndarray) -> dict:
    order = tiebreak_order(scores, tiebreak)
    lab = labels[order]
    return {"auc": auc(labels, scores), "ndcg10": ndcg_at(lab, int(labels.sum())),
            "p10": float(lab[:10].mean())}


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------

def bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 42) -> dict:
    """Paired bootstrap over users for mean(b - a), with a two-sided p-value."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = ~(np.isnan(a) | np.isnan(b))
    d = (b - a)[keep]
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {"n": int(len(d)), "mean_a": float(a[keep].mean()), "mean_b": float(b[keep].mean()),
            "diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "p": float(min(1.0, max(p, 1.0 / n_boot)))}


def holm(pvalues: dict, alpha: float = 0.05) -> dict:
    """Holm step-down: {name: rejected?} controlling family-wise error."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    rejected, still = {}, True
    for rank, (name, p) in enumerate(items):
        still = still and p <= alpha / (m - rank)
        rejected[name] = bool(still)
    return rejected


# ----------------------------------------------------------------------
# Baselines (all fitted on big_matrix train)
# ----------------------------------------------------------------------

class Baselines:
    """B0–B6 from docs/evaluation_protocol.md §6."""

    def __init__(self, train: pd.DataFrame, model: DurationModel, n_users: int, n_items: int,
                 item_play_progress: np.ndarray, eval_users: np.ndarray, target_items: np.ndarray,
                 pos_threshold: float = 0.7, seed: int = 42):
        self.n_items = n_items
        self.seed = seed
        items = train[Cols.ITEM_ID].to_numpy()
        users = train[Cols.USER_ID].to_numpy()
        p = model.percentile(items, train[Cols.WATCH_RATIO].to_numpy())
        ok = ~np.isnan(p)
        items, users, p = items[ok], users[ok], p[ok]

        self.b1 = np.bincount(train.loc[train[Cols.WATCH_RATIO] >= pos_threshold, Cols.ITEM_ID],
                              minlength=n_items).astype(float)
        self.b2 = -np.nan_to_num(model.item_duration, nan=np.nanmax(model.item_duration))
        sums = np.bincount(items, weights=p, minlength=n_items)
        cnts = np.bincount(items, minlength=n_items)
        self.b3 = np.where(cnts > 0, sums / np.maximum(cnts, 1), 0.5)

        bucket = model.bucket(np.arange(n_items))
        self.b5 = np.full(n_items, 0.5)
        pp = np.asarray(item_play_progress, float)
        for k in range(N_BUCKETS):
            m = (bucket == k) & ~np.isnan(pp)
            if m.sum() > 1:
                self.b5[m] = rankdata(pp[m]) / m.sum()

        band_of_item = np.where(bucket >= 0, bucket * N_BANDS // N_BUCKETS, -1)
        band = band_of_item[items]
        user_mean = pd.Series(p).groupby(users).mean()
        self.user_mean = np.full(n_users, 0.5)
        self.user_mean[user_mean.index.to_numpy()] = user_mean.to_numpy()
        g = pd.DataFrame({"u": users, "band": band, "p": p}).groupby(["u", "band"])["p"].agg(["mean", "size"])
        self.b6 = np.repeat(self.user_mean[:, None], N_BANDS, axis=1)
        g = g[(g["size"] >= 5) & (g.index.get_level_values("band") >= 0)]
        self.b6[g.index.get_level_values("u"), g.index.get_level_values("band")] = g["mean"].to_numpy()
        self.band_of_item = band_of_item

        self.target_items = np.asarray(target_items)
        self.eval_users = np.asarray(eval_users)
        self.b4 = self._knn(users, items, p, n_users)

    def _knn(self, users, items, p, n_users):
        """Item-kNN on user-centred percentiles; returns eval_users x target_items."""
        df = pd.DataFrame({"u": users, "i": items, "p": p}).groupby(["u", "i"], as_index=False)["p"].mean()
        df["c"] = df["p"] - df.groupby("u")["p"].transform("mean")
        C = sparse.csr_matrix((df["c"].to_numpy(), (df["u"], df["i"])), shape=(n_users, self.n_items))
        B = sparse.csr_matrix((np.ones(len(df)), (df["u"], df["i"])), shape=(n_users, self.n_items))
        J = self.target_items
        norms = np.sqrt(np.asarray(C.multiply(C).sum(axis=0)).ravel())
        G = (C[:, J].T @ C).toarray().astype(np.float32)
        N = (B[:, J].T @ B).toarray().astype(np.float32)
        denom = np.outer(norms[J], norms).astype(np.float32)
        sim = np.divide(G, denom, out=np.zeros_like(G), where=denom > 0)
        del G, denom
        sim *= N / (N + KNN_SHRINK)
        del N
        sim[np.arange(len(J)), J] = 0.0
        k = min(KNN_NEIGHBOURS, sim.shape[1] - 1)
        nb = np.argpartition(-np.abs(sim), k - 1, axis=1)[:, :k]
        rows = np.repeat(np.arange(len(J)), k)
        S = sparse.csr_matrix((sim[rows, nb.ravel()], (rows, nb.ravel())), shape=(len(J), self.n_items))
        Ce, Be = C[self.eval_users], B[self.eval_users]
        num = (Ce @ S.T).toarray()
        den = (Be @ abs(S).T).toarray()
        return np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    def scores(self, name: str, user: int, items: np.ndarray) -> np.ndarray:
        if name == "B0 random":
            return np.random.default_rng(self.seed + int(user)).random(len(items))
        if name == "B1 popularity":
            return self.b1[items]
        if name == "B2 shortest-first":
            return self.b2[items]
        if name == "B3 item quality within duration":
            return self.b3[items]
        if name == "B4 item-kNN":
            row = np.searchsorted(self.eval_users, user)
            col = np.searchsorted(self.target_items, items)
            return self.b4[row, col]
        if name == "B5 platform completion within duration":
            return self.b5[items]
        if name == "B6 user duration preference":
            band = self.band_of_item[items]
            return np.where(band >= 0, self.b6[user, np.clip(band, 0, N_BANDS - 1)], self.user_mean[user])
        raise KeyError(name)

    NAMES = ("B0 random", "B1 popularity", "B2 shortest-first", "B3 item quality within duration",
             "B4 item-kNN", "B5 platform completion within duration", "B6 user duration preference")


# ----------------------------------------------------------------------
# Probe used for checkpoint selection (validation users only)
# ----------------------------------------------------------------------

class SmallMatrixProbe:
    """Primary-label evaluation on validation users, cheap enough to run per epoch."""

    def __init__(self, cfg: dict, train: pd.DataFrame, n_items: int, split: str = "val"):
        import pickle
        processed = Path(cfg["data"]["processed_dir"])
        with open(processed / "id_maps.pkl", "rb") as f:
            maps = pickle.load(f)
        pairs, _ = load_small_matrix(cfg["data"]["kuairec"]["raw_dir"], maps["user_id_map"], maps["item_id_map"])
        model = DurationModel.fit(train, n_items)
        df, _ = build_labels(pairs, model)
        val, test = frozen_split(df["u"].unique(), maps["user_id_map"], cfg["project"]["seed"])
        chosen = val if split == "val" else test
        df = df[df["u"].isin(chosen)]
        self.users, self.items, self.labels = [], [], []
        for u, g in df.groupby("u"):
            self.users.append(int(u))
            self.items.append(g["i"].to_numpy())
            self.labels.append(g[f"rel_{int(PRIMARY_FRAC * 100)}"].to_numpy())
        self.users = np.array(self.users)

    def mean_auc(self, score_fn) -> float:
        """score_fn(user, items) -> scores over the full list."""
        return float(np.nanmean([auc(lab, score_fn(u, it)) for u, it, lab in zip(self.users, self.items, self.labels)]))

    def mean_two_stage_ndcg(self, retrieval_fn, ranker_batch_fn) -> float:
        """retrieval_fn(user, items) -> scores; ranker_batch_fn(user, items) -> ranker scores."""
        vals = []
        for u, it, lab in zip(self.users, self.items, self.labels):
            ret = retrieval_fn(u, it)
            cand = top_candidates(ret)
            rk = ranker_batch_fn(u, it[cand])
            order = cand[np.argsort(-rk, kind="stable")]
            vals.append(ndcg_at(lab[order], int(lab.sum())))
        return float(np.nanmean(vals))
