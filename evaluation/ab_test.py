"""
Experiment statistics for comparing two ranking systems.

What changed and why
--------------------
The first version imitated a live A/B test: split users by a hash, score group A
with model A and group B with model B, compare the pooled metric. That mirrors
the *mechanics* of an online experiment but inherits its central limitation for
no reason.

A live A/B test splits traffic because it has to — the same person cannot be
shown two different feeds at the same moment, so the counterfactual is
unobservable, and you pay for it with between-group population variance.
Offline, that constraint does not exist: every model can be scored on every
user. Copying the split therefore discards half the data per model and *adds*
noise while answering a strictly weaker question.

So this module does the thing offline evaluation can do and online cannot:
a **paired** comparison on identical users, with a confidence interval.

Three capabilities:

  paired_bootstrap            is the difference real, or sampling noise?
  minimum_detectable_effect   how much traffic would a live test need?
  assign_group /              deterministic bucketing for real traffic splits,
  check_sample_ratio          plus the sample-ratio check that validates it

The motivating incident is in docs/engineering_log.md: a 150-user sample showed
the pipeline beating popularity by +0.7% at recall@20, and the full 1,411-user
set reversed that to -0.9%. A confidence interval would have flagged it
immediately instead of it being caught by chance on a re-run.
"""
import hashlib

import numpy as np


def assign_group(user_id: int, traffic_split: float = 0.5, seed: int = 42) -> str:
    """
    Deterministically assign a user to group 'A' or 'B'.

    MD5 of (seed + user_id) mod 1000 — used as a fast uniform hash, not for
    security. Deterministic assignment is what makes an experiment replayable:
    the same user lands in the same bucket across restarts and redeploys, so
    nobody flips variants mid-experiment.

    This is the function a serving layer calls to route real traffic.
    """
    h = int(hashlib.md5(f"{seed}{user_id}".encode()).hexdigest(), 16)
    return "A" if (h % 1000) < int(traffic_split * 1000) else "B"


def check_sample_ratio(user_ids, traffic_split: float = 0.5, seed: int = 42) -> dict:
    """
    Sample-ratio mismatch check on the bucketing.

    The first thing a real experimentation platform verifies: did the split land
    where it was supposed to? A skew usually means the assignment logic is
    broken — a bad hash, a filter applied after bucketing, one variant erroring
    out — and every downstream number is then untrustworthy.

    Returns observed counts and a z score; |z| > 3 is the conventional alarm.
    """
    groups = [assign_group(int(u), traffic_split, seed) for u in user_ids]
    n_a = sum(1 for g in groups if g == "A")
    n = len(groups)
    expected_a = n * traffic_split
    sd = np.sqrt(n * traffic_split * (1 - traffic_split))
    z = (n_a - expected_a) / sd if sd > 0 else 0.0

    return {
        "n": n,
        "n_a": n_a,
        "n_b": n - n_a,
        "observed_split": n_a / n if n else float("nan"),
        "expected_split": traffic_split,
        "z": float(z),
        "srm_suspected": bool(abs(z) > 3),
    }


def paired_bootstrap(a, b, n_boot: int = 10_000, confidence: float = 0.95,
                     seed: int = 42) -> dict:
    """
    Bootstrap confidence interval for the per-user difference b - a.

    Pairing matters here. Users differ enormously in how easy they are to
    recommend for, and that between-user variance dwarfs the effect being
    measured. Comparing group means throws that structure away; differencing
    within each user removes it, so this detects effects an unpaired test would
    miss at the same sample size.

    Args:
        a, b:       per-user metric values for the two systems — same users,
                    same order.
        n_boot:     resamples.
        confidence: interval width, e.g. 0.95.

    Returns:
        mean difference, CI bounds, relative lift, and whether the CI excludes 0.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            f"paired comparison needs equal shapes, got {a.shape} and {b.shape}"
        )
    if len(a) == 0:
        raise ValueError("no observations to compare")

    diff = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot_means = diff[idx].mean(axis=1)

    alpha = 1 - confidence
    lo, hi = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])
    base = a.mean()

    return {
        "n_users": len(diff),
        "mean_a": float(base),
        "mean_b": float(b.mean()),
        "mean_diff": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "relative_lift": float(diff.mean() / base) if base else float("nan"),
        "significant": bool(lo > 0 or hi < 0),
        "confidence": confidence,
    }


def minimum_detectable_effect(values, n_users: int = None, power: float = 0.8,
                              alpha: float = 0.05) -> dict:
    """
    Smallest effect a live experiment could detect, given observed variance.

    Answers the question that precedes shipping any model: how much traffic does
    this need? Normal approximation for a two-sample comparison, with
    z(1-alpha/2) + z(power) = 1.96 + 0.84 = 2.80 at the conventional settings.

    Args:
        values:  per-user metric values, used for their standard deviation.
        n_users: users available per arm; defaults to len(values) split in two.

    Returns:
        Absolute and relative MDE, plus users-per-arm needed for a 5% and a 10%
        relative lift.
    """
    values = np.asarray(values, dtype=float)
    mean = values.mean()
    sd = values.std(ddof=1)
    per_arm = n_users if n_users else len(values) // 2

    z_sum = 2.80  # 1.96 (alpha=0.05, two-sided) + 0.84 (power=0.8)
    mde_abs = z_sum * sd * np.sqrt(2.0 / per_arm) if per_arm else float("nan")

    def users_for(relative):
        target = relative * mean
        return int(np.ceil(2 * (z_sum * sd / target) ** 2)) if target else -1

    return {
        "mean": float(mean),
        "sd": float(sd),
        "users_per_arm": int(per_arm),
        "mde_absolute": float(mde_abs),
        "mde_relative": float(mde_abs / mean) if mean else float("nan"),
        "users_for_5pct_lift": users_for(0.05),
        "users_for_10pct_lift": users_for(0.10),
        "power": power,
        "alpha": alpha,
    }
