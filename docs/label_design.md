# Label design — is `watch_ratio >= 0.7` the right positive?

Short answer: **it is defensible but confounded.** The label is substantially
measuring how long the video is rather than whether the user liked it.

Everything below is measured on `big_matrix`, which is what actually trains this
system.

---

## Which matrix trains the models

`big_matrix`, confirmed three independent ways:

| source | evidence |
|---|---|
| `config/kuairec.yaml` | `interaction_file: big_matrix.csv` |
| `datastore/processed/ranker_meta.json` | `"interaction_file": "big_matrix.csv"` |
| artifact dimensions | 7,176 users × 9,958 items (small is 1,411 × 3,327) |

`small_matrix` was used earlier in the project and its results are retained for
the dense-vs-sparse comparison, but every current artifact is `big_matrix`.

---

## The thresholds were justified with the wrong dataset's numbers

`config/base.yaml` justified the 0.7/0.3 cut with "56.6% of interactions are
>= 0.7, 14.6% are <= 0.3, and all 1,411 users have at least one item below 0.3."
Those are **`small_matrix` statistics**. Training runs on `big_matrix`:

| | claimed (small) | actual (big) |
|---|---|---|
| positives, `>= 0.7` | 56.6% | **52.0%** (5,212,987) |
| negatives, `<= 0.3` | 14.6% | **23.7%** (2,374,581) |
| discarded, 0.3–0.7 | — | **24.3%** (2,435,722) |

The thresholds still function — every one of the 7,176 users has both a positive
and a negative, so nobody falls out of training — but the stated justification
did not describe the data it was applied to. Corrected in config.

Worth noting separately: **24.3% of interactions are discarded**, 2.4M rows. That
is the deliberate price of not forcing ambiguous engagement into a class, and it
is a real cost that should be stated rather than glossed.

---

## The problem: watch_ratio is a duration measurement

`corr(video_duration, watch_ratio) = -0.4048`

| video length | % labeled positive | % labeled negative |
|---|---|---|
| < 10 s | **69.1%** | 13.1% |
| 10–20 s | 39.6% | 23.4% |
| 20–30 s | 9.3% | 54.0% |
| 30–60 s | 4.4% | 83.8% |
| 60–120 s | 2.8% | 93.8% |
| > 120 s | **1.2%** | **97.4%** |

A video longer than two minutes is labeled "this user disliked it" **97.4% of
the time, whoever watched it**. A video under ten seconds is labeled "liked"
69.1% of the time. Median duration is 8.3 s among positives and 12.6 s among
negatives.

The blunt version of this: **a rule that ignores the user entirely and predicts
"short video = positive" reproduces the training label 68.5% of the time**,
against a 52.0% base rate. A large share of what the model is being taught is
video length.

This is not a subtle effect and it is not specific to this project — it is the
known duration-bias problem in watch-time prediction, and Kuaishou (whose data
this is) has published on it directly.

### Does it reach the output?

Retrieved top-200 skews shorter than the catalogue: median 8.2 s vs 9.3 s, with
62.2% under ten seconds against 55.9% in the catalogue. The skew is modest, but
the embeddings currently on disk come from a two-epoch smoke run, so this is
weak evidence rather than a clean measurement. It should be re-checked against a
fully trained model.

The stronger connection is to the collapse investigation: the "global item
quality direction" that every user embedding slid into is, on this evidence,
substantially a *duration* direction.

---

## The fix: deconfound within duration

Replace the raw ratio with its **percentile rank inside a duration bucket** —
"did this user watch this more than is typical for a video of this length?"
Twenty quantile buckets, rank within bucket:

| | corr with duration |
|---|---|
| raw `watch_ratio` | **−0.4048** |
| debiased quantile | **−0.0526** |

The confound is largely removed, and the positive rate flattens across lengths:

| duration | % positive, raw | % positive, debiased |
|---|---|---|
| < 10 s | 69.1 | 31.1 |
| 10–30 s | 36.4 | 30.6 |
| 30–60 s | 4.4 | 33.4 |
| > 60 s | 2.1 | 14.1 |

**47% of the training positives would change.** That is not a tweak; it is close
to half the supervision signal, and it means every metric in this repository
would need re-measuring after the switch.

---

## Verdict

The current definition is a reasonable *first* label and a poor *final* one.

What it gets right: engagement read off observed behaviour rather than inferred
from absence, negatives drawn from the same user, and an ambiguous middle band
excluded rather than forced.

What it gets wrong: `watch_ratio` is not a clean preference signal on a
short-video corpus with a wide duration distribution. Completion is easy on a
five-second clip and near-impossible on a three-minute one, so the label encodes
a property of the *item* where it should encode the *interaction*.

The debiased quantile is the correction, and it is cheap to compute. It is not
enabled by default because switching it invalidates every number currently
reported here — that is a deliberate sequencing decision, not an endorsement of
the raw label.
