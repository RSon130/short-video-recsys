# Engineering Log

What was changed, why, and what it measured. Chronological. Every number here was
produced by a run in this repo, not estimated.

The project began as a complete, well-documented codebase that had **never been
executed**: no data downloaded, no checkpoints, no metrics — the README's results
table was seven dashes. Nine defects surfaced, none of which was visible by
reading the code. That is the theme worth carrying out of this project: the code
looked finished. Sections 10–14 came later, once the system ran end to end — the
most consequential (§11) was a model that trained, evaluated, beat its baseline,
deployed, and gave every user the same feed.

---

## 1. Dataset download was fabricated

`scripts/download_kuairec.py` fetched per-file URLs from an Aliyun OSS bucket.
Every path returns **404**; they never existed. KuaiRec ships as a single ~432 MB
zip from a Zenodo record linked off kuairec.com.

Rewritten to fetch the archive, extract by basename (the zip nests files under a
versioned directory), stage through a `.part` file so an interrupted transfer is
never mistaken for a complete one, and skip work on re-run.

## 2. Config merge was shallow

`load_config` used `dict.update()`. Because `config/kuairec.yaml` defines a
`data:` block, that **replaced base's entire `data:` section**, silently deleting
`train_ratio`, `val_ratio`, `raw_dir`, and `processed_dir`. The first real run
died on `KeyError: 'train_ratio'`.

Replaced with a deep merge in `config_loader.py`, used by every entry point —
`serving/api.py` had been carrying a second, divergent copy of the same logic.

## 3. Paths assembled from literal parts

Three files built paths as `project_root / "data" / "processed"` rather than from
config, so a directory rename silently missed them and output kept landing in the
`data/` *package*. Now read from `cfg[data][processed_dir]`, which had been
defined all along and consumed by nothing.

## 4. `lr: 1e-3` loaded as a string

PyYAML follows YAML 1.1, where an exponent needs a decimal point and a signed
exponent (`1.0e-3`) to parse as a float. Bare `1e-3` is a **string**, and it
fails deep inside `torch.optim.Adam` with `'<=' not supported between float and
str` — an error pointing nowhere near the config. Plain decimals now, with a test
asserting every hyperparameter reaching PyTorch is numeric.

## 5. Train/serve skew in the ranker

`train_ranking.py` trained on real dense side features. `evaluate.py` and
`serving/api.py` both passed **zero vectors** — 113 of the 241 input dimensions,
**47% of the input**, were zeros at inference but real during training.

The failure mode is deceptive: the model does not crash, it just underperforms,
which reads as "the ranker adds nothing" rather than as a bug. It also silently
invalidates the retrieval-vs-ranker comparison the whole project rests on.

Fixed by making `features/dense_features.py::DenseFeatureStore` the single
definition of the input layout, called by all three. When the assembly was later
vectorised for speed, a test was added asserting `build_matrix` and `build_input`
produce byte-identical rows — the vectorised path is exactly where this class of
bug creeps back.

## 6. Negative sampling was invalid for this dataset

Retrieval would not learn: BPR loss flatlined at **0.597** against a random-init
baseline of **log(2) ≈ 0.693**.

Two causes compounded. Every interaction was treated as a positive — the
`watch_ratio > 0.5` rule in the design doc was never implemented. And negatives
were drawn uniformly from the item vocabulary, the standard recipe for a *sparse*
implicit-feedback matrix, where an unobserved pair is a fair guess at
disinterest.

Measured density: **99.62%**. KuaiRec `small_matrix` is fully observed, so "an
item the user has not interacted with" is a nearly empty set, and a uniformly
drawn negative is an observed interaction with the *same* `watch_ratio`
distribution as the positives (mean 0.702 either way). **Positives and negatives
were statistically identical** — there was no ranking signal to learn.

Because the matrix is fully observed, engagement is *known* rather than inferred.
See "How a pair is labelled" below. Loss dropped **0.597 → 0.264**, and epoch
time **780s → 119s** after removing a per-sample `df.iloc` lookup that dominated
the epoch.

## 7. Dense features were never normalised

The ranker collapsed: validation MSE was identical to six decimal places across
every epoch. The arithmetic identifies it exactly — MSE `= var + (1-mean)^2 =
0.1005 + 0.0995 = 0.1999`, matching the observed `0.199913`. The model emitted
**1.0 for every input**.

Raw item features reach **2.6e11** and user features ~2e3, while tower embeddings
are L2-normalised to ~0.1 — twelve orders of magnitude into the same Linear
layer, saturating the sigmoid. The original design doc (§4.1) had specified "normalised
numerics"; nothing implemented it.

Signed `log1p` (count columns span many orders of magnitude) then a z-score,
applied at write time so training, evaluation, and serving read identical values.
Verified after regeneration: user dense mean 0.000 std 0.983, item dense
mean -0.000 std 1.000.

## 8. Epoch count was a guess

The first full retrieval run showed training loss bottoming at **epoch 4** then
drifting upward for 16 more — and the exported embeddings came from epoch 20, a
measurably worse model. `val.parquet` was being generated and never read.

Both models now track validation loss, restore the best checkpoint, and stop
early. On re-run, early stopping selected epoch 4 independently.

## 9. The evaluation was rigged — by this repo

The first three-way comparison showed the pipeline losing to popularity by 96%.
That was a bug in `evaluate.py`, not a property of the model: the already-seen
filter was applied to **popularity only**.

Users have seen 2,651 of 3,327 items in training, and the temporal split makes
train and test pairs disjoint — so popularity drew from the small pool containing
all test items while retrieval drew mostly from items that *cannot* appear in the
test set. Structurally unwinnable. All systems now exclude seen items.

Related, same root cause: ground truth counted **any** test-split row as
relevant. On a 99.6%-dense matrix that measures *exposure*, not preference.
Relevance now uses the same `watch_ratio` threshold as training.

## 10. Checkpoints were selected on a loss that did not track the metric

Two pairwise ranker runs with identical config reached validation losses of
0.2718 and 0.2708 — effectively the same — and recall@10 of **0.0141 and
0.0102**, a 38% gap. Choosing "the best epoch" by loss was close to arbitrary
with respect to ranking quality, and run-to-run noise was larger than the
differences being compared between objectives.

The ranker now scores a fixed 300-user validation probe every epoch and keeps
the checkpoint with the best recall@10. Training is seeded, and the selected
epoch, objective, and training time go into `ranker_meta.json`; `evaluate.py`
refuses to run if that file disagrees with the config.

Retrieval later needed the same fix (see §14): its in-batch softmax validation
loss rose after epoch 1 while held-out recall told a different story.

## 11. Every user received the same recommendations

Nothing crashed and no metric flagged it. It surfaced while inspecting what a
row of `user_embeddings.npy` looks like: **all 7,176 user embeddings were one
identical vector** (per-dimension std ~3e-06, one unique row). Retrieval returned
the same list to everybody, and still beat popularity by +136%, because a learned
global item ordering plus per-user exclusion of seen items is enough to do that.

Each step of the diagnosis was measured before moving to the next:

| hypothesis | evidence | fix tried | result |
|---|---|---|---|
| weight decay on LayerNorm gains | user tower weights all 0.000000 except the final bias | 1-D params excluded from decay | still collapsed |
| dying ReLU | 0% positive pre-activations | GELU | still collapsed |
| no signal to learn | ~59% of watch_ratio variance is user×item interaction; users' top-50 lists overlap at Jaccard 0.045 vs 0.003 by chance | — | signal exists: an optimisation failure, not a data limit |
| **BPR has no cross-user term** | item quality alone explains 29.5% of variance and acts as an attractor every user slides into | in-batch softmax (InfoNCE) | 7,176 unique embeddings, but loss at chance: 6.9308 vs log(1024) = 6.9315 |
| **false negatives in the batch** | 13.5%+ of in-batch "negatives" are the user's own positives — the sparsity assumption fails on dense data | mask known positives out of the softmax | **learns** |

The last row is the same mistake as §6 in a new place: a standard recipe that
assumes a sparse interaction matrix, applied to one that is not.

A guard now exists so this cannot ship silently again: training refuses to
export an embedding table whose rows have collapsed, and a test pins that check.

## 12. The label measures video length

`corr(video_duration, watch_ratio) = −0.40`. Completing a 5-second clip is easy
and completing a 3-minute one is rare, so videos over 120 s are labelled negative
**97.4%** of the time whoever watched them, and a rule that ignores the user and
predicts "short = positive" reproduces the label 68.5% of the time against a 52%
base rate.

Ranking `watch_ratio` within 20 duration buckets drops the correlation to −0.05,
and **changes 47% of the positives**. It is designed and measured but not
enabled: switching it invalidates every reported number, and it was deliberately
kept separate from the §11 retrain so the effect of each change can be
attributed. Full analysis: [label_design.md](label_design.md).

## 13. Exact search beats ANN at this scale — measured

FAISS flat, IVF, and HNSW are all implemented and benchmarked on the trained
embeddings. At 9,958 items exact search takes 0.087 ms p50 with recall 1.0; IVF
is 0.036 ms at recall 0.905 and HNSW 0.045 ms at recall 0.768. Against a 9.6 ms
end-to-end p95, ANN would save 0.5% of a request for 10–23% of the true
neighbours. Exact search scales linearly (9.19 ms at 1M), so the crossover is
around 300–500K items. Full sweep: [index_benchmark.md](index_benchmark.md).

## 14. After the collapse fix: retrieval personalises, and the ranker now hurts

Full retrain with masked InfoNCE, checkpoints selected on held-out recall@10
over 1,000 users. Validation loss would have picked badly: it rose every epoch
after the first while training loss kept falling.

| epoch | train loss | val loss | val recall@10 | val recall@200 |
|---|---|---|---|---|
| **1** | 6.2840 | 6.6436 | **0.02673** | 0.04680 |
| 2 | 6.1008 | 6.6628 | 0.02333 | 0.04370 |
| 3 | 6.0171 | 6.6870 | 0.02503 | 0.05543 |
| 4 | 5.9631 | 6.7294 | 0.02310 | 0.05239 |

7,176 distinct user embeddings (per-dim std 0.070). The ranker was retrained on
the new embeddings. Test set, 6,873 users, paired bootstrap:

| system | recall@10 | ndcg@10 | vs popularity, 95% CI |
|---|---|---|---|
| popularity | 0.0055 | 0.0052 | — |
| retrieval only | 0.0129 | 0.0134 | **+132.6%** [+0.0063, +0.0084] |
| full pipeline | 0.0095 | 0.0093 | +72.1% [+0.0030, +0.0050] |

**Full pipeline vs retrieval only: −26.0%**, CI [−0.0045, −0.0022]. Under the
collapsed retrieval the ranker added +14.3%. It now makes results worse.

Three honest readings:

- The fix restored personalisation but did **not** raise offline recall: the
  collapsed model scored +136.2% over popularity, the fixed one +132.6%.
- Retrieval's watch-time AUC is **0.34** — below chance — so its scores run
  against watch_ratio among its own candidates. Untested hypothesis: in-batch
  negatives are drawn in proportion to item frequency with no logQ correction,
  which pushes popular items down, and popular items here skew short and
  high-watch-ratio (§12).
- Both stages select epoch 1. Something overfits early, which argues for
  regularisation or a higher temperature before any architecture change.

The ranker regression is not yet explained, so the deployed service has **not**
been updated to this model.

> **Superseded by §15–§17.** Two hypotheses above turned out wrong. The 0.34
> watch-time AUC came from pooling all users together (per user it is 0.54),
> and logQ is not the lead explanation. The numbers in this section were also
> measured before duplicate rows were removed (§16).

## 15. Diagnosing §14: exposure, staleness, and a ranker that sorts by duration

Read-only diagnosis (`scripts/diagnose_ranker.py`, `diagnose_retrieval.py`,
`diagnose_staleness.py`), checked by a fresh-context AI-agent reviewer, which corrected two
of my conclusions.

**What the ranker does**
- Its scores are essentially "shortest video first": per-user Spearman −0.92
  with duration, and +0.96 with the same score averaged across users. It
  barely agrees with retrieval (+0.08).
- It trains on pairs of videos the user *watched*, so it learns completion
  given exposure. It is evaluated on *unwatched* candidates, where relevance
  also requires the platform to have shown the video.
- It promotes stale videos: 46.5% of its top-10 had recent training activity,
  against 67.5% for retrieval. When its picks were shown, they were hits more
  often (0.817 vs 0.620). It fails at predicting exposure, not taste.

**Recency matters for both stages.** A training-window recency filter lifted
retrieval by +30% and the ranker by +55%. The ranker still lost 20% to
retrieval afterwards.

**Refuted along the way**
- Ranker overfitting: the test used could not detect it, and ranker and
  retrieval scores are nearly uncorrelated anyway.
- Popularity as the missing signal: re-ranking candidates by popularity also
  scores 0.0084.

**Found**
- The ranker's item features were averaged over the full data period,
  including val and test.

## 16. The evaluation was measuring exposure, and the data had 968K duplicate rows

**An unbiased test set was sitting unused.** KuaiRec's `small_matrix`:
- shares all of its 1,411 users and 3,327 videos with `big_matrix`;
- shares **no** (user, video) pair with it;
- is 99.6% observed.

That makes it an exposure-unbiased test set for models trained on `big_matrix`.

**The existing models, scored on it before any fix:**
- The ranker tied "shortest first".
- Retrieval was anti-correlated with finishing (AUC 0.29).
- With duration controlled, everything sat at chance.

**Exact duplicate rows.** `big_matrix.csv` has **968,005 exact duplicate
rows** (same user, video and millisecond timestamp): 6.3% of the old train
split, 22.6% of val, 4.6% of test.
- They cluster on a few days: 67% of all rows on Jul 27.
- Items first seen on Aug 4 got thousands of copies.
- Removed in the loader. A test pins it.

**What this changes**
- A test-retest analysis that read duplicate pairs 11 seconds apart as
  re-views had shown watch_ratio "repeating" at 0.61. Genuine re-views a day
  or more apart agree at 0.27.
- The old popularity baseline on the big hold-out (recall@10 0.0055) was
  inflated by those duplicated recent items. Deduplicated it is 0.0016, so the
  "+132.6% over popularity" claim no longer stands.

A pre-registered protocol replaces the temporal hold-out:
[evaluation_protocol.md](evaluation_protocol.md). A reviewer checked it before
implementation and changed the label, the two-stage metrics and the baselines.
A second reviewer checked the implementation.

## 17. Results under the pre-registered protocol

1,012 test users. Primary label: within-user top 30% of within-duration-bucket
watch_ratio percentiles. The label check passed. Per-user AUC over ~3,300
videos per user:

| system | per-user AUC | NDCG@10 |
|---|---|---|
| **B3 item quality within duration** (non-personal) | **0.548** | 0.320 |
| B4 item-kNN | 0.519 | 0.322 |
| B6 user duration preference | 0.517 | 0.338 |
| ranker alone | 0.516 | 0.424 |
| B2 shortest first | 0.515 | **0.440** |
| retrieval | 0.512 | 0.240 |
| B5 platform completion within duration | 0.509 | 0.321 |
| B0 random | 0.500 | 0.297 |
| B1 popularity | 0.492 | 0.350 |

**Pre-registered verdict (§7)**
- Retrieval does **not** add value: it beats only popularity.
- The ranker alone does **not** add value: it beats only popularity.
- "Ranker adds value in two-stage": **passes**. NDCG@10 +0.155 and
  within-candidate AUC +0.090 over retrieval, Holm-corrected, item bootstrap
  above zero.

**Why that pass does not mean what it says.** A fresh-context AI-agent review of the
results showed this, and I verified the key numbers.

*1. The comparison point is broken.* Retrieval orders its own top 200 worse
than random: NDCG@10 0.240 vs 0.316 for a shuffle. It prefers long, unpopular
videos: item score vs duration +0.49, vs positive-count popularity −0.74. That
is consistent with in-batch softmax having no logQ correction, but it is not
tested.

*2. The ranker is a duration sort.* Spearman 0.96 with shortest-first inside
the candidates. Replacing the ranker with a plain shortest-first reorder of the
same 200 candidates scores **higher** (NDCG@10 0.407 vs 0.395).

*3. The label is not debiased at the extremes.* The per-bucket check passed,
but the widest bucket (57.6–315 s) still has a strong duration gradient
inside it. Positive rate by duration decile within that bucket:
0.73 → 0.72 → 0.67 → 0.54 → 0.32 → 0.12 → 0.06 → 0.05 → 0.04 → 0.03.
A bucket-level check averaged it away.

**Big-matrix hold-out, re-run for continuity.**
- Retrieval 0.0162 recall@10, full pipeline 0.0154 (−4.9%, not significant),
  popularity 0.0016.
- A trivial baseline, popularity over only the last 3 days of training, scores
  **0.0502**, about 3× the pipeline. That check was post-hoc, a single run
  with no bootstrap.
- The −26% → −4.9% change mixes deduplication, new checkpoint selection and a
  retrained ranker, so it cannot be attributed to any one of them.

**What this project can honestly claim**
- Neither learned stage beats simple, fair baselines on KuaiRec under an
  exposure-unbiased evaluation.
- The strongest signal available is non-personal (item quality within
  duration) or temporal (recency).
- The protocol, a label check, and two fresh-context AI-agent reviews are what kept a
  letter-of-the-protocol "pass" from being reported as a win.

**Protocol limitations that need a revision (v3) before the next result**
- Condition on duration continuously, or with log-spaced buckets, and check
  the gradient *within* buckets.
- Add random and shortest-first reranker controls to the two-stage comparison.
- Add a recency baseline.
- Record the 20% and 50% label checks: both fall outside the range, though
  only 30% was pre-registered.

---

## 18. Phase 2, Phase 0: KuaiRand audit and signal gate — NO-GO

Plan: `docs/phase2_kuairand_plan.md` (see its "Phase 0 amendments"). Code:
`scripts/audit_kuairand.py`, `scripts/gate_kuairand.py`, `data/kuairand.py`,
`evaluation/gauc.py`.

**Question.** Does engagement carry learnable personal preference once exposure
and duration are controlled?
- Train on KuaiRand-Pure's recommender-exposed standard log (4/09–4/21).
- Evaluate on its random-exposure log (4/22–5/08).
- Metric: per-user AUC (GAUC), validation users only (8,147 of 27,285; test
  users untouched).

**What the audit changed before the gate**
- The README's `is_click` = valid_play rule is not exact in any tab.
  Two-column tabs were re-identified by zero play time on unclicked
  impressions. The random log is 99.3% single-column tab 1, so **two-column
  click is not evaluable**.
- **Explicit feedback is sparse on random exposure.** Like rate is 0.48% vs
  1.87% on recommended items.

  | label | validation users with both classes |
  |---|---|
  | like | 705 |
  | composite `explicit_positive` | 875 |
  | follow, comment, forward | 77–108 |
  | hate | 139 |

  The pre-registered half-width rule left the composite as the only gate label.
- **Data problems**
  - Duplicate rows: 15,609 exact, plus ~1K sharing (user, video, time).
  - The log is front-loaded (4/10–4/12 hold 58% of rows).
  - 239 items have unknown duration: `video_duration` is NaN and `duration_ms`
    is 0 on every log row.
  - `upload_dt` has three values.

**Gate design.** Pre-registered in the script docstring and committed
(`50aae84`) before the validation run. A fresh-context AI-agent review of the
design replaced three things before any validation run:
- noisy early stopping → 300 fixed rounds;
- 94K fit rows → 344K via expanding daily cutoffs;
- raw-count features → shares.

It also fixed a crash in the duration-band baseline.

**Result**

Run 3 is final. Runs 1–2 had duration bugs, disclosed below.

| label | users | personal LightGBM | best non-personal | diff (95% CI) |
|---|---|---|---|---|
| **explicit_positive (gate)** | 875 | 0.541 | shortest-first 0.553 | −0.012 (−0.033, +0.010) |
| | | | item impressions 0.541 | 0.000 |
| like | 705 | 0.557 | shortest-first 0.555 | +0.002 (−0.020, +0.024) |
| valid play (`is_click`) | 6,798 | 0.577 | item impressions 0.568 | +0.0095 (+0.0045, +0.0145) |
| long view | 5,485 | 0.602 | item impressions 0.596 | +0.006 (−0.001, +0.012) |

- **Gate verdict: NO-GO** in all three runs and in the tab-1 sensitivity
  check. It does not depend on duration: the personal model also only ties
  item impression count.
- **The rare labels are too thin to conclude.** On follow and comment (77–80
  users) the personal model is *below* shortest-first (about −0.10, CIs
  excluding 0).
- **Valid play and long view are watch-time thresholds, not gated.** The
  personal gains there are real, but below the pre-set 0.01 bar.
- **Power.** 70% of gate users have a single positive, and the personal and
  impressions per-user AUCs correlate −0.26 (run 1), so the CI is ±0.02–0.03. The
  claim is "no gain above about 0.03 detected", not "no personal signal".

**Post-hoc finding: signs flip between logs.** This was a fresh-context
AI-agent diagnostic on validation users, and it cannot change the verdict.

| single feature | label | standard log | random log |
|---|---|---|---|
| item impression share | explicit feedback | 0.44 | 0.54 |
| author impression share | explicit feedback | 0.45 | 0.54 |
| user's tag share | explicit feedback | 0.47 | 0.51 |
| item's own like rate | like | 0.65 | 0.55 |
| duration | explicit feedback | 0.45 | 0.44 |

- Popularity and familiarity reverse. Among items the recommender already
  chose, heavily pushed items and familiar tags get *fewer* likes; across the
  catalogue they get more.
- An item's own like rate weakens, and duration is stable.
- The run-1 trained model's scores correlate −0.15 with impressions on
  random items. This is consistent with models trained on exposed logs learning
  exposure artefacts. On watch-time labels nothing flips; the features just
  get stronger.
- The non-personal LightGBM scores below a raw impression count on every
  label (e.g. valid play 0.548 vs 0.568).

**Bugs caught by the process**

*Run 1.* NaN durations for 239 items corrupted the shortest-first per-user
AUCs.
- Pandas `rank` leaves NaN unranked, and the audit's
  `abs(a - b) > 1000` check is false for NaN, so it reported 0 mismatches.
- `per_user_auc` now rejects NaN.

*Run 2.* Filled those durations from the logs, which were all 0. That made
them the "shortest" videos. A second AI-agent review caught it.

*Run 3.* Treats the duration as unknown:
- NaN for LightGBM;
- its own duration band;
- the median duration in the duration baselines.

All three runs' JSON outputs are kept in `datastore/processed/`. The
per-item duration fill in run 2 used all logs, including test users' rows.
That is a fixed item property, so nothing leaked.

**Disclosure.** The audit's duration table scored all random-log users,
including test users, with fixed shortest-first. No model or tuned scorer
touched test users.

**What is claimable.** Two datasets and two label types now agree under
exposure-unbiased, pre-registered tests:
- personalised models trained on logged data do not beat simple non-personal
  scorers (duration, exposure count) at a detectable margin;
- features learned on exposed logs can point the wrong way on random
  exposure.

---

## 19. Phase 2b: exposure bias, run once on held-out users

Pre-registered in `docs/phase2_exposure_bias_plan.md`, committed (`1ea2e90`)
before the run. Code: `scripts/exposure_bias_kuairand.py`. Two fresh-context
AI-agent reviews ran before the experiment (design, then code) and one after
(results). Test users were scored exactly once.

**Setup.** Train on KuaiRand's recommender-exposed standard log (4/09–4/21) or
on randomly exposed rows; evaluate per-user AUC on **test users'** random-
exposure rows (4/22–5/08, tabs 1–2): 16,046 users for valid play, 12,858 for
long view. Labels are KuaiRand's thresholds: `is_click` is **valid play**, not
a click.

| arm | training data |
|---|---|
| A0 | exposed log, 343,892 rows (the Phase 0 model) |
| A1 | A0 reweighted by inverse item popularity, `p^(-1/2)` |
| A2 | A0 without the six exposure-volume features |
| A3 | validation users' random rows, 352,477 |
| A0′ | validation users' **exposed** rows, same users and window, 77,184 |
| A3s | A3 subsampled to 77,184, matching A0′ |
| N1 | non-personal: item impression count, exposed log |
| N3 | non-personal: item positive rate from validation users' random rows |

### Result 1: the sign flip replicates on untouched users

Single feature, no model, standard fit rows vs test users' random rows:

| feature | label | exposed log | random exposure | difference |
|---|---|---|---|---|
| item impression share | explicit feedback | 0.441 | 0.548 | +0.107 (CI +0.092, +0.122) |
| author impression share | explicit feedback | 0.442 | 0.543 | +0.101 |
| item impression share | like | 0.428 | 0.546 | +0.117 |
| user's tag share | explicit feedback | 0.471 | 0.507 | +0.036 |

- Among the items the recommender chose to show, exposure volume ranks explicit
  engagement **below chance**. Across randomly shown items it ranks **above**
  chance. Targeted exposure is a selection effect, not evidence that popular
  videos repel users.
- On the watch-time labels nothing flips; the same features simply get
  stronger under random exposure.
- `utag_share` on likes moved in the same direction but stayed below 0.5, so by
  the pre-registered rule that one is **not** a flip.
- The two sides differ in history window and row population as well as in
  exposure, so the size of the shift is not a clean causal estimate. Its
  direction agrees with the validation-set finding in §18.

### Result 2: none of the three standard corrections helped

| comparison | valid play | long view | verdict |
|---|---|---|---|
| H1 inverse-popularity weighting vs A0 | −0.0013 | +0.0008 | no material difference |
| H2 drop exposure features vs A0 | −0.0057 | −0.0090 | significantly **worse** |
| H5 A0 vs impression count | +0.0009 (p 0.59) | +0.0021 (p 0.32) | **did not replicate** |

- **A1 behaved as designed.** The weights are deliberately under-corrected
  (effective sample size 59.7%), and item popularity is a crude stand-in for
  the real logging policy.
- **A2 is mildly informative:** the exposure-volume features are net useful
  under random exposure *despite* flipping sign, so the flip is not fixed by
  deleting them.
- **§18's +0.0095 valid-play gain over impression count is superseded and
  should not be cited again.** On held-out users it is +0.0009, about 2.8 SE
  away, too far for noise alone. The model barely moved (0.577 → 0.576); the
  **comparator** rose (0.568 → 0.575). The mechanism is selective emphasis: of
  four labels, valid play was the one whose validation CI excluded zero, and it
  became the narrative.

### Result 3: what the training distribution is beat how much of it there is

The only pre-registered claims. Users, window, feature snapshot and row count
are matched; only the exposure mechanism differs:

| | valid play | long view |
|---|---|---|
| A3s, trained on 77K random rows | 0.6400 | 0.6658 |
| A0′, trained on 77K exposed rows | 0.5850 | 0.6095 |
| difference | **+0.0550** (CI +0.052, +0.058) | **+0.0563** (CI +0.053, +0.060) |

It survives the duration guard (+0.052, +0.042), holds across three seeds, and
A3s wins with **2.8× fewer positives** (13,578 vs 38,023).

**What it does not mean.** The post-hoc results review bounded this hard:
- **It is not personalisation.** A3s does not beat N3, the non-personal item
  rate from the same rows: −0.0056 and −0.0019. Splitting A3s into a per-item
  mean and a within-item residual gives 0.629 of its 0.640 from the item mean
  alone; the within-item part, 0.542, is barely above A0′'s 0.532. Demeaning
  both arms within N3 deciles shrinks the gap from +0.055 to +0.026.
- **It reverses on exposed data.** On the 4/21 exposed diagnostic, A3s scores
  0.548 against A0′'s 0.602. Each arm wins on the distribution it was trained
  on, so this measures **train/evaluation distribution match**, not model
  quality.
- **It is not deployable.** A3, A3s, A0′ and N3 all use data from the
  evaluation window.
- On the explicit-feedback labels the same contrast is *negative* and not
  significant (−0.013), so it does not generalise across label types.

### Other pre-registered outcomes

- **H4, personal model vs the unbiased item rate:** +0.0057 and +0.0131,
  significant but **fails the duration guard** (+0.0014, +0.0007), so no claim.
  The margin over an item rate is essentially duration. The guard is
  conservative — it also removes genuine duration-correlated preference — and a
  null here does not refute personalisation in general, only at this power.
- **Secondary labels** (2,016 and 1,649 users) resolve nothing below about
  0.03.
- **N3 beats N1 on watch labels** (0.646 vs 0.575) but *loses* on explicit
  feedback (0.516 vs 0.549). That is estimator noise, not a finding: with 1,927
  explicit positives across 352K rows, smoothing collapses 7,530 items into 185
  distinct values, so N3 is nearly constant there.

### Caveats carried

- `is_click` is valid play, a watch-time threshold.
- The random log is uniform over a **platform-selected candidate pool**, not the
  whole corpus.
- A0/A1/A2 use expanding-history features while A0′/A3/A3s use the
  evaluation-window snapshot, so A0′ vs A0 is not a like-for-like comparison of
  data volume.
- A2's drop list was chosen after the validation diagnostic.
- No policy-value or online-lift claims: everything here is offline ranking on
  logged data.

### What this adds

- A replicated, model-free demonstration of exposure bias on held-out users.
- Evidence that two common corrections do not fix it, and that matching the
  training distribution does far more than either.
- A result that did not replicate, caught by holding out users and running once.

---

## How a pair is labelled

*"How do you decide one item is better than another?"* — the model never
decides. It is read off observed
behaviour.

For a given user:

| | condition | meaning |
|---|---|---|
| positive | `watch_ratio >= 0.7` | watched it through |
| negative | `watch_ratio <= 0.3` | bailed out early |
| excluded | `0.3 < watch_ratio < 0.7` | ambiguous — not used for training |

Three constraints, all enforced in code:

**Same user, always.** Negatives come from that user's own low-engagement items
(`InteractionDataset.neg_pools`, keyed by user id). Comparing across users would
smuggle popularity back in — a globally popular item would look "better" than a
niche one the user actually preferred. Only users with no low-engagement item of
their own fall back to a global pool, and on this data every user has one.

**Only confident pairs.** The middle band is dropped rather than forced into a
class it does not belong to. Those items are still scored at inference; they are
simply not used to teach the ordering.

**Only the margin matters.** The loss is `-log σ(score_pos - score_neg)`, so
absolute score values are free — the model is graded purely on getting the order
right. This is why the sigmoid was moved out of the network: squashing scores
into [0,1] compresses the positive/negative margin and flattens the gradient
exactly where a ranking loss needs signal. Sigmoid is monotonic, so ordering and
every ranking metric are unchanged; `predict()` reapplies it for the serving
contract.

The same definition is used by retrieval (`InteractionDataset`), by the ranker
(`PairwiseRankingDataset`), and by evaluation's ground truth — one threshold pair
in config, three consumers. Training and evaluation therefore agree on what
"good" means, which is not automatic and was not true earlier in this project.

---

## The two findings worth discussing

### Which stage carries the system depends on scale

| | `small_matrix` | `big_matrix` |
|---|---|---|
| interactions | 4.68M | 12.53M |
| density | 99.6% | 17.5% |
| eligible items/user | ~676 | ~8,771 |
| relevance base rate | 25% | 1.1% |
| popularity recall@10 | 0.4806 | 0.0055 |
| what wins | popularity, unbeatable | the two-stage pipeline |

On the dense subset a popularity list is a genuinely strong baseline, not a
strawman, and retrieval's *narrowing* has little value across only 3,327 items.
On the sparse subset popularity collapses and personalisation wins outright.

### The ranker's objective mattered more than its architecture

Same network, same features, same candidates — only the loss changed:

| objective | recall@5 | recall@10 | recall@20 | watch-time AUC |
|---|---|---|---|---|
| pointwise — MSE on `watch_ratio` | 0.0031 | 0.0070 | 0.0093 | **0.714** |
| **pairwise — BPR** | **0.0112** | **0.0141** | 0.0099 | 0.622 |
| listwise — sampled softmax (N=4) | 0.0081 | 0.0099 | **0.0100** | 0.609 |

MSE optimises *calibration* and duly wins the metric that rewards calibration.
Recall@K rewards *ordering*, and a regression head minimising squared error is
pulled toward the conditional mean. Under MSE the full pipeline scored below
retrieval alone at every cutoff; under a ranking loss it overtakes retrieval at
K=10 and beats popularity everywhere (+115% recall@5, +155% recall@10).

Retrieval had trained with a ranking loss from the start. The ranker was the only
stage optimising something other than the metric it was judged on.

The trade is visible: watch-time AUC falls 0.714 → 0.622. A multi-task head
would recover both.

### Listwise did not beat pairwise — a prediction that failed

Sampling several negatives per step and taking a softmax normally beats a single
pair; it is why large-scale rankers use sampled softmax. Here it lost: recall@5
0.0081 against pairwise's 0.0112.

**Hypothesis tested and refuted.** The obvious suspect was degenerate sampling —
drawing 4 negatives with replacement from a small per-user pool would yield
duplicates and a softmax with redundant candidates. Measured on `big_matrix`:
median pool is **233 items**, mean 282, and only **1% of users** have fewer than
10. Duplicates are not the problem.

**Hypothesis still untested.** Saturation. With four *easy* random negatives the
softmax is satisfied the moment the positive outranks all of them, so gradients
shrink faster than single-pair BPR's, which keeps drawing a fresh comparison each
step. If that is the mechanism, the fix is harder negatives rather than more of
them — sample from retrieval's shortlist, which is the distribution the ranker
actually faces at inference, instead of from the whole low-watch pool.

Worth stating plainly: this is a negative result recorded as such. `pairwise`
remains the default because it measured best, not because the theory preferred
it. Both objectives stay selectable so the comparison is reproducible.

---

## Open work

> The comparisons in "The two findings worth discussing" predate §11: their
> ranker results were measured on candidates from the collapsed retrieval model.

### Next experiment: logQ correction in retrieval (planned, not run)

This applies to the **retrieval** stage's in-batch softmax only. The ranker's
pairwise loss draws each negative from the *same user's* low-watch items rather
than from other users' positives, so it has no in-batch sampling bias to correct.

**The bias.** With in-batch softmax, a user's negatives are the other users'
positives in the batch. An item therefore appears as a negative in proportion to
its frequency among training positives, `Q(i)`. Popular items are pushed down
far more often than niche ones, and the learned score drifts toward
`true affinity − log Q(i)`: the model systematically under-scores popular items.

**Why it is the lead hypothesis for §14.** Retrieval's watch-time AUC is 0.34,
below chance, so its scores run against watch_ratio. Popular items on this
dataset skew short and high-watch-ratio (§12), so an anti-popularity bias would
produce exactly that sign. The ranker consumes retrieval embeddings, so the same
bias could plausibly contribute to the −26% regression. Neither link is measured.

**The change.** Subtract `log Q(i)` from every logit during training only,
`logit(u, i) = u·i / τ − log Q(i)`, with `Q` counted once from training
positives (Yi et al., RecSys 2019). Serving keeps the plain dot product, so the
index and API do not change. About ten lines in `sampled_softmax_loss`.

**What would falsify it.** Retrieval watch-time AUC staying at or below 0.5 after
the change. **What it could cost.** Popularity is a strong signal here (it is
the baseline), so removing the anti-popularity penalty can raise or lower
recall@10; a scaled correction `α·log Q`, α in (0, 1], is the fallback if the
full correction overshoots.

**Protocol.** One change against the §14 run. Same seed, same recall-based
checkpoint selection, full retrieval → index → ranker → evaluate, and both
outcomes recorded here.

### Other open items

- **Explain the ranker regression (§14)** also needs a direct check, whatever
  logQ shows: compare the ranker's input feature distributions under the
  collapsed and the fixed embeddings.
- **Duration-debiased label (§12)**, as its own measured step, after the above.

- **Hard-negative mining** — the highest-value open item. Both unexplained
  results point at it: listwise gained nothing from more *easy* negatives, and
  retrieval still leads at K=5. Sampling negatives from retrieval's top-200
  shortlist trains the ranker on the distribution it actually sees at inference.
- **LambdaRank-style weighting**, scaling each pair by its effect on NDCG, is the
  metric-aware step beyond listwise. Worth trying only after hard negatives —
  the sampling distribution looks like the binding constraint, not the loss form.
- **Multi-task head** — ranking loss for ordering plus regression for calibrated
  watch-time, recovering the AUC the pairwise objective trades away.
- **Stronger candidate generation.** The ranker's ceiling is retrieval's
  shortlist, and retrieval still leads at K=5. Hard-negative mining and
  multi-source recall target that directly.
