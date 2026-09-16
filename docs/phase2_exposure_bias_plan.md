# Phase 2b: exposure bias on KuaiRand (pre-registration)

**Status:** RUN AND COMPLETE, 2026-09-15. Results are in engineering log §19;
raw output in `datastore/processed/kuairand_exposure_bias.json`. The text below
is the pre-registration as committed in `1ea2e90`, before the single test-user
run; it is kept unchanged as the record. Once this file is committed,
the thresholds are fixed. Test users are scored exactly once, by
`scripts/exposure_bias_kuairand.py`.

## Why

The Phase 0 gate (engineering log §18) was NO-GO on explicit feedback. Two of
its results motivate this experiment.

**1. Signs flip between logs** (post hoc, validation users). Popularity and
familiarity features change direction between the log the recommender chose
and random exposure:

| single-feature per-user AUC, explicit feedback | standard log | random log |
|---|---|---|
| item impression share | 0.44 | 0.54 |
| author impression share | 0.45 | 0.54 |
| user's tag share | 0.47 | 0.51 |

Reading: among the items the recommender *chose to show*, the heavily pushed
ones draw less explicit engagement, because exposure is targeted — a selection
effect, not a statement that popularity repels users. Across the catalogue,
exposure volume is a quality signal. A model fitted on the exposed log learns
the first relation and is scored against the second.

**2. There is a small personal gain on watch-time labels.** The personal
LightGBM beat impression count by +0.0095 on valid play (CI +0.0045 to
+0.0145). These labels have ~6.8K validation users with both classes.
**On watch-time labels nothing flipped**, so the sign-flip and the model
questions are separate and are tested separately below.

**Question.** Models trained on logged data inherit the logging policy's
selection bias. Does it replicate on untouched users, and do three standard
responses fix it?
1. Reweight the log.
2. Remove exposure-volume features.
3. Train on a uniformly exposed sample instead.

All three are established techniques, used here as cited comparisons. None is
new.

**One-line summary of the exposure effect** (validation users, same window):
valid-play rate 0.4496 on exposed impressions vs 0.1750 on random ones;
long view 0.3175 vs 0.0837.

**Duration.** On random exposure, play-ratio and completion targets are
dominated by duration (longest-first per-user AUC 0.15 and 0.24). Raw play
time is flat there, but rises with duration in the exposed log (0.62) — itself
an exposure effect. The labels here are KuaiRand's thresholded `is_click`
(valid play; shortest-first 0.513, within-bucket 0.500) and `long_view`
(shortest-first 0.538). A duration guard is pre-registered below.

## Deliverable 1 (primary): does the sign flip replicate?

Model-free, and the best-powered test here.

- **Features:** `item_impr_share`, `author_impr_share`, `utag_share`.
- **Labels:** `explicit_positive` (where the flip was reported in engineering
  log §18) and `is_like` (an extension, marked as such),
  plus `is_click` and `long_view` (where it was not).
- **Row sets:** standard-log fit rows (as in A0) vs **test users'** random
  rows.
- **Statistic:** per-user AUC of the single feature on each row set, both
  restricted to tabs 1 and 2, and their difference, with an unpaired user
  bootstrap (10,000) on each side. The two user sets overlap (the standard side
  includes test users' exposed rows), which makes the interval conservative; a
  sensitivity value excluding test users is reported. Rows where the feature is
  missing are dropped per side, and the kept fraction is reported.
- **Pre-registered direction:** for explicit labels, standard < 0.5 < random.
- **Decision:** replicated if the CI of the difference excludes 0 **and** the
  direction matches. Expected effect ~0.10, SE ~0.006 on ~2,050 test users, so
  this is well powered.

## Data and arms

Pipeline unchanged from `scripts/gate_kuairand.py` run 3: feature builder,
LightGBM parameters, fixed 300 rounds, deduplication on (user, video, time_ms),
unknown durations kept unknown. `video_features_statistic`, post-impression
fields and upload date are not used.

**Row sets** (single-column tabs 1, 2, 4, 5, 6 everywhere; measured counts):

| set | rows | definition |
|---|---|---|
| A0/A1/A2 fit | 343,892 | standard log, label days 4/13–4/20, features from earlier days only |
| A3 fit | 352,477 | validation users' random rows 4/22–5/08, features from the standard log 4/09–4/21 |
| A0′ fit | 77,184 | validation users' **standard** rows 4/22–5/08, same users, same window, same feature snapshot |
| A3s fit | 77,184 | A3 subsampled to A0′'s size, seed 42 |
| evaluation | test users' random rows 4/22–5/08, **tabs 1 and 2** | all tabs reported as sensitivity |

Tabs 11 and 14 are excluded from the primary evaluation: `is_click` is
structurally 0 there. They are 0.6% of rows.

**Arms** (all use the personal feature set unless stated; each is trained with
**seeds 42, 43, 44** and scored as the mean of the three boosters, with
per-seed differences reported):

- **A0** — reference: the gate model, retrained identically.
- **A1** — popularity-propensity weighting: `w = p_i^(-1/2)` with
  `p_i = (n_i + 1)/(N + I)` from the row's own history window; clipped at the
  99th percentile **of that label day's weights** (not a global cap), normalised
  to mean 1. Measured: effective sample size 59.7%, max weight 5.05, 1.0% of
  rows clipped. Note: `w ∝ n_i^(-1/2)` leaves the reweighted item
  distribution `∝ n_i^(1/2)`, i.e. **deliberately under-corrected**, chosen for
  variance. `min_data_in_leaf` counts rows, not weight.
- **A2** — exposure-feature ablation: drop `item_impr_share`,
  `author_impr_share`, `utag_share`, `uauth_share`, `uband_share`,
  `user_single_col_share`. **Disclosure:** this list was chosen after the
  validation diagnostic.
- **A3** — uniform-exposure training (see confound note below).
- **A0′** — exposed twin of A3: same users, same window, same feature
  snapshot. Only the exposure mechanism differs.
- **A3s** — A3 subsampled to A0′'s row count, so the contrast is size-matched.

**Comparators** (fixed in advance, not best-of-N):
- **N1** — item impressions, single-column standard log. Best non-personal on
  validation for both primary labels.
- **N3** — item smoothed positive rate on validation users' random rows
  (m = 20). N3 is not deployable: it uses data from the evaluation window.

**A3's confound.** A3's fit rows and the evaluation rows share one frozen
feature snapshot, so a booster can memorise an item's 4/22–5/08 random-exposure
outcome from validation users and carry it to test users. A3 therefore carries
N3's information plus contemporaneity. A3 − A0 is *not* "uniform vs exposed
training data" and is reported as descriptive only. The clean contrast is
**A3s − A0′**, and the personalisation question is **A3 − N3**.

## Hypotheses and decision rule

**Primary labels:** `is_click` (valid play) and `long_view`.

| id | comparison | question | status |
|---|---|---|---|
| H1 | A1 − A0 | does reweighting help? | decision |
| H2 | A2 − A0 | does dropping exposure features help? | decision |
| H3s | A3s − A0′ | uniform vs exposed, matched users, window and size | decision |
| H4 | A3 − N3 | personal signal given uniform training data | decision |
| H5 | A0(seed 42) − N1 | does the gate's watch-time gain replicate? | decision |
| — | A3 − A0, N1 − N3, A1/A2/A3 − N1 | context | descriptive |

**Test.** Paired user bootstrap (10,000) of per-user AUC over test users with
both classes; two-sided p; Holm at α = 0.05 across the 10 decision
comparisons (5 × 2 labels).

**Claim rule.** "X improves on Y" requires Holm rejection **and** a point
estimate ≥ 0.005. Significant results below 0.005 are reported as "significant
but below the pre-set effect size".

**Equivalence rule.** If the CI lies entirely within ±0.005, the finding is
"no material difference". Given the power below, this is a likely and
legitimate headline.

**Power (estimated from the gate's validation numbers, scaled to test size)**

| comparison | SE | detectable at Holm's tightest threshold |
|---|---|---|
| cross-family, is_click (~16K users) | 0.0017 | 0.0048 |
| cross-family, long_view (~12.9K users) | 0.0020 | 0.0057 |
| matched arms, e.g. A2 − A0 | ~0.0008 | ~0.002 |

So the 0.005 floor is about 6 SE for matched arms, but for **H5 on long_view
power binds**: the gate's own +0.0056 would be a coin flip, and a null there
says nothing about effects near 0.005.

**Duration guard.** Every claimed improvement is recomputed with scores
demeaned within 20 duration bands. **The guard is part of the claim rule**: a
comparison is only a claim if it is Holm-rejected, at least 0.005, and keeps
its sign and at least half its magnitude after demeaning. Per-tercile
differences are reported descriptively.

**Secondary** (CIs, no decisions): `explicit_positive` and `is_like`, same
comparisons. About 2,050 and 1,660 test users with both classes, so gains
below about 0.03 are undetectable.

## Procedure

1. **Implement.** Tests: A1 weights finite, mean 1, clipped; A2's feature list;
   every feature date ≤ 4/21; no test user in any fit set; A3s ⊂ A3.
2. **Bug check on validation users, invariants only:** row counts, weight ESS
   and clip fraction, NaN rates, feature-list diff, prediction hashes, the
   4/21 standard-log diagnostic GAUC, and A0 reproducing gate run 3. **No
   arm-vs-arm random-log difference is printed**, so no fix can be
   outcome-driven.
3. **Fresh-context AI-agent review** of the code and the bug-check output.
4. Commit, then run once on test users.
5. **Fresh-context AI-agent review of the results**, then the write-up:
   engineering log §19 and the README.

**Fit-set scope, disclosed.** A0, A1 and A2 fit on the standard log's rows for
*all* users, including test users' **exposed** rows, exactly as gate run 3 did.
Test users' random-log rows and labels are never used for fitting, and the
random log is what every evaluation scores. A3 and A0' use validation users
only.

**Locked:** `PARAMS` byte-identical to gate run 3, `num_threads` pinned to 4,
LightGBM version recorded, each arm's feature list written to
`datastore/processed/` and asserted, fit-matrix hashes recorded.

## Write-up guardrails

- `is_click` is **valid play**, a watch-time threshold, never "a click".
- The random log is uniform over a **platform-selected candidate pool**, not
  the whole corpus; say so rather than "unbiased" alone.
- No policy-value or online-lift language.
- N3 is not deployable.
- The exposure-bias framing belongs to the sign-flip result; the model arms
  are a separate, less-powered question.

## Time box

About 2 days. Drop A1 first if it runs over; then fall back to seed 42 only,
stating that the study cannot resolve differences below the seed spread.

## What the review changed

- The sign-flip replication was promoted from descriptive to deliverable 1
  with a decision rule: the flip is on explicit feedback, and the drafted
  primary labels were the ones where it never appeared.
- A3's contemporaneity confound was named. A0′ and A3s were added, H3 demoted
  to descriptive, H4 made the decision comparison.
- Three seeds averaged: feature and row subsampling made seed noise comparable
  to the claim floor.
- The bug check was restricted to invariants.
- Row counts corrected (352,477 not ~341K); primary evaluation restricted to
  tabs 1 and 2; bootstrap raised to 10,000; the duration guard replaced with
  band-demeaned scores; an equivalence rule and an MDE table added; the
  claimed "2 SE" justification for 0.005 was wrong and is now stated
  correctly.

## References

Verified online 2026-09-16.

- **IPS for recommendation:** Schnabel, Swaminathan, Singh, Chandak, Joachims,
  "Recommendations as Treatments: Debiasing Learning and Evaluation", ICML 2016.
  [arXiv 1602.05352](https://arxiv.org/abs/1602.05352)
- **Popularity-based propensity:** Yang, Cui, Xuan, Wang, Belongie, Estrin,
  "Unbiased Offline Recommender Evaluation for Missing-Not-At-Random Implicit
  Feedback", RecSys 2018.
  [doi:10.1145/3240323.3240355](https://doi.org/10.1145/3240323.3240355)
- **Training on a uniformly exposed sample:** Bonner, Vasile, "Causal Embeddings
  for Recommendation", RecSys 2018.
  [arXiv 1706.07639](https://arxiv.org/abs/1706.07639),
  [code](https://github.com/criteo-research/CausE)
- **Dataset:** Gao et al., "KuaiRand: An Unbiased Sequential Recommendation
  Dataset with Randomly Exposed Videos", CIKM 2022.
  [arXiv 2208.08696](https://arxiv.org/abs/2208.08696)

These are cited as prior work whose *ideas* the arms follow. No paper's
architecture, protocol or splits were reproduced, and no result here is
compared against a published number.
