# Evaluation protocol (pre-registered)

**Status:** v2, fixed on 2026-09-14, before any model was trained or scored under it.
v1 was reviewed before implementation and amended; see "Amendment history" at the
end, which also discloses the numbers the reviewer observed while checking v1.
Changing anything below after results exist must be logged as a protocol change,
with the reason and the results under both versions.

## Why this protocol replaces the big_matrix temporal hold-out

The big_matrix hold-out only contains videos the platform chose to show, so
offline recall rewards predicting exposure. The measured consequences are in
the engineering log: which stage "wins" flips with the protocol, and the ranker
is indistinguishable from "shortest video first". KuaiRec ships a second,
near-fully-observed matrix built for this problem, and the project never used it.

Measured on the raw files: all 1,411 `small_matrix` users and all 3,327 of its
videos appear in `big_matrix`, **no (user, video) pair appears in both**,
`small_matrix` is 99.6% observed, and it has no repeated (user, video) pairs.

**Expected outcome, stated in advance:** an earlier signal-ceiling check found
little personal signal once duration is controlled (simple collaborative
filtering ~0.55–0.59 per-user AUC on a different split). A null result — no
model beating the baselines — is the most likely outcome, and would be reported
as the finding.

## 1. Data

- **Training data:** the `big_matrix` train split (first 80% by timestamp),
  **after removing exact duplicate rows** on (user, video, timestamp). The raw
  file has 968,005 such rows. Split mechanism otherwise unchanged.
- **Evaluation data:** `small_matrix`, deduplicated the same way.
- **Nothing from `small_matrix` is used for training, features, label
  statistics, or baselines.**

### Stated limitations

- **Same time period, not forecasting.** Both matrices cover roughly
  July–September 2020; train runs to Aug 28, `small_matrix` to Sep 5. This
  measures filling in unobserved same-period preferences, not predicting next
  week.
- **Item side features are platform-wide daily aggregates for the same period.**
  The evaluation users' own views are a median 0.03% of an item's total plays
  (90th percentile 0.31%, max 0.92%; median item ~4.3M plays), so direct
  leakage is small. The aggregates are still a strong same-period item-quality
  signal, so B5 below gives a baseline that uses them.

## 2. Users: validation vs test

`small_matrix` users are split once, by hashing the user ID with the project
seed: **30% validation** (~423) and **70% test** (~988).

- Validation is used for checkpoint selection and any tuning. Every
  configuration scored on validation is logged, with a count.
- Test is scored once per pipeline version. No decision may use test users.
- Power (measured on a v1 label): at 80% power the minimum detectable per-user
  AUC difference on test is ~0.004–0.010, depending on how similar the two
  scorers are. On validation it is ~0.006–0.015, so validation cannot reliably
  separate 0.01 differences between dissimilar systems.

## 3. Relevance label

### Primary: within-user top 30% of within-duration-bucket percentiles

1. **Duration buckets:** 50 quantile buckets of video duration. Edges come from
   `big_matrix` train items.
2. **Bucket distributions:** for each bucket, the empirical distribution of
   `watch_ratio` over `big_matrix` train rows.
3. **Percentile:** each evaluation pair's `watch_ratio` becomes its mid-rank
   percentile `p` within its bucket's train distribution. Percentiles, not
   residuals, because short videos' `watch_ratio` is far more spread out
   (looping), so a subtracted median leaves duration in the label.
4. **Relevant** if `p` is in the **top 30% of that user's own percentiles**.
   Every user then has ~30% positives, so heavy watchers do not dominate.

Item quality within a duration bucket is intentionally **kept** — it is a
legitimate signal, with its own baselines (B3, B5).

**Pre-registered label check, run before any model is scored:** the positive
rate within each duration bucket, over all evaluation pairs, must lie in
[0.20, 0.45]. If it fails, that is reported and the label is revisited as a
logged protocol change.

### Sensitivity

The same label is also computed at **20% and 50%**. A claim counts only if its
direction holds at 20%, 30% and 50%.

### Secondary, continuity only

Raw `watch_ratio >= 0.7`. It is never used for a decision.

## 4. What is ranked

For each evaluation user: all of that user's observed `small_matrix` videos
(~3,300).

| system | ordering |
|---|---|
| retrieval | dot product over all items |
| ranker alone | ranker score over all items |
| two-stage (the deployed shape) | retrieval's top 200, reordered by the ranker; the other items follow in retrieval order |

## 5. Metrics

| question | metric |
|---|---|
| single scorers (retrieval, ranker alone, baselines) | **primary:** mean per-user AUC over the full list; also NDCG@10, P@10 |
| does the ranker improve on retrieval? | NDCG@10 and P@10 of two-stage vs retrieval; **within-candidate AUC** (ranker order vs retrieval order over the same 200 items) |
| candidate quality | share of each user's relevant items that land in retrieval's top 200 |

Full-list AUC is **not** used for two-stage vs retrieval, because the ranker
moves only 6% of the list.

**Uncertainty:**
- Paired bootstrap over test users (10,000 resamples, 95% CI).
- For every comparison that passes, an **item bootstrap** as a check, because
  users share the same items: 200 resamples of items, per-user metrics
  recomputed, on 300 test users sampled with a fixed seed.

## 6. Baselines

All computed from `big_matrix` train only, except B5, whose inputs are disclosed.

| id | baseline | definition |
|---|---|---|
| B0 | random | uniform scores, fixed seed |
| B1 | popularity | count of train rows with `watch_ratio >= 0.7` |
| B2 | shortest first | −duration |
| B3 | item quality within duration | mean train percentile `p` per item |
| B4 | item-kNN (collaborative filtering) | see below |
| B5 | platform completion within duration | item's mean daily `play_progress` (`item_daily_features`), as a percentile within its duration bucket; non-personal; same-period data, like the ranker's features |
| B6 | user duration preference | the user's mean train percentile in each of 5 duration bands (quintiles of bucket edges); fall back to the user's overall mean if a band has fewer than 5 train rows |

### B4 definition

- **Input:** user-centred percentiles `c(u, i) = p(u, i) − mean_i p(u, i)` on
  observed train cells only. Unobserved cells are 0 (neutral), so similarity is
  not driven by co-exposure alone.
- **Similarity:** cosine similarity between item columns of that matrix, shrunk
  by `n_common / (n_common + 100)`, where `n_common` is the number of users who
  watched both items. Top 50 neighbours per target item.
- **Score:** `score(u, j) = Σ sim(i, j)·c(u, i) / Σ |sim(i, j)|`, summed over
  neighbours `i` of `j` in the user's train items. If the user watched no
  neighbour, the score is 0.
- **Rows used:** all train users, including `small_matrix` users' `big_matrix`
  rows. That is legitimate, because no pair is shared.

## 7. Decision rule

**Pre-registered comparisons (Holm-corrected together, α = 0.05):**

- retrieval vs each of B1–B6, on primary AUC (6)
- ranker alone vs each of B1–B6, on primary AUC (6)
- two-stage vs retrieval, on NDCG@10 and on within-candidate AUC (2)

A system **adds value** only if all of the following hold:
- It beats **every** baseline B1–B6 on test, Holm-corrected.
- The direction holds at 20%, 30% and 50%.
- The item bootstrap does not reverse the sign.

The ranker **adds value** only if two-stage beats retrieval on both of its
metrics under the same conditions.

Anything else is reported as not adding value, with the numbers.

## 8. Models under this protocol (first run)

The existing pipeline, unchanged except for:
- deduplicated training data;
- checkpoint selection on validation users:
  - **retrieval:** full-list per-user AUC;
  - **ranker:** two-stage NDCG@10.

Training labels, losses and architectures stay as they are, so this run
measures the current system under a fair evaluation. Later training changes
come one at a time, each evaluated under this protocol.

## Amendment history

**v2 clarifications (2026-09-15).** These came from a fresh-context
implementation review after retrieval training had started scoring validation
users for checkpoint selection, and before any test user was scored. None
changes which users or labels training sees.

- **The split is frozen by raw user id** in `config/small_matrix_user_split.json`.
  - Why: the hash was computed on model ids, which any change to filtering
    renumbers, so users could silently move between validation and test.
  - Verified: the frozen file reproduces exactly the split training was
    already using (399 validation, 1,012 test users).
- **B5 weighting.** B5 weights daily `play_progress` by `play_cnt` and ignores
  days with no plays. Those days report `play_progress = 0`, and an unweighted
  mean would penalise items for empty days.
- **Item bootstrap, operationally defined.**
  - A comparison survives only if its 95% item-bootstrap interval lies above
    zero. That is stricter than "does not reverse the sign".
  - Each user's top-200 candidates and two-stage order are held fixed while
    items are resampled.
- **Direction check.** The 20/30/50% check uses the sign of the point
  estimates, not significance at 20% and 50%.
- **Watch ratio is no longer clipped** (`clip_watch_ratio: false`), so the
  label's percentiles do not tie every replayed short video at 1.0. Current
  training uses only the 0.7 / 0.3 thresholds, so it is unaffected. The
  unused `regression` ranker objective would see values up to 573.
- **Realised sizes:**
  - 1,012 test and 399 validation users. The validation share is 28.3%, not
    30%.
  - 3,326 items mapped. 7,051 pairs dropped because their item has no train
    duration.
  - Primary label check: bucket positive rates 0.242–0.446. PASS.
- **Validation scoring is recorded.** Retrieval keeps per-epoch history in its
  meta file; the ranker's epochs are recorded in its training log.

**v1 → v2 (before implementation, after a fresh-context AI-agent review).** The reviewer
computed the following on raw data under v1's label (residual = `watch_ratio`
minus bucket median, within-user top 30%). These numbers were seen before v2
was fixed.

- **Label.** v1's positive rate by duration bucket ran 0.49 → 0.10, and
  shortest-first scored per-user AUC 0.617, the strongest baseline. v2
  percentiles: bucket positive rates 0.24–0.44, shortest-first 0.516. The v2
  check range [0.20, 0.45] was set with that 0.24–0.44 already known.
- **Cutoff sensitivity.** Under v1 the strongest baseline changed between 30%
  and 50%, hence the three-cutoff rule.
- **Two-stage AUC.** Full-list AUC cannot show the ranker's effect (two-stage
  0.4924 vs retrieval-order 0.4920 in an earlier log), hence the top-of-list
  and within-candidate metrics.
- **Baselines.** B5 (the same same-period item aggregates the ranker sees) and
  B6 (per-user duration preference, 0.534 under v1) were added. B3 and B4 were
  fully specified.
- **Comparisons.** Beat every baseline with Holm correction, replacing "beat
  the strongest baseline selected on validation".
- **Uncertainty.** An item bootstrap was added, because users share items.
- **Collapse rule.** v1's "collapse repeats to max" was dropped: `small_matrix`
  has no repeated pairs.
