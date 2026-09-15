# Phase 2 plan: KuaiRand

**Status:** approved 2026-09-15, not started. The plan was reviewed by a
fresh-context AI agent before approval. Its corrections are folded in and listed
at the end.

## Why phase 2 exists

On KuaiRec the only behavioural label was watch time, and under an
exposure-unbiased test it mostly measured video length (engineering log
§15–§17). Neither learned stage beat simple baselines. The personal signal left
once duration was controlled was small, about 0.55–0.59 AUC for simple models.

KuaiRand, from the same group, adds two things KuaiRec lacks:
- explicit per-interaction feedback: like, follow, comment, forward, hate;
- a **random-exposure log** that comes after its training window in time.

So it can test the question KuaiRec could not answer:

> **Do engagement signals carry learnable personal preference once exposure
> and duration are controlled?**

Duration bias is the research thread that runs through both phases.

**Story:** On KuaiRec, the only label was watch time, and an unbiased test
showed it mostly measured video length. So I moved to KuaiRand, whose
random-exposure log and explicit feedback labels test whether engagement signals
carry personal preference once exposure and duration are controlled. I shipped
the result behind the same deployed service.

**Wording rules:**
- Published methods (D2Q, WTG, D2Co, CWM) are cited prior work, used as
  comparisons.
- Never say "novel", "we propose" or "our method".
- Do not frame this as a reproduction.
- True and usable: "compares established watch-time debiasing targets on
  KuaiRand's random-exposure log". The papers checked (CWM, RAD) evaluate on
  standard logs.

## Data facts

Confirmed from the [KuaiRand README](https://github.com/chongminggao/KuaiRand)
and [paper](https://arxiv.org/abs/2208.08696). Verify every one locally in
Phase 0.

**Dataset**
- KuaiRand-Pure: 194 MB, CC BY-SA 4.0, downloaded from the Zenodo link in the
  README. 27,285 users.

**Logs**

| file | dates | rows | items |
|---|---|---|---|
| standard | 4/08–4/21 | ~1.1M (estimated from 80 MB) | — |
| standard | 4/22–5/08 | together with the above: 1,436,609 | 7,551 |
| random | 4/22–5/08 only | 1,186,059 | 7,583 |

- 32 items appear only in the random log.
- Random items were one video in a normal recommendation list replaced by a
  uniformly sampled one, across 15 scenarios.
- Every user has at least 10 random items; 55% have 10–25; the mean is about 43.

**Label semantics — critical**
- `is_click`: in the two-column UI it is a real click. In the single-column UI
  it is *valid_play*: `play_time ≥ duration` if duration ≤ 7 s, otherwise
  `play_time > 7 s`. That makes it a watch-time threshold.
- `long_view`: `play_time ≥ duration` if duration ≤ 18 s, otherwise
  `play_time ≥ 18 s`. Also a watch-time threshold.
- `tab`: the scenario, 0–14. Which tabs are two-column is not documented;
  infer it from rows where `is_click` breaks the valid_play rule.

**Leakage**
- `video_features_statistic` holds "average statistics of the video each day
  over one month". That window overlaps the test period.
- `profile_stay_time`, `comment_stay_time` and `is_profile_enter` are recorded
  after the impression.
- User count features (follow, fans) are snapshots of unknown date.

**Unknown until Phase 0:** positive rates per label per log; the tab mix;
whether `play_time` is 0 for unclicked two-column impressions; users with both
classes per task.

## Evaluation design

- **Train:** standard log 4/08–4/21.
- **Evaluate:** random log 4/22–5/08. This is forward in time and
  exposure-unbiased, and each user's random items are a uniform sample of the
  catalogue.
- **Users:** random-log users hash-split 30% validation / 70% test, frozen by
  raw user id (same mechanism as `config/small_matrix_user_split.json`).
- **Contrast only:** standard log 4/22–5/08. It is exposure-biased and never
  used for decisions.
- **Metrics:**
  - primary: per-task **GAUC**, per-user AUC averaged with users equally
    weighted. It estimates catalogue-wide AUC. NDCG is dropped: the per-user
    lists are too short.
  - watch time: XAUC within narrow duration bins.
  - reported per tab.
- **Uncertainty:** user bootstrap plus an item-bootstrap check. Items get
  ~157 random impressions each, so item-rate baselines are noisy.
- **Rare-task fallback:** pooled AUC on *user-centred* scores, never raw pooled
  AUC, plus a user-propensity-only baseline.
- **Baselines:**
  - random;
  - item rate per task (train window);
  - shortest-first;
  - duration-conditioned item rate;
  - user propensity;
  - user × duration band;
  - recency;
  - reranker controls: random and shortest-first reorder.
- **Decision rule:** Holm across pre-listed comparisons, a minimum effect of
  ΔGAUC ≥ 0.01, and a sensitivity check on key thresholds.

**Carried over from KuaiRec:**
- check duration gradients *within* buckets, not only per bucket;
- deduplicate and audit dates;
- a frozen split;
- pre-registration;
- fresh-context AI-agent reviews of design, code and results.

## Phases

### Phase 0: audit and go/no-go gate (~1.5 days)

1. Download, then run a loader and audit: rows, dates, duplicates, id overlap
   between logs, and the 32 cold items.
2. Positive rate per label, per log, per tab. Infer two-column tabs from the
   `is_click` rule check.
3. Duration dependence per label, with gradients within buckets.
4. Per task: users with both classes in validation, and the bootstrap
   half-width of the item-rate baseline. If the half-width exceeds 0.01, the
   task is demoted to secondary or folded into the pre-registered composite
   label `explicit_positive = like | follow | comment | forward`.
5. Signal ceiling on validation users, with settings frozen in advance:
   LightGBM per task vs the best non-personal baseline.

**GO** if at least one *preference* label beats the best non-personal baseline
by ΔGAUC ≥ 0.01, Holm-corrected across labels.
- Preference labels: like, follow, comment, forward, hate, two-column click,
  and the composite.
- Single-column click and `long_view` are watch-time labels and do not count.

**NO-GO:** write the null-result chapter, drafted before running, keep the
build small, and still ship.

A fresh-context review of the audit is required before building on it.

### Phase 1: protocol (~1 day)

Write `docs/evaluation_protocol_kuairand.md` from the design above, with every
threshold fixed. It is reviewed before any model is trained. Tests assert every
feature timestamp is before 4/22, and there is a with/without-statistics-features
ablation.

### Phase 2: models (~3 days)

**1. LightGBM per task.** The strong, cheap reference.

**2. Shared-bottom multi-task model** with heads for like, follow, comment,
forward, hate, two-column click and watch.
- Loss: unweighted per-task BCE plus fixed task weights chosen on validation.
  No `pos_weight`, or recalibrate if used. No GradNorm or PCGrad.
- Negative transfer: each multi-task head vs its single-task twin, paired and
  Holm-corrected.

**3. MMoE** as a one-day-or-less ablation. PLE is skipped: with ~1.1M rows the
differences would sit inside the CIs.

**4. Utility score.** It is a **ranking-time fusion, not the training label**:

```
utility = Σ w_k · p_k − w_hate · p_hate + w_watch · debiased_watch
```

- Calibrate each head on validation users' random rows first, or fuse z-scored
  logits. Probabilities learned on recommended items are miscalibrated on
  random items.
- Weights: one vector pre-registered with a written rationale before any
  validation look, plus a validation sweep reported as a Pareto front of
  per-task GAUC.
- Always evaluated against each task's own label, with a hate-rate check in
  the top ranks. Never against the fused utility.
- Ablation: a single-task model on a hand-crafted utility label, with
  components standardised, `watch_ratio` capped and the debiased watch term,
  and the same fixed weights. Without standardising it becomes a watch-time
  model: `watch_ratio` variance swamps a ~1% like rate.

### Phase 3: duration-bias thread (~1.5 days)

- Watch-time targets on one backbone:
  - raw play time;
  - WTG (Zheng et al., MM 2022);
  - D2Q (Zhan et al., KDD 2022).
- Only single-column rows. In two-column rows, play time depends on the click.
- Evaluation: XAUC on raw play time within narrow duration bins on the random
  log.
- External validity: does the debiased watch score predict like and follow
  better than raw watch time?
- D2Co (RecSys 2023) and CWM (KDD 2024) are cited as related work, not
  implemented in the minimum scope.

### Phase 4: ship (~1 day)

- Brute-force scoring of all ~7.6K items. Phase 1 measured the ANN crossover at
  300–500K items, so there is no retrieval stage.
- Reuse `serving/api.py`, Docker and `scripts/deploy_cloudrun.sh`.
- The response returns top-K with per-head probabilities, the utility score and
  the weights version.
- Re-measure p95 cloud-side.
- **Deploy whichever scorer wins under the protocol, even if it is a baseline**,
  and say so. This replaces the README's current "redeploy only a model that
  beats the baselines".

### Write-up

- README gets a phase 2 chapter.
- Engineering log sections.
- A fresh-context AI-agent review of the results.
- No resume claim until it is done.

## Code changes the loader needs

- `data/base.py` builds id maps only from the interactions it loads, and
  `temporal_split` splits on a global percentile. For KuaiRand:
  - build id maps from all logs;
  - handle the 32 cold items explicitly;
  - split by the log files' date ranges, bypassing `temporal_split`.
- New `data/kuairand.py` behind the existing loader interface. Keep KuaiRec
  runnable.

## Scope and risk

- **Estimate:** 8–10 working days.
- **Minimum viable scope if time slips:**
  - Phase 0;
  - protocol;
  - baselines;
  - LightGBM;
  - shared-bottom multi-task model with fixed-weight fusion vs the
    utility-label ablation;
  - raw vs one debiased watch target;
  - deploy.

**Top risks and the gate for each**

| risk | gate |
|---|---|
| No personal signal, a repeat of KuaiRec | day-1 signal ceiling with frozen settings; null chapter pre-drafted |
| Rare labels too thin to conclude anything | counts of users with both classes and baseline CI half-width before modelling; composite fallback pre-registered |
| A leak or label-semantics error producing a false win | timestamp assertion tests, with/without-statistics ablation, `is_click` rule check per tab, review of the audit |

## Plan review: what changed

The fresh-context AI-agent review of the draft plan made these changes:
- The gate's label set excludes single-column click and `long_view`, which are
  watch-time thresholds.
- The gate needs Holm, frozen settings and a minimum effect. "1 of 5 labels,
  CI > 0" had about a 23% false-go rate.
- Video statistics features are excluded or rebuilt: their window overlaps the
  test period.
- The rare-label plan: counts of users with both classes, CI half-width check,
  composite fallback.
- Pooled AUC only on user-centred scores, plus a user-propensity baseline.
- Watch-time work is limited to single-column rows.
- Retrieval stage dropped.
- The deploy rule resolved.
- PLE dropped, MMoE kept as an ablation.
- Calibration before fusion.
- Effort raised from 5–6 to 8–10 days.

## References

The D2Q and WTG entries are from memory and have not been checked online. Verify
the authors, titles and venues before citing them in the README or anywhere
public.

- KuaiRand: Gao et al., CIKM 2022. [arXiv 2208.08696](https://arxiv.org/abs/2208.08696)
- D2Q: Zhan et al., "Deconfounding Duration Bias in Watch-time Prediction for Video Recommendation", KDD 2022
- WTG: Zheng et al., "DVR: Micro-Video Recommendation Optimizing Watch-Time-Gain under Duration Bias", ACM MM 2022
- D2Co: "Uncovering User Interest from Biased and Noised Watch Time in Video Recommendation", RecSys 2023. [arXiv 2308.08120](https://arxiv.org/abs/2308.08120)
- CWM: Zhao et al., "Counteracting Duration Bias in Video Recommendation via Counterfactual Watch Time", KDD 2024. [arXiv 2406.07932](https://arxiv.org/abs/2406.07932), [code](https://github.com/hyz20/CWM)
- RAD: "Relative Advantage Debiasing for Watch-Time Prediction in Short-Video", [arXiv 2508.11086](https://arxiv.org/abs/2508.11086)
- Ferrari Dacrema et al., "Are We Really Making Much Progress?", RecSys 2019. [arXiv 1907.06902](https://arxiv.org/abs/1907.06902v3)
