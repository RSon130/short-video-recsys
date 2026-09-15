# Phase 2 NO-GO chapter (pre-drafted)

**Status:** drafted 2026-09-15, before the Phase 0 gate was run on validation
users. **The gate returned NO-GO**; figures filled from run 3, the final run
(engineering log §18). Corrections to the draft's own pre-run claims are
marked *[corrected]*. The plan requires a null-result write-up before the gate, so the outcome
cannot shape the framing. If the gate says GO, this file is kept as a record and
not used.

Bracketed values are filled from `datastore/processed/kuairand_gate.json`; the
text around them does not change.

---

## Do engagement labels carry learnable personal preference? Not detectably, at this scale

**Question.** On KuaiRec the only behavioural label, watch time, mostly measured
video length. KuaiRand adds explicit feedback (like, follow, comment, forward,
hate) and a log of randomly exposed videos, so it can ask whether engagement
carries personal preference once exposure and duration are controlled.

**Test.** Pre-registered, on validation users only:
- Training uses the standard log up to 4/21.
- Evaluation uses the random-exposure log from 4/22 to 5/08.
- A personalised LightGBM (user history, user × tag / author / duration-band
  crosses) is compared with the best of seven non-personal scorers, one of
  them a LightGBM on item features, by per-user AUC.
- GO requires a Holm-significant gain of at least 0.01 on the gate label.

**Result.** On `explicit_positive` (like, follow, comment or forward), across
875 users:

| scorer | GAUC |
|---|---|
| personalised model | 0.541 |
| best non-personal: shortest-first | 0.553 |
| item impression count | 0.541 |

The difference is −0.012, 95% CI −0.033 to +0.010, so the gate is **NO-GO**.

**What limits the conclusion**
- **Positives are sparse on random exposure.**
  - Likes occur on 0.48% of randomly shown videos, against 1.87% on
    recommended ones.
  - Only 705 of 8,147 validation users have both a liked and an unliked
    random video; follow, comment and forward have 77–108 such users.
  - *[corrected]* The pre-run estimate of the paired standard error was
    0.007–0.010. Measured, it is about 0.011–0.015: 70% of users have a single
    positive, and the personal and impressions per-user AUCs are negatively
    correlated. NO-GO means no gain above about 0.03 was detected, not that no
    personal signal exists.
- **Duration is still a strong non-personal signal.** Shortest-first alone
  reaches GAUC 0.555 on likes for validation users *[corrected: the draft's 0.571
  included unknown-duration items as duration 0]*.
- **The training log is small and front-loaded**, with 343,892 labelled
  rows (expanding daily cutoffs, 4/13–4/20).

"No detectable personal signal at this sample size" is the claim.
"Personalisation does not work" is not.

**What still ships**
- The same API and Cloud Run service now serves the best scorer under the
  protocol, which is a non-personal model, and says so in the response
  metadata.
- p95 is re-measured cloud-side.

**What this adds to phase 1**
- Two datasets and two label types now give the same answer under
  exposure-unbiased tests: at public-dataset scale, simple non-personal scorers
  are hard to beat, and duration explains much of what looks like preference.
- The evaluation machinery caught these results instead of reporting a biased
  lift: pre-registration, frozen splits, audit gates, and fresh-context
  AI-agent reviews.
