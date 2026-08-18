# Engineering Log

What was changed, why, and what it measured. Chronological. Every number here was
produced by a run in this repo, not estimated.

The project began as a complete, well-documented codebase that had **never been
executed**: no data downloaded, no checkpoints, no metrics — the README's results
table was seven dashes. Nine defects surfaced, none of which was visible by
reading the code. That is the theme worth carrying out of this project: the code
looked finished.

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
layer, saturating the sigmoid. `system_design.md` §4.1 had specified "normalised
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

---

## How a pair is labelled

The question an interviewer asks as *"how do you decide one item is better than
another?"* — the answer is that the model never decides. It is read off observed
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
- **Cloud Run deployment** and p50/p95 under load.
