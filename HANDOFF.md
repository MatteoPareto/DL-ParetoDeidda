# Handoff — state of the project

For whoever (human or AI assistant) picks this up next. Read before touching
`Project.ipynb`.

## TL;DR

The notebook runs clean end-to-end and the report is written from measured
numbers. Section 11 (added 2026-08-23) questions the formulation the first ten
sections assume and changes the headline. Remaining work: **Colab verification**
(still not done — the main risk).

| | R@10 |
|---|---|
| C1 text-only control (no image) | 0.065 |
| C2 image-only control (no text) | 0.076 |
| L1 zero-shot arithmetic (mandatory baseline) | 0.110 |
| L2 visual directions | 0.117 |
| L3 deployed fusion (slim) | 0.338 (0.336 ± 0.016 over 5 seeds) |
| **A predicted-attribute scoring (§11)** | **0.558** (0.557 ± 0.002 over 5 seeds) |
| A with a plain 20.5k-param linear probe instead | 0.544 |
| attribute oracle — LEAKAGE, not a method | 1.000 |

## The two results, and why both are reported

**L3** is the composed-query method the assignment sketches: 3.02M trainable
params on frozen CLIP, 3.1× the mandatory baseline, open-vocabulary (any text
condition CLIP can encode).

**A** (§11.3) drops the single-vector assumption. A probe maps each frozen CLIP
embedding to 40 attribute probabilities offline, and candidates are scored with
the benchmark's ground-truth rule kept as two separate clauses:
`log P(satisfies C) + w_i · E[agreement off C]`. 1.13M params, +65% relative over
L3, significant on 11/13 queries, 0.398 vs 0.275 on the custom queries.

**Capacity is not what makes it work.** Swap the MLP probe for plain logistic
regression — 20,520 parameters, no attention, no mining, no contrastive loss —
and it still scores **0.544**. That number, not 0.558, is the one that says
something about the benchmark.

**A is not strictly better — it is narrower.** It answers exactly 40 fixed
attributes; L1–L3 answer arbitrary text. Report both, and report the trade-off.
Do not quietly replace L3 with A in the narrative.

## What §11 established (do not relearn)

- **The benchmark is attribute matching.** `satisfies constraints AND Hamming ≤ 2
  on the rest` reproduces `celeba_evaluation.json` **exactly**, for every source
  of every query. The oracle over true test attributes scores R@10 = 1.000 —
  labelled as leakage in the notebook, and it must stay labelled.
- **The diagnostics predicted the result before any training.** Corrupting oracle
  bits at 10% gives R@10 = 0.61; the probe's held-out error is 9.6% and A scores
  0.558. Two cells, no training, ~1 minute. Run diagnostics like this first.
- **The CLIP-space branch is redundant given predicted attributes.** Adding the
  trained L3 query to A at the validation-selected weight is worth **+0.007**.
- **The collapse is a property of the metric.** A collapses onto as few distinct
  images as L3 does, sharing no architecture, loss, or space with it. §8.4's
  claim now has independent support.
- **Three of the four alternatives failed, and the failures are reported.**
  F (retrieve-then-rerank) is a truncation of A and improves monotonically as it
  stops truncating. H (training-free conditional-similarity mask) is selected
  away — validation picks λ = 0. E (repair L3 with intra-modal visual tokens plus
  an explicit `<log_mu(v_ref), d_a>` satisfaction feature) does not beat the
  design it was meant to repair.

## Hard-won lessons — do not relearn these either

- **Never conclude from one seed or one learning rate.** Five conclusions here
  were overturned by adding a controlled dimension:
  - "unbiased mining generalises better" — n=1 seed. False at n=5.
  - "sign embeddings are harmful" — n=1 LR. An optimisation artefact.
  - "L3 fails on custom queries" — undertuning. 0.055 → 0.254 after the LR fix.
  - "unbiased mining is unstable/bimodal" — also undertuning.
  - "the composed query is the object to optimise" — never tested until §11.
- **The learning rate mattered more than the entire architecture** (~+0.08 R@10).
  If you change the model, re-sweep the LR; they interact strongly.
- **Hyperparameters are selected on validation only.** §11 keeps this: its
  validation world is a 32,770-image held-out slice of TRAIN, with the probe
  fitted on the other 130,000 and synthetic queries mined there under the
  official rule. The benchmark chooses nothing, in §11 as in §10.
- **Watch the sign convention.** `sign_ids` are `0 = "+" / 1 = "-"` (the collate
  convention), while mined `signs` are `±1`. Feeding one where the other is
  expected silently produces a plausible-looking, badly wrong benchmark number.
  This cost one full §11.7 run.
- **Cache keys.** `TRAIN_HPARAMS` keys the §10 caches. Changing `LR` or
  `FUSION_CFG` invalidates all of them and triggers ~65 retrainings (~70 min).
- **Ablating cross-attention makes attention weights uniform** by construction.
- **Commit the notebook WITH outputs.** `nbstripout` is active on Matteo's WSL
  machine but not the VM.

## Honest caveats in the results

- **Both L3 and A collapse**: ~26% distinct images in the top-10 vs ~75% for
  L1/L2. §10.5 probes it with an identity anchor; §11.9 explains why it is the
  metric, not the model.
- **Only 1 of 4 designed L3 mechanisms is justified** (the gate).
- **`+Male, +Wearing_Lipstick`** is 0.000 for every composed-query method; A
  scores 0.040 — the only non-zero result on it.
- Ablations use 3 seeds; anchor 2; the two seed studies 5.
- A's ceiling is attribute-prediction accuracy (90.4% held-out mean), not CLIP.

## Study caches are committed

`database/{ablations,robustness,lr_sweep,anchor_sweep,joint_sweep,alt_study}.pt`
are a few KB each (dataframes, not weights) and are **tracked in git** so a cold
Colab run displays every study table instantly instead of retraining ~70 models.
`alt_study.pt` holds §11's seed study and the five Level-3 repair variants; delete
it to force §11 to recompute (~20 min). The large `.pt` files (embeddings, mined
examples, checkpoints) remain gitignored and regenerate on first run.

§11 deliberately does *not* cache the probe or A's benchmark evaluation: the
headline number is recomputed live on every run (~6 min).

## The composed-query ceiling (added 2026-08-26)

Five independent attempts to raise the single-composed-vector number on frozen
ViT-B/32 all land in the same band:

| attempt | section | benchmark R@10 |
|---|---|---|
| component ablations | 10.2 | 0.258 - 0.338 |
| learning-rate sweep | 10.4 | plateau ~0.32 - 0.34 |
| identity anchor | 10.5 | 0.337 best under validation |
| Level-3 repair (visual tokens + satisfaction) | 11.7 | 0.285 - 0.326 |
| **auxiliary attribute loss** | **11.8** | **0.340 (vs 0.338 control)** |

Section 11.9 says the same thing from the other side: adding the whole trained
fusion module on top of attribute scoring is worth +0.007. **~0.34 is a ceiling
for this formulation and this backbone, not a tuning gap.** Do not spend more
time tuning composed-query methods expecting a materially different number.

### What 11.8 actually found

Auxiliary attribute supervision (predict the target's 40-bit profile from the
composed query, head discarded at inference) does NOT lift the benchmark:
0.340 +/- 0.013 at the validation-selected beta = 5, against 0.338 +/- 0.008 for
the beta = 0 control.

It DOES improve generalisation: custom-query R@10 goes 0.261 -> 0.304, winning on
**5 seeds out of 5**. So the attribute signal helps a composed query travel to
query families it was never mined for, but cannot move the benchmark the way
scoring attributes directly (approach A) does.

A cautionary detail: at 3 seeds, beta = 3 looked like a real win (0.353). At 5
seeds it fell to 0.345 +/- 0.011 with 3 wins / 2 losses. **That is the fourth
time in this project a promising signal evaporated under more seeds.** Treat any
3-seed result here as provisional.

### The only two ways past 0.34

1. **Change the formulation** (Section 11's approach A, 0.558) — abandons the
   fusion mechanism and the open-vocabulary property.
2. **Change the backbone** (e.g. ViT-L/14). Section 11.1's noisy-oracle curve
   shows attribute accuracy is the whole bottleneck and it is steep: 9% bit
   error -> ~0.62, 5% -> ~0.90. Better embeddings lift both paths. But
   ViT-B/32 is the mandated model to report, so this is an extra comparison,
   not a better mandatory number.

Neither is free. Note also that the assignment's bar is beating the L1 baseline
(0.110); the deployed model is 3.1x that.

## Still to do

1. **Submit.** The deliverable is `Project.ipynb` on this branch: 77 cells, 47/47
   executed, zero errors, outputs committed. Nothing else in the repository is part
   of the submission.
2. Optional, and deliberately not done: a learned conditional-similarity mask
   (Section 11.4 rules out only the training-free one) and distilling the probe's
   signal back into a composed query so a single vector inherits what A knows while
   staying open-vocabulary. Both are weeks of work, not days.

## Colab verification — DONE (2026-08-28)

Run end to end on Colab with a T4. It works, and it found one bug that only
appears there.

**The bug.** The study caches hold pandas DataFrames pickled by pandas 3.0.5 on
the lab VM. Colab ships pandas 2.x, and `StringDtype`'s constructor signature
changed between them, so `torch.load` raised

```
TypeError: StringDtype.__init__() takes from 1 to 2 positional arguments but 3 were given
```

and stopped Run all at Section 10.2. Fixed in `a312d88`: all twelve cache reads go
through `load_cache()`, which treats an unreadable file as a cache MISS and
recomputes instead of aborting.

**What this means for cache portability.** The claim that all `.pt` caches are
portable between machines is wrong and has been corrected in `RUNNING.md`:

- **Portable** (tensors and plain dicts): `celeba_clip_test_emb.pt`,
  `celeba_clip_train_emb.pt`, `fusion_examples.pt`, `fusion_model.pt`. These were
  copied VM -> Drive and loaded on Colab without complaint, saving the ~40 minute
  extraction.
- **Not portable** (contain DataFrames): the study caches — `ablations.pt`,
  `robustness.pt`, `lr_sweep.pt`, `anchor_sweep.pt`, `joint_sweep.pt`,
  `alt_study.pt`, `aux_confirm.pt`, `collapse_diag.pt`. They load only where the
  pandas major version matches the one that wrote them. Since `a312d88` a mismatch
  costs time, not a crash.

**Note:** a fresh run with only the two course files on Drive never had the study
caches and so never hit this. It is triggered by copying them across.
