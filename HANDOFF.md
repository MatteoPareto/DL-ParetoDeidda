# Handoff — state of the project

For whoever (human or AI assistant) picks this up next. Read before touching
`Project.ipynb`.

## TL;DR

The notebook runs clean end-to-end and the report is written from measured
numbers. Remaining work: **Colab verification** (not done — the main risk).

| | R@10 |
|---|---|
| C1 text-only control (no image) | 0.065 |
| C2 image-only control (no text) | 0.076 |
| L1 zero-shot arithmetic (mandatory baseline) | 0.110 |
| L2 visual directions | 0.117 |
| **L3 deployed (slim)** | **0.338** (0.336 ± 0.016 over 5 seeds) |

**3.1× the baseline**, 3.02M trainable params on frozen CLIP. Custom queries:
0.254 vs L1 0.038 / L2 0.068, significant on 3 of 4.

## Deployed model

Self-attention over condition tokens + **mean pooling** (no cross-attention) +
sigmoid gate + residual, sign encoded by negating the text embedding (no learned
sign embeddings). `lr = 3e-4`. This is *not* the architecture `METHOD.md`
describes — see its status banner.

Why: ablations (§10.2) and the joint LR × architecture sweep (§10.6) show the
slim variant is better (0.338 vs 0.301 designed), 25% smaller, and stable across
a 6.7× LR range where the cross-attention model diverges.

## Hard-won lessons — do not relearn these

- **Never conclude from one seed or one learning rate.** Four conclusions here
  were overturned by adding a controlled dimension:
  - "unbiased mining generalises better" — n=1 seed. False at n=5.
  - "sign embeddings are harmful" — n=1 LR. An optimisation artefact.
  - "L3 fails on custom queries" — undertuning. Went 0.055 → 0.254 after LR fix.
  - "unbiased mining is unstable/bimodal" — also undertuning; stable once tuned.
- **The learning rate mattered more than the entire architecture** (~+0.08 R@10).
  If you change the model, re-sweep the LR; they interact strongly.
- **Hyperparameters are selected on the synthetic validation metric only.** The
  test benchmark must never choose anything. Note §10.5: `lambda=0.1` scores
  better on *both* test benchmarks than the deployed `lambda=0`, and we still did
  not adopt it, because validation did not select it. Keep that discipline.
- **Cache keys.** `TRAIN_HPARAMS` keys the ablation/robustness/LR/anchor/joint
  caches. Changing `LR` or `FUSION_CFG` invalidates all of them and triggers ~65
  retrainings (~70 min). Expect it; it is not a hang.
- **Ablating cross-attention makes attention weights uniform.** Anything calling
  `forward(..., return_attn=True)` gets uniform weights by construction. This
  already crashed one run before `forward()` was fixed to return them explicitly.
- **Commit the notebook WITH outputs.** An early full run was lost because it was
  saved stripped. `nbstripout` is active on Matteo's WSL machine but not the VM.

## Honest caveats in the results

- **L3 collapses**: ~26% distinct images in the top-10 vs ~75% for L1/L2. §10.5
  probes this with an identity anchor — diversity rises monotonically with the
  anchor weight, but recall degrades beyond a mild setting. The collapse is
  largely the recall-maximising response to a Hamming-based ground truth, not a
  fixable bug. §8.4 says this; keep it.
- **Only 1 of 4 designed mechanisms is justified** (the gate).
- **`+Male, +Wearing_Lipstick` scores 0.000 for every method**, L3 included.
- Ablations use 3 seeds; anchor 2. Enough for the large effects, not small ones.

## Study caches are committed

`database/{ablations,robustness,lr_sweep,anchor_sweep,joint_sweep}.pt` are a few
KB each (dataframes, not weights) and are **tracked in git** so a cold Colab run
displays every study table instantly instead of retraining ~65 models. Delete one
to force its section to recompute. The large `.pt` files (embeddings, mined
examples, checkpoint) remain gitignored and regenerate on first run.

## Still to do

1. **Colab verification — not done, and the main risk.** Everything ran on the
   lab VM with `transformers 5.14.1` and warm caches. Put `celeba.zip` and
   `celeba_evaluation.json` in `MyDrive/datasets/`, open the notebook, Run all,
   confirm zero errors. Budget ~45-50 min cold (embeddings dominate; the study
   tables load from the committed caches).
2. Explore the mild identity anchor (`lambda ~ 0.1`) properly — more seeds, finer
   grid, and ideally a validation metric that rewards diversity so the selection
   rule can see what the test metric already does.
3. `RUNNING.md` timings are approximate; re-check after a real Colab run.
