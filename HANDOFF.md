# Handoff — state of the project after the 2026-08-11 experiment session

Written for whoever (human or AI assistant) picks this up next. Read this before
touching `Project.ipynb`.

## TL;DR

The notebook now runs clean end-to-end (34/34 cells, 0 errors, 11 figures) and
the report sections are written from real numbers. The remaining work is
**Colab verification** and one **open design decision** (see below).

| | R@10 |
|---|---|
| C1 text-only control (no image) | 0.065 |
| C2 image-only control (no text) | 0.076 |
| L1 zero-shot arithmetic (mandatory baseline) | 0.110 |
| L2 visual directions | 0.117 |
| **L3 learned fusion (deployed)** | **0.317** single run / **0.324 ± 0.014** over 5 seeds |
| Best configuration measured (`− sign − cross`) | **0.338 ± 0.017** |

## What was done this session

1. **Fixed a crash** (commit `916c01e`): `transformers` 5.x returns
   `BaseModelOutputWithPooling` from `get_image_features`, not a tensor. The
   projection is already applied to `pooler_output` — do **not** re-apply
   `visual_projection`, that silently corrupts text features (they are 512→512
   so it fails quietly rather than erroring). See `clip_features()` in §3.
2. **Ran the pipeline** on the full test split (27,697 source-query pairs).
3. **Added §10** — controls, retrieval-diversity metric, component ablations,
   seed robustness, mining-bias comparison, learning-rate sweep.
4. **Tuned the learning rate**: `1e-4 → 3e-4`. Worth ~+0.08 R@10, more than
   every architectural effect combined.
5. **Wrote §8 and §11** from the measured numbers; filled in team names.
6. **Annotated `METHOD.md`** with a status banner — most of its design claims
   were refuted (see below).

## Things that will bite you if you don't know them

- **Never conclude from one seed or one learning rate.** Three conclusions in
  this project were overturned by adding a controlled dimension:
  - "unbiased mining generalises better" — n=1 seed. False; at n=5 it was worse
    and bimodal. That instability then vanished once the LR was tuned.
  - "sign embeddings are harmful" — n=1 LR. False; an optimisation artefact that
    disappears by `lr=1e-3`.
  - "L3 fails to generalise to custom queries" — an undertuning artefact. At the
    tuned LR, custom-query R@10 went 0.055 → 0.256.
- **Cache keys.** `TRAIN_HPARAMS` keys the ablation, robustness and LR caches.
  Changing `LR` or `FUSION_CFG` invalidates all of them and triggers ~50
  retrainings (~50 min). Expect it; it is not a hang.
- **Hyperparameter selection** uses the *synthetic validation* metric (mined
  queries, train corpus). The test benchmark must never be used to choose
  anything. Keep it that way.
- **The `.pt` caches are gitignored** and live only on the lab VM in
  `database/`. A cold Colab run regenerates everything (~40 min embeddings +
  ~30 min experiments).
- **Do not commit a stripped notebook.** An earlier full run was lost because
  the executed notebook was never saved with outputs. `nbstripout` is active on
  at least one dev machine (Matteo's WSL) but not on the VM.

## Honest caveats in the current results

- **L3 collapses.** It fills only ~27% of its top-10 slots with distinct images
  vs ~75% for L1/L2, and wins most where it concentrates most. It exploits the
  modal structure of the Hamming-based ground truth rather than tracking each
  reference identity. §8.4 says this explicitly — keep it.
- **Only 1 of 4 designed mechanisms is justified** (the gate). See METHOD.md
  banner and §8.3.
- Ablations use 3 seeds; seed/LR studies use 5 and 1. Enough for the large
  effects reported, not the small ones.

## Open decision (needs a human)

`FUSION_CFG` currently deploys `− sign embeddings` at `lr=3e-4`
(0.324 ± 0.014). The **best measured** configuration is `− sign − cross`
(0.338 ± 0.017, 25% fewer parameters). Adopting it would mean rewriting §7 and
METHOD.md further, since cross-attention is the design's centrepiece. Left as
deployed deliberately — this is a framing call about how to present the
contribution, not a technical one.

## Still to do

1. **Colab verification (not done, and the main risk).** The submission must be
   a self-contained Colab notebook the professor runs top-to-bottom. Everything
   so far ran only on the lab VM with `transformers 5.14.1` and warm caches.
   Put `celeba.zip` + `celeba_evaluation.json` in `MyDrive/datasets/`, open the
   notebook, Run all, confirm zero errors. Budget ~70 min cold.
2. Consider trimming §10's grids if the cold Colab runtime is judged too long.
3. Re-check `RUNNING.md` timings — they predate §10 and now understate runtime.
