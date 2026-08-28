# Running the project — Google Colab and Azure VM

> **Updated 2026-08-23.** Section 11 (alternative formulations) adds ~10 minutes
> to a warm run: it fits three attribute probes live (two on the validation
> world, one on the full train split) because its headline number should not come
> from a cache. Its two expensive studies — the 5-seed probe study and the five
> Level-3 repair variants — load from `database/alt_study.pt`, which is tracked in
> git. A cold run is roughly **80-100 min**; with all `.pt` caches present a
> re-run is ~15 min. Changing `LR` or `FUSION_CFG` invalidates the §10 caches and
> forces ~50 retrainings — expect ~50 min, it is not a hang. See `HANDOFF.md`.

`Project.ipynb` runs unmodified in two environments. It detects Colab automatically and
adjusts its paths; everything else (extraction, caching, training, evaluation) is the
same code path.

| | **Google Colab** (official — the professor runs it here) | **Azure GPU VM** (development) |
|---|---|---|
| Dataset zip | `MyDrive/datasets/celeba.zip` (Google Drive) | `<repo>/database/celeba.zip` |
| Benchmark JSON | `MyDrive/datasets/celeba_evaluation.json` | `<repo>/database/celeba_evaluation.json` (already in the repo) |
| Extracted dataset | `/content/datasets/celeba/` (runtime SSD, recreated per session) | `<repo>/datasets/celeba/` (persistent, extracted once) |
| Caches (`.pt`) | `MyDrive/datasets/` | `<repo>/database/` |
| GPU | Runtime → T4 | NC-series VM (e.g. T4) |

The large `.pt` caches (embeddings, mined examples, model checkpoint) are **portable**:
they hold tensors and plain dicts, so you can compute them in one environment and copy
them to the other. The small **study** caches (`ablations.pt`, `robustness.pt`,
`lr_sweep.pt`, `anchor_sweep.pt`, `joint_sweep.pt`, `alt_study.pt`, `aux_confirm.pt`,
`collapse_diag.pt`) are **not**: they contain pandas DataFrames, and pandas changed
`StringDtype`'s signature between 2.x and 3.x, so a file written by one major version
fails to unpickle under the other. Since `a312d88` that is not fatal — an unreadable
cache is treated as a miss and the section recomputes — but do not expect copying them
across machines to save time unless the pandas versions agree — the notebook validates them
(model name, split, size) before using them.

---

## 1. Google Colab (the official environment)

The submission requirement is a *single self-contained notebook hosted on Google Colab*,
and the Drive layout matches the course skeleton exactly, so the professor's existing
setup works as-is.

### One-time setup

1. On your Google Drive, make sure `MyDrive/datasets/` contains:
   - `celeba.zip` — the course-provided dataset archive;
   - `celeba_evaluation.json` — the official benchmark file (from Drive/Moodle).
2. Open `Project.ipynb` in Colab (upload it, or open it from GitHub via
   *File → Open notebook → GitHub*).
3. **Runtime → Change runtime type → T4 GPU.**

### Running

*Runtime → Run all.* The first cell prompts for Drive authorization (every new runtime —
this is normal). Expected timings on a T4:

| Stage | First run | Later runs (caches on Drive) |
|---|---|---|
| Setup + unzip | ~3 min | ~3 min (unzip redone per runtime) |
| Embedding extraction (test + train) | ~5 + ~35 min | seconds (cache hit) |
| Triplet mining | ~5 min | seconds (cache hit) |
| Fusion training | ~10 min | seconds (checkpoint hit) |
| All benchmarks + figures | ~5 min | ~5 min |
| **Total** | **~60 min** | **~10 min** |

Caches are written to `MyDrive/datasets/` (~400 MB total, mostly the train embeddings).
If the runtime disconnects mid-extraction, just re-run: the extraction resumes from the
last partial checkpoint instead of restarting.

### Before submitting — checklist

- [ ] `SMOKE_TEST = False` in the config cell.
- [ ] Fresh runtime, *Run all*, zero errors top to bottom.
- [ ] Cell outputs (tables, figures) saved in the submitted `.ipynb`.
- [ ] `TODO` placeholders in §8/§9 replaced with the actual discussion.
- [ ] Team names filled in the title cell.

The professor's run needs only the two course files on her Drive — the notebook
regenerates everything else from scratch (first-run timings above).

---

## 2. Azure GPU VM (development environment)

### 2.1 Provision the VM

- **Size**: `Standard_NC4as_T4_v3` (1× T4 16 GB — same GPU as Colab, cheapest sensible
  choice) or any other NC-series size.
- **Image**: easiest is an image with NVIDIA drivers preinstalled — the *Data Science
  Virtual Machine (Ubuntu)* or NVIDIA's *GPU-optimized* image from the marketplace.
  With a plain **Ubuntu 22.04 LTS** image, install the driver first:

  ```bash
  sudo apt update && sudo apt install -y nvidia-driver-535
  sudo reboot
  ```

- Verify the GPU is visible before anything else:

  ```bash
  nvidia-smi   # must show the T4; if it errors, the driver is not installed
  ```

> Remember to **stop (deallocate) the VM** when you are not using it — NC-series billing
> runs while it is powered on.

### 2.2 Set up the environment

```bash
git clone <your-repo-url> DL-ParetoDeidda
cd DL-ParetoDeidda

python3 -m venv .venv
source .venv/bin/activate

# PyTorch with CUDA support (cu121 wheels work with driver >= 525)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate pandas matplotlib tqdm jupyterlab ipywidgets

python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

### 2.3 Get the dataset onto the VM

`celeba.zip` is **not** in the git repo (it is git-ignored). Copy it into `database/`
with either:

```bash
# from your laptop (if you have the zip locally):
scp celeba.zip azureuser@<vm-ip>:~/DL-ParetoDeidda/database/

# or directly from Google Drive (get the file's share link -> file id):
pip install gdown
gdown <drive-file-id> -O database/celeba.zip
```

`celeba_evaluation.json` is already in `database/` (it ships with the repo).

**That's all.** You do *not* need to extract the zip manually: the notebook's setup cell
detects the missing dataset, extracts `database/celeba.zip` into `datasets/`, and
verifies the layout, printing an actionable error if anything is missing.

If you prefer to extract manually, the layout torchvision requires is:

```
DL-ParetoDeidda/
├── database/
│   ├── celeba.zip
│   ├── celeba_evaluation.json
│   └── (caches *.pt appear here after the first run)
└── datasets/
    └── celeba/                  <- root passed to torchvision is datasets/, NOT datasets/celeba/
        ├── img_align_celeba/    <- the .jpg images
        ├── list_attr_celeba.txt
        ├── list_eval_partition.txt
        └── (other list_*.txt / identity files)
```

The classic failure — `RuntimeError: Dataset not found or corrupted` — comes from
extracting one level too deep (`datasets/celeba/celeba/...`) or from a partial unzip.
Fix: delete `datasets/celeba/` entirely and re-run the setup cell.

### 2.4 Run the notebook

**Important: start Jupyter from the repo root** — the notebook resolves `database/` and
`datasets/` against the working directory. (Alternatively, set
`export DL_PROJECT_ROOT=~/DL-ParetoDeidda` and launch from anywhere.)

Option A — JupyterLab + SSH tunnel:

```bash
# on the VM:
cd ~/DL-ParetoDeidda && jupyter lab --no-browser --port 8888

# on your laptop (separate terminal):
ssh -N -L 8888:localhost:8888 azureuser@<vm-ip>
# then open http://localhost:8888 in your browser
```

Option B — VS Code Remote-SSH: connect to the VM, open the `DL-ParetoDeidda` folder,
open `Project.ipynb` and select the `.venv` kernel. (Make sure the folder you open is the
repo root, for the same working-directory reason.)

First run on the VM: same timings as Colab's first run (~60 min, dominated by the train
embedding extraction). Use `SMOKE_TEST = True` for a fast end-to-end shakedown first.

### 2.5 Moving caches between VM and Colab

To avoid paying the ~40 min extraction twice, copy the cache files between environments
(either direction):

| File | Content |
|---|---|
| `celeba_clip_test_emb.pt` (~41 MB) | test-split CLIP embeddings |
| `celeba_clip_train_emb.pt` (~333 MB) | train-split CLIP embeddings |
| `fusion_examples.pt` | mined training triplets |
| `fusion_model.pt` | trained fusion checkpoint |

VM → Colab: upload them to `MyDrive/datasets/`. Colab → VM: download from Drive into
`database/`. The notebook validates each cache on load (model, split, size, mining/training
parameters) and regenerates it automatically if it does not match — so a wrong or stale
file cannot silently corrupt results.

The **study** caches are deliberately absent from that table. They hold pandas
DataFrames and only load where the pandas major version matches the machine that wrote
them, so copying them across gains nothing and, before `a312d88`, crashed the run. Leave
them where they are and let the other environment recompute: that costs about 50 minutes
on a cold Colab run and nothing at all on the VM, where the committed files are valid.

---

## 3. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `AssertionError: Dataset archive not found: .../celeba.zip` | The zip is not where the environment expects it (see table at the top). On the VM, copy it into `database/`. |
| `RuntimeError: Dataset not found or corrupted` (torchvision) | Wrong extraction layout or partial unzip. Delete `datasets/celeba/` and re-run the setup cell; check the tree in §2.3. |
| `AssertionError: Dataset folder incomplete ... missing: [...]` | Interrupted extraction. Delete `datasets/celeba/` and re-run the setup cell. |
| `torch.cuda.is_available()` is `False` on the VM | Driver not installed (`nvidia-smi` fails) or CPU-only torch wheel installed. Reinstall torch with the `--index-url .../whl/cu121` flag. |
| Notebook can't find `database/` on the VM | Jupyter was launched from another directory. Start it from the repo root or set `DL_PROJECT_ROOT`. |
| `unzip` command not found | Nothing to do — the setup cell automatically falls back to Python's `zipfile` (slower but equivalent). |
| tqdm progress bars not rendering in JupyterLab | `pip install ipywidgets`, then restart the kernel. |
| CUDA out of memory on a small GPU | Lower `batch_size` in the extraction call and `BATCH_SIZE`/`chunk_size` in §7/§4 — all are plain constants. |
| `Cache was built with a different model / split` | A cache file was renamed or moved onto the wrong path. Delete it; it will be regenerated. |
| `Checkpoint was trained with different hyperparameters — retraining` | Not an error: you changed a training constant and the notebook is deliberately refusing the stale checkpoint. |
| Colab asks for Drive authorization every session | Normal Colab behavior; mounts do not persist across runtimes. |
