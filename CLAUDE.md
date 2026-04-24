# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

University deep learning project: **Composed Image Retrieval** on the CelebA dataset. Given a **source image + text query** (describing an attribute modification), the system must retrieve target images that match the query from a 19,962-image test split.

The relevant papers are in `documentation/Papers-20260423/`: CLIP, CLAY, and a CLIP Compositionality study. The project likely involves combining image and text embeddings compositionally (e.g., CLIP-based methods).

I need you to help me achieve the objective of the project step by step, and after each step i ask you to do i need you to tell me the full context of what you did and why, and then confirmation of the change:
here is the context of the whole thing into the file "CONTEXT.md"

## Environment

Development is done in **Google Colab** using `Project_Skeleton.ipynb`. The notebook mounts Google Drive to access the dataset.

Dataset setup (run once per Colab session):
```python
from google.colab import drive
drive.mount("/content/drive")
!mkdir /content/datasets
!unzip -q /content/drive/MyDrive/datasets/celeba.zip -d /content/datasets/
```

The local copy of the zip is in `database/celeba.zip` but the notebook expects it on Google Drive.

## Data Format

**CelebA dataset** loaded via `torchvision.datasets.CelebA`:
- `root` must be the parent of the `celeba/` directory (i.e., `/content/datasets`, not `/content/datasets/celeba`)
- `split="test"` → 19,962 images
- `celeba.attr_names` → 40 facial attribute names; `celeba[i][0]` → PIL image, `celeba[i][1]` → attribute tensor

**Evaluation JSON** (`database/celeba_evaluation.json`): a list of query dicts:
```json
{
  "query": "<text query>",
  "ground_truth": {
    "<source_idx_str>": [<list of valid target indices>]
  }
}
```
- Keys in `ground_truth` are **strings** — convert with `int(key)` to get the image index.
- Ground truth targets have Hamming distance ≤ 2 from the source on CelebA attributes.

## Evaluation Metric

`evaluate_retrieval(retrieved_indices, ground_truth_indices, k)` in the notebook:
- **Recall@K**: 1 if any of the top-K retrieved images is in ground truth, else 0.
- **Precision@K**: fraction of top-K retrieved images that are in ground truth.
- `retrieved_indices` must be ordered by descending similarity.

## Attribute Mappings

```python
idx2attribute = {idx: name for idx, name in enumerate(celeba.attr_names)}
attribute2idx = {name: idx for idx, name in enumerate(celeba.attr_names)}
```

