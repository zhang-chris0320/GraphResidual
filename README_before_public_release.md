# Graph-Residual

## Overview

Graph-Residual is a relation-aware adaptation framework for combining fixed sequence embeddings with graph-derived node features. This release packages separate Graph-Residual implementations for ESM-2 and AMPLIFY-120M, portable configuration templates, and evaluation entry points.

The repository is code-first: raw datasets, licensed graph databases, logs, and training outputs are not included.

## Method

The ESM-2 implementation uses graph_dim=64, residual hidden width 256, and output_dim=640. The AMPLIFY-120M P3 implementation is a separate frozen-backbone path: graph features are projected 64→640, normalized before a learned sigmoid gate, fused with a bounded positive residual, and rescaled to preserve the input sequence norm. Neither backbone is merged into the other or modified by the adapter.

## Installation

    conda env create -f environment.yml
    conda activate graph-residual

Or install dependencies directly:

    python -m pip install -r requirements.txt

## Data preparation

Raw data are excluded. Prepare graph embeddings, fixed ESM2 embeddings, node mapping, and relation edge tables described in data/README.md, then run the input audit before training.

## Training

For ESM-2, edit configs/esm2.yaml so PROJECT_ROOT resolves to your prepared asset directory, then run:

    python scripts/train.py --config configs/esm2.yaml

Use scripts/run_sanity.py first for a small relation-coverage and representation-preservation check.

For AMPLIFY-120M, use configs/amplify.yaml as the verified dimension and protocol contract. The exact P3 source is in src/models/amplify_graph_residual.py and the official backbone source is in src/backbones/amplify/; keep the AMPLIFY checkpoint outside Git.

## Evaluation

The evaluation scripts support mutation–PTM ranking and ClinVar-style classification when the corresponding benchmark assets are available locally:

    python scripts/evaluate.py mutation_ptm --project_dir /path/to/project-assets --out_dir /path/to/eval
    python scripts/evaluate.py clinvar --project_dir /path/to/project-assets --out_dir /path/to/eval

Selection and thresholding are validation-only in the packaged evaluation logic; the test split is reserved for final reporting.

## Reproducing reference tables

Reference TSV data are in figures/. Picture files are intentionally not part of the public release. Run the local reproduction script only when you need to generate figures outside Git:

    python scripts/reproduce_figures.py

The archived selective PPI audit recorded NO_STABLE_GAIN against its controls. Generated figures document that evaluation context and are not a performance claim.

## Model checkpoints

Checkpoints are not committed by default. See checkpoints/README.md for the recommended model-registry workflow.

## Citation

If you use this code, cite the associated Graph-Residual / ESM2 V2 manuscript or preprint and the upstream ESM2 model. Bibliographic metadata and DOI should be inserted here once the public manuscript record is available; this release does not invent a citation.

For the full end-to-end workflow, see [docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md).
