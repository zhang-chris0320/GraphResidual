# Graph-Residual

## Overview

Graph-Residual is a relation-aware adaptation framework for combining fixed sequence embeddings with graph-derived node features. This release packages the Graph-Residual ESM2 V2 implementation, portable configuration templates, evaluation entry points, and a small set of reference figure data.

The repository is code-first: raw datasets, licensed graph databases, logs, and training outputs are not included.

## Method

For sequence-bearing node types, the model projects a 64-dimensional graph representation into a 640-dimensional residual and fuses it with a fixed ESM2 representation using a learned scalar gate. Relation-specific vectors score positive and sampled negative edges. Validity masks select the fused representation or a graph-only fallback. Preserve and residual regularizers provide explicit stability checks.

## Installation

    conda env create -f environment.yml
    conda activate graph-residual

Or install dependencies directly:

    python -m pip install -r requirements.txt

## Data preparation

Raw data are excluded. Prepare graph embeddings, fixed ESM2 embeddings, node mapping, and relation edge tables described in data/README.md, then run the input audit before training.

## Training

Edit configs/esm2_residual.yaml so PROJECT_ROOT resolves to your prepared asset directory, then run:

    python scripts/train.py --config configs/esm2_residual.yaml

Use scripts/run_sanity.py first for a small relation-coverage and representation-preservation check.

## Evaluation

The evaluation scripts support mutation–PTM ranking and ClinVar-style classification when the corresponding benchmark assets are available locally:

    python scripts/evaluate.py mutation_ptm --project_dir /path/to/project-assets --out_dir /path/to/eval
    python scripts/evaluate.py clinvar --project_dir /path/to/project-assets --out_dir /path/to/eval

Selection and thresholding are validation-only in the packaged evaluation logic; the test split is reserved for final reporting.

## Reproducing figures

Reference TSV data and curated figure assets are in figures/. Run:

    python scripts/reproduce_figures.py

The archived selective PPI audit recorded NO_STABLE_GAIN against its controls. The figures are included for transparency and reproduction of that evaluation context, not as a performance claim.

## Model checkpoints

Checkpoints are not committed by default. See checkpoints/README.md for the recommended model-registry workflow.

## Citation

If you use this code, cite the associated Graph-Residual / ESM2 V2 manuscript or preprint and the upstream ESM2 model. Bibliographic metadata and DOI should be inserted here once the public manuscript record is available; this release does not invent a citation.

For the full end-to-end workflow, see [docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md).
