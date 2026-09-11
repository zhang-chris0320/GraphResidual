# Graph-Residual usage guide

This guide describes the complete local workflow for the public release. The repository contains code and configuration templates; raw datasets, licensed graph databases, logs, and checkpoint binaries are intentionally excluded.

## 1. Install the environment

Using Conda:

    conda env create -f environment.yml
    conda activate graph-residual

Or using an existing Python environment:

    python -m pip install -r requirements.txt

The implementation was authored for Python 3.10 and uses PyTorch, NumPy, pandas, scikit-learn, matplotlib, and PyYAML.

## 2. Prepare data outside the repository

Set aside a project-assets directory containing:

- 64-dimensional graph embeddings for disease, mutation, protein, ptm, and pathway nodes;
- 640-dimensional fixed ESM2 embeddings and validity masks for mutation, protein, and ptm nodes;
- a node mapping table with node_type, node_index, and node_id;
- compressed relation tables under work/graph_core/edges/ named source__relation__target.tsv.gz.

Do not commit ClinVar, BioGRID, Reactome, raw sequences, or other licensed data. Record each asset’s source, version, license, and checksum in a private manifest.

## 3. Validate inputs

Run the input checks before training:

    python scripts/inspect_inputs.py \
      --project_dir /path/to/project-assets \
      --out_dir /path/to/project-assets/work/input_audit

The audit checks embedding shapes, finite values, node-ID alignment, edge bounds, and benchmark coverage.

## 4. Run a sanity job

Use the small relation-preserving sweep before a full run:

    python scripts/run_sanity.py \
      --project_dir /path/to/project-assets \
      --out_dir /path/to/project-assets/work/graph_residual_sanity

Review sanity_selection.json, relation coverage, cosine preservation, and residual-to-ESM norm ratio.

## 5. Train ESM-2 Graph-Residual

Copy configs/esm2.yaml and replace PROJECT_ROOT with the project-assets directory. The portable wrapper resolves that placeholder to the release root; for external assets, pass a config containing the absolute or relative asset directory explicitly.

    python scripts/train.py --config configs/esm2.yaml

Important controls are graph_dim=64, hidden_dim=256, output_dim=640, learning_rate=0.001, batch_size=1024, epochs=20, and seed=42. The training code writes metrics and checkpoints to the configured output directory.

## 6. Use the AMPLIFY-120M implementation

AMPLIFY-120M is kept as an independent backbone family. Use configs/amplify.yaml for the verified contract: graph_dim=64, projection 64→640, normalized sequence/graph vectors, sigmoid gate, bounded positive alpha, and input-norm preservation. The copied official backbone source is under src/backbones/amplify/ and the unchanged P3 Graph-Residual source is src/models/amplify_graph_residual.py. Keep the official AMPLIFY checkpoint and all raw sequence/graph assets outside Git.

## 7. Export embeddings

Keep a reviewed checkpoint outside Git, then export embeddings:

    python scripts/export_embeddings.py \
      --project-dir /path/to/project-assets \
      --checkpoint /path/to/checkpoint.pt \
      --out-dir /path/to/exported-embeddings

The exporter writes node mapping, fusion-mode metadata, float16 embeddings, and an integrity report.

## 8. Evaluate

Mutation–PTM ranking:

    python scripts/evaluate.py mutation_ptm \
      --project_dir /path/to/project-assets \
      --out_dir /path/to/ranking-evaluation

ClinVar-style classification:

    python scripts/evaluate.py clinvar \
      --project_dir /path/to/project-assets \
      --out_dir /path/to/clinvar-evaluation

The packaged evaluators reserve the test split for final reporting and select thresholds/models on validation data.

## 9. Reproduce reference figures locally

The release includes only the tabular source data. Picture outputs are intentionally not committed:

    python scripts/reproduce_figures.py

Generated images appear under figures/reproduced/ locally. The archived selective PPI audit recorded NO_STABLE_GAIN against its controls; generated figures document that evaluation context and should not be read as a performance guarantee.

## 10. Reproducibility checklist

Before sharing a run, record the code commit, configuration file, asset manifest and checksums, Python/package versions, device, random seed, split definition, checkpoint checksum, and evaluation output directory. Keep all raw data and private artifacts outside the Git repository.
