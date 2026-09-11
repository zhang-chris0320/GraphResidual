# Graph-Residual

Graph-Residual is a relation-aware adaptation framework that integrates heterogeneous biological graph context into pretrained protein language-model embeddings through a bounded residual pathway.

This repository provides separate implementations for ESM-2 and AMPLIFY-120M. The release is code-first: raw datasets, licensed graph databases, logs, model checkpoints, and generated pictures are not redistributed.

## Installation

### Clone repository

```bash
git clone https://github.com/zhang-chris0320/GraphResidual.git
cd GraphResidual
```

### Create environment

```bash
conda env create -f environment.yml
conda activate graph-residual
```

If you are using an existing Python environment:

```bash
python -m pip install -r requirements.txt
```

The package specification is in `environment.yml`; the direct dependency fallback is `requirements.txt`.

## Repository structure

```text
GraphResidual/
├── src/           # model and training implementation
├── configs/       # experiment configurations
├── scripts/       # training, evaluation, and utility entry points
├── checkpoints/   # checkpoint usage guidance; binaries stay outside Git
├── docs/          # usage and data-access documentation
├── audit/         # release and implementation audit reports
└── README.md
```

Generated pictures and archived figure-data tables are intentionally outside the current public tree. Figure reproduction is described below as a local-only workflow.

## Model

Supported backbones:

- ESM-2
- AMPLIFY-120M

The shared Graph-Residual contract uses a 640-dimensional sequence representation and a 64-dimensional graph representation.

For ESM-2, the residual adapter follows `64 → 256 → 640` with `graph_dim=64`, hidden width `256`, and output width `640`.

For AMPLIFY-120M, the independent P3 path projects graph features `64 → 640`, normalizes the sequence and graph vectors, applies a learned sigmoid gate and bounded positive residual, and preserves the input sequence norm. The backbone families remain separate; the adapter does not merge or modify them.

## Data preparation

The original biological resources are not redistributed. Obtain and license the required resources independently, then place the prepared assets in a project-assets directory outside the repository.

Required resources include:

- UniProt
- ClinVar
- Reactome
- Disease Ontology
- PTM resources
- 64-dimensional graph node embeddings
- fixed 640-dimensional ESM-2 embeddings and validity masks
- node mappings and compressed relation edge tables

See [`docs/data_access.md`](docs/data_access.md) for the download, preprocessing, placement, and audit workflow. Do not upload raw or licensed data to GitHub.

## Checkpoints

No Graph-Residual checkpoint binaries are bundled in this public repository, and no verified GraphResidual Hugging Face model URL is configured here. Obtain an approved, versioned checkpoint from the project maintainer or model registry, record its license and SHA-256 checksum, and keep the binary outside Git.

For a local checkpoint, use a path such as:

```bash
mkdir -p checkpoints
cp ../approved-checkpoints/graph_residual_v2.pt checkpoints/
```

The recommended registry and provenance workflow is documented in [`checkpoints/README.md`](checkpoints/README.md).

## Training

### ESM-2 Graph-Residual

The verified generic training entry point is:

```bash
python scripts/train.py --config configs/esm2.yaml
```

Run `scripts/run_sanity.py` first after preparing the required assets. The requested launcher `scripts/train_esm2.sh` is not present in this release; use the Python entry point above.

### AMPLIFY-120M Graph-Residual

Use `configs/amplify.yaml` as the verified dimension and fusion contract. The requested launcher `scripts/train_amplify.sh` is not present in this release, so its launcher status is **Coming soon**. The official AMPLIFY source and the unchanged P3 Graph-Residual source remain under `src/backbones/amplify/` and `src/models/amplify_graph_residual.py`; keep AMPLIFY weights outside Git.

## Evaluation

The dispatcher requires a task name. Set a repository-relative project-assets directory and run the command corresponding to the benchmark assets you prepared:

```bash
export PROJECT_ASSETS=../project-assets

python scripts/evaluate.py mutation_ptm \
  --project_dir "$PROJECT_ASSETS" \
  --out_dir "$PROJECT_ASSETS/work/ranking-evaluation"

python scripts/evaluate.py clinvar \
  --project_dir "$PROJECT_ASSETS" \
  --out_dir "$PROJECT_ASSETS/work/clinvar-evaluation"
```

The evaluation contract includes AUROC, AUPRC, MCC, MRR, MAP, Hits@k, and nDCG metrics as applicable. Model and threshold selection is validation-only; reserve the test split for final reporting. `configs/evaluation.yaml` records the metric and bootstrap contract.

## Reproduce figures

The public release does not include picture files or the archived `figures/` data directory. The reproduction script is retained for local use, but it requires an approved local figure-data bundle at `figures/`:

```bash
python scripts/reproduce_figures.py
```

Generated outputs are local-only. Do not commit the generated pictures or licensed source data. The archived selective PPI audit recorded `NO_STABLE_GAIN` against its controls; any reproduced plots document that evaluation context and are not a performance claim.

## Citation

Use the following placeholder until the associated manuscript or preprint record is public:

```bibtex
@article{graphresidual,
  title  = {Graph-Residual: Relation-aware adaptation of protein embeddings},
  author = {Graph-Residual authors},
  year   = { n.d. }
}
```

Insert the final authors, venue, DOI, and URL only when verified.

## Limitations

The current implementation depends on available graph context and prepared relation-specific assets. It is not presented as a generic unseen-node inductive model or as a universal predictor.

## Path and command audits

See [`audit/README_PATH_CHECK.md`](audit/README_PATH_CHECK.md) and [`audit/README_COMMAND_CHECK.md`](audit/README_COMMAND_CHECK.md) for the release checks.
