# Data access

This release distributes code and asset contracts only. Original biological resources and licensed/raw datasets are not redistributed.

## Required resources

Obtain current, licensed copies of UniProt, ClinVar, Reactome, Disease Ontology, PTM resources, graph node embeddings, fixed 640-dimensional ESM-2 embeddings, validity masks, node mappings, and relation edge tables. Record source, version, license, accession, and SHA-256 checksum in a local manifest.

## Download and preprocess

1. Create a project-assets directory outside this repository, for example `../project-assets`.
2. Download each resource from its official provider or an approved institutional archive.
3. Apply the provider's license terms and retain the raw files outside Git.
4. Convert graph embeddings to 64-dimensional node features and prepare fixed ESM-2 embeddings with validity masks.
5. Build node mappings with `node_type`, `node_index`, and `node_id`.
6. Store compressed relation tables under `work/graph_core/edges/` with names such as `source__relation__target.tsv.gz` and columns `src_index` and `dst_index`.

## Placement and audit

Set a repository-relative project-assets path and run the input audit before training:

```bash
export PROJECT_ASSETS=../project-assets

python scripts/inspect_inputs.py \
  --project_dir "$PROJECT_ASSETS" \
  --out_dir "$PROJECT_ASSETS/work/input_audit"

python scripts/run_sanity.py \
  --project_dir "$PROJECT_ASSETS" \
  --out_dir "$PROJECT_ASSETS/work/graph_residual_sanity"
```

Do not place raw or licensed data, credentials, logs, or checkpoint binaries in this public repository. The audit output and local manifest should remain local unless they contain no restricted information.
