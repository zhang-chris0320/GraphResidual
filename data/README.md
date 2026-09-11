# Data preparation

This directory contains the public, sanitized data contract for Graph-Residual:

- `dataset_manifest.tsv` records every dataset family used by the ESM-2 and AMPLIFY-120M workflows, its server provenance, observed size, and public redistribution status;
- `schemas/` contains header-only schemas for ClinVar-style benchmark rows, graph node mappings, relation edges, and embedding dimensions;
- `examples/clinvar_pathogenicity.synthetic.tsv` is a three-row synthetic fixture for testing parsers. It is not a real ClinVar extract.

The repository does not redistribute raw or licensed data. Do not add ClinVar, BioGRID, IntAct, Reactome, UniProt, Disease Ontology, PTMD2, raw sequences, or model/embedding binaries to this public repository. The manifest is an inventory and provenance record, not a license to redistribute those resources.

Required external inputs:

- graph node embeddings with 64 features for disease, mutation, protein, ptm, and pathway;
- fixed ESM2 or AMPLIFY sequence representations with 640 features and validity masks where required by the selected workflow;
- a node mapping table with `node_type`, `node_index`, and `node_id`;
- compressed relation edge tables named `source__relation__target.tsv.gz` with `src_index` and `dst_index` columns;
- task-specific benchmark tables for mutation–PTM, ClinVar-style classification, and PPI evaluation.

Obtain each external asset from its approved source or an independently licensed mirror. Record accession, version, license, checksum, and download date in a private manifest before use. The verified server provenance is preserved in `dataset_manifest.tsv` without copying the source data into GitHub.

Validate a prepared external asset directory before training:

    python scripts/inspect_inputs.py --project_dir /path/to/project-assets --out_dir /path/to/project-assets/work/input_audit

After the audit passes, run a small sanity job:

    python scripts/run_sanity.py --project_dir /path/to/project-assets --out_dir /path/to/project-assets/work/graph_residual_sanity

The synthetic fixture can be used for parser smoke tests only; it must not be used as a scientific benchmark.
