# Data preparation

Raw data and third-party databases are intentionally excluded. Do not place ClinVar, BioGRID, Reactome, or other licensed/raw datasets in the repository.

Required inputs:
- graph node embeddings with 64 features for disease, mutation, protein, ptm, and pathway;
- fixed ESM2 embeddings with 640 features and validity masks for mutation, protein, and ptm;
- a node mapping table with node_type, node_index, and node_id;
- compressed relation edge tables named source__relation__target.tsv.gz with src_index and dst_index columns.

Download source: obtain graph and fixed-embedding assets from the project’s approved internal archive or an independently licensed public mirror. Record accession, license, version, and checksums in a local manifest before use. Only the code and asset contract are included here.

Preprocessing and audit command:
    python scripts/inspect_inputs.py --project_dir /path/to/project-assets --out_dir /path/to/project-assets/work/input_audit

After the audit passes, run a small sanity job:
    python scripts/run_sanity.py --project_dir /path/to/project-assets --out_dir /path/to/project-assets/work/graph_residual_sanity
