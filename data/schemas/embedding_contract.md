# Embedding contract

The audited Graph-Residual inputs use the following row-aligned arrays:

| Asset family | Node types | Width | Required companion |
|---|---|---:|---|
| Graph node embeddings | disease, mutation, protein, ptm, pathway | 64 | `node_id_mapping.tsv` |
| Fixed ESM-2 embeddings | mutation, protein, ptm | 640 | sequence-valid mask and mapping |
| AMPLIFY-120M sequence representations | protein and task-specific nodes | 640 | backbone provenance and mapping |

The public repository contains this contract only. Keep the actual NumPy arrays, raw sequences, and model weights in an approved external asset store.
