# Configuration guide

The templates use PROJECT_ROOT as a portable placeholder. Replace it with the directory containing the graph and fixed-embedding assets before running a job.

The ESM-2 template uses graph_dim=64, hidden_dim=256, and output_dim=640. The independent AMPLIFY-120M template records graph_dim=64, projection 64→640, normalized gated residual fusion, and input-norm preservation. Training controls include learning_rate, batch_size, epochs, and seed; the original ESM-2 implementation also accepts the short lr alias.

The data loader expects graph embeddings, fixed ESM2 embeddings, validity masks, and relation edge tables. See data/README.md for the asset contract.
