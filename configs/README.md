# Configuration guide

The templates use PROJECT_ROOT as a portable placeholder. Replace it with the directory containing the graph and fixed-embedding assets before running a job.

Core dimensions are graph_dim=64, hidden_dim=256, and output_dim=640. Training controls include learning_rate, batch_size, epochs, and seed; the original implementation also accepts the short lr alias.

The data loader expects graph embeddings, fixed ESM2 embeddings, validity masks, and relation edge tables. See data/README.md for the asset contract.
