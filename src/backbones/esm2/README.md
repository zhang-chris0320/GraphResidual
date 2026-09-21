# ESM-2 backbone

The Graph-Residual ESM-2 implementation is copied without code changes from the verified `residual_v2` source tree. The model implementation is exposed at `src/models/esm2_graph_residual.py`; the original configuration example and dependency list are retained beside this file.

The ESM-2 checkpoint and embedding assets are intentionally not committed to this public Git repository. The verified model bundle, including the backbone files and Graph-Residual checkpoint, is published at [ESM2-GraphResidual-v2](https://huggingface.co/Marcochris/ESM2-GraphResidual-v2). Prepare the remaining project assets locally and keep them outside Git.
