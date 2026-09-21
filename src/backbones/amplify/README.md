# AMPLIFY-120M backbone

This directory contains the copied AMPLIFY-120M official remote-code files and the locked P3 Graph-Residual source protocol. The AMPLIFY Graph-Residual implementation is kept separate from the ESM-2 implementation at `src/models/amplify_graph_residual.py`.

Model weights are intentionally excluded from this public Git repository. The official AMPLIFY-120M backbone bundle is published at [AMPLIFY-120M-GraphResidual-v1](https://huggingface.co/Marcochris/AMPLIFY-120M-GraphResidual-v1). Supply that bundle when running experiments; a separate trained P3 adapter checkpoint is not currently listed there.
