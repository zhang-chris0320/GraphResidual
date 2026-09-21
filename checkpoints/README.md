# Checkpoints

Model binaries are kept outside public Git history and are published through Hugging Face:

- [ESM2-GraphResidual-v2](https://huggingface.co/Marcochris/ESM2-GraphResidual-v2) — includes the ESM-2 backbone bundle and `models/esm2/residual/graph_residual_v2.pt`.
- [AMPLIFY-120M-GraphResidual-v1](https://huggingface.co/Marcochris/AMPLIFY-120M-GraphResidual-v1) — includes the official AMPLIFY-120M backbone bundle and Graph-Residual P3 source. The current repository does not advertise a separate P3 adapter checkpoint file.

For the published ESM-2 checkpoint, download it automatically and export embeddings:

```bash
python scripts/export_embeddings.py --project-dir /path/to/project-assets --checkpoint-repo Marcochris/ESM2-GraphResidual-v2 --out-dir /path/to/exported-embeddings
```

The default file inside the model repository is `models/esm2/residual/graph_residual_v2.pt`. To use a local checkpoint instead, pass `--checkpoint /path/to/checkpoint.pt`.
