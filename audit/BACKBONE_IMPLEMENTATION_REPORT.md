# Backbone implementation report

Date: 2026-09-11

Scope: recover the two independent Graph-Residual backbone families without merging them, modifying model code, or re-implementing a missing model.

## ESM2

Source paths:

- Verified local mirror: `/Users/zhangchris/Desktop/bio/ESM系列/服务器完整备份_20260802_verified/server_root/disease_mutation_ptm_gcl/dmptm_graph_residual_esm2_v2/`
- Matching server source: `/root/autodl-tmp/bio/disease_mutation_ptm_gcl/dmptm_graph_residual_esm2_v2/`

Files copied without code changes:

- `src/models/esm2_graph_residual.py` from `residual_v2/model.py`
- `src/models/relation_decoder.py` from `residual_v2/relation_decoder.py`
- `src/backbones/esm2/config.example.yaml` from the verified source tree
- `src/backbones/esm2/requirements.txt` from the verified source tree
- `configs/esm2.yaml` from the portable release configuration

Verified contract:

- graph dimension: 64
- residual hidden width: 256
- output dimension: 640
- source implementation uses a fixed 640-dimensional ESM representation plus the graph residual path

## AMPLIFY-120M

Source paths:

- Canonical P3 implementation: `/root/autodl-tmp/bio/disease_mutation_ptm_gcl/amplify_generalization/scripts/step3_graph_adaptation.py`
- Official backbone source: `/root/autodl-tmp/bio/disease_mutation_ptm_gcl/data_raw/amplify/AMPLIFY_120M/`
- Local official mirror: `/Users/zhangchris/Desktop/bio/AMPLIFY-120M系列/官方模型与代码/AMPLIFY_120M_official/`

Files copied without code changes:

- `src/models/amplify_graph_residual.py` from the canonical P3 `step3_graph_adaptation.py`
- `src/backbones/amplify/p3_source/step3_graph_adaptation.py` from the same canonical P3 source
- `src/backbones/amplify/p3_source/P3_V2_FUSION_PROTOCOL.json` from the locked protocol
- `src/backbones/amplify/amplify.py`
- `src/backbones/amplify/rmsnorm.py`
- `src/backbones/amplify/rotary.py`
- `src/backbones/amplify/config.json`
- `src/backbones/amplify/special_tokens_map.json`
- `src/backbones/amplify/tokenizer.json`
- `src/backbones/amplify/tokenizer_config.json`
- `configs/amplify.yaml` records the verified P3 contract and selected validation settings

Verified contract:

- official AMPLIFY hidden size: 640
- graph projection: 64 → 640
- sequence and graph vectors are normalized before the gate
- residual uses a sigmoid gate and bounded positive alpha
- fused direction is normalized, then rescaled to preserve the input sequence norm
- the locked protocol is `v2_fusion_coupled`; the backbone remains frozen

## Shared residual component

- `src/models/residual_adapter.py` is copied from `/root/autodl-tmp/bio/disease_mutation_ptm_gcl/graph_residual_ptm_mamba/gr_ptm_mamba/adapter/fusion.py`.
- `src/models/relation_decoder.py` is retained as the exact ESM-2 relation-decoder source and remains separate from either backbone implementation.

## Missing / intentionally excluded

- No standalone AMPLIFY file with the original filename `amplify_graph_residual.py` existed in the source tree. The requested target is therefore a byte-identical copy of the canonical P3 implementation, not a reimplementation.
- Official ESM-2 and AMPLIFY checkpoint weights are excluded from the public release.
- Raw sequences, graph databases, embeddings, cached arrays, training checkpoints, logs, and benchmark data are excluded from the public release.
- No model code was merged, edited, or newly implemented during recovery.

## File identity checks

- AMPLIFY P3 source SHA-256: `60a05ca4a52314802a733414c1a6e09f214dcfd8c42bd91f50a00ccc16cd8e8b`
- ESM2 model source SHA-256: `08ee3d073433a8d0a0b7711681651291d2077c7ea165bba42a5b40d16d690a84`
- Shared PTM-Mamba adapter source SHA-256: `b843f87e282129ef706af5240bbc9604f26d2c5a374e00e103d1bec085de7b4d`

## Status

ESM_IMPLEMENTATION: READY

AMPLIFY_IMPLEMENTATION: READY

SHARED_RESIDUAL_COMPONENT: READY
