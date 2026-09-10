# Version selection report

SELECTED_SOURCE:
    ARCHIVED_SERVER_SNAPSHOT/disease_mutation_ptm_gcl/dmptm_graph_residual_esm2_v2

MODEL_VERSION:
    Graph-Residual ESM2 V2 (self-supervised relation pretraining with fixed ESM2 inputs)

WHY_SELECTED:
1. The selected source is an explicit, self-contained module in the verified server snapshot.
2. It contains model, data loader, training, export, sanity, input-audit, and two evaluation entry points.
3. Its archived input-integrity report recorded passing shape, finite-value, node-alignment, relation-bound, and benchmark-coverage checks.
4. Its archived training summary recorded PASS, full relation coverage, and a loadable checkpoint.
5. The archived checkpoint was 5866722 bytes with SHA-256 6326c42d2bc812bc12a74da92e3e69a6aff0366a9761092476a1f93ec8b6d269 and is intentionally not committed here.

ALTERNATIVE_SOURCES:
- LOCAL_DESKTOP_BIO/supplementary_build_v1/inputs/server/clinvar_strict_rebuild_v1/12_code_snapshot/recovered_v2 — compact recovered snapshot; its config hard-codes server paths and it has fewer public entry points.
- LOCAL_DESKTOP_BIO/AMPLIFY-120M系列/服务器完整轻量归档_20260811/AMPLIFY-120M_Project_Archive/Code/amplify_generalization — downstream graph-adaptation and PPI analysis code; it is result-oriented rather than an isolated pretraining implementation.
- LOCAL_DESKTOP_BIO/AMPLIFY-120M系列/现有已验收结果/01_PPI与网络任务/stage2_selective_graph_residual_v1 — evaluation evidence; its locked audit recorded NO_STABLE_GAIN against controls, so it is not used as the core model source.

LIVE_SERVER_STATUS:
    LIVE_SERVER_SOURCE_ROOT was not accessible in the current local environment. The archived server snapshot was used as the traceable fallback; no live-server files were modified.
