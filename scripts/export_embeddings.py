#!/usr/bin/env python3
"""Export embeddings from a trained checkpoint."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.exporting import export_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, required=True)
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        "--checkpoint",
        type=Path,
        help="Local Graph-Residual checkpoint file.",
    )
    checkpoint_group.add_argument(
        "--checkpoint-repo",
        help="Hugging Face model repository containing the Graph-Residual checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-file",
        default="models/esm2/residual/graph_residual_v2.pt",
        help="Path inside --checkpoint-repo (default: ESM-2 V2 checkpoint).",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face revision/tag/commit to download (default: main).",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chunk", type=int, default=50000)
    args = parser.parse_args()
    checkpoint = args.checkpoint
    if args.checkpoint_repo:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise SystemExit(
                "Install the Hugging Face client first: "
                "python -m pip install huggingface_hub"
            ) from exc
        checkpoint = Path(
            hf_hub_download(
                repo_id=args.checkpoint_repo,
                filename=args.checkpoint_file,
                revision=args.revision,
            )
        )
        print(f"Downloaded checkpoint: {checkpoint}")
    export_all(args.project_dir, checkpoint, args.out_dir, chunk=args.chunk)


if __name__ == "__main__":
    main()
