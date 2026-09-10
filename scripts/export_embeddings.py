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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--chunk", type=int, default=50000)
    args = parser.parse_args()
    export_all(args.project_dir, args.checkpoint, args.out_dir, chunk=args.chunk)


if __name__ == "__main__":
    main()
