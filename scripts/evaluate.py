#!/usr/bin/env python3
"""Dispatch the packaged evaluation scripts."""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["mutation_ptm", "clinvar"])
    parser.add_argument("args", nargs=argparse.REMAINDER)
    values = parser.parse_args()
    name = "evaluate_mutation_ptm_residual_v2.py" if values.task == "mutation_ptm" else "evaluate_clinvar_residual_v2.py"
    script = Path(__file__).resolve().parents[1] / "src/evaluation" / name
    raise SystemExit(subprocess.call([sys.executable, str(script), *values.args]))


if __name__ == "__main__":
    main()
