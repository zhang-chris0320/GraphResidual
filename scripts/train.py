#!/usr/bin/env python3
"""Train Graph-Residual from a portable YAML configuration."""

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training.training import train_job


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    project_dir = Path(config["project_dir"].replace("PROJECT_ROOT", str(ROOT)))
    out_dir = Path(config["out_dir"].replace("PROJECT_ROOT", str(ROOT)))
    training = dict(config["training"])
    training["lr"] = training.get("learning_rate", training.get("lr", 0.001))
    training.pop("learning_rate", None)
    train_job(
        project_dir,
        out_dir,
        **training,
        beta_init=config["model"]["beta_init"],
        preserve_weight=config["loss"]["preserve_weight"],
        residual_weight=config["loss"]["residual_weight"],
    )


if __name__ == "__main__":
    main()
