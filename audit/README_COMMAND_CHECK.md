# README command check

Checked against the public `main` tree on 2026-09-11.

| README workflow item | Status | Evidence |
|---|---|---|
| `git clone` | PASS | Repository URL verified |
| `cd GraphResidual` | PASS | Repository-relative project path |
| `conda env create -f environment.yml` | PASS | `environment.yml` exists |
| `conda activate graph-residual` | PASS | Environment name verified from `environment.yml` |
| `python -m pip install -r requirements.txt` | PASS | `requirements.txt` exists |
| `python scripts/train.py --config configs/esm2.yaml` | PASS | `scripts/train.py` and `configs/esm2.yaml` exist |
| `scripts/train_esm2.sh` | COMING SOON | File not present; README uses the verified Python entry point |
| `scripts/train_amplify.sh` | COMING SOON | File not present; README labels the launcher as Coming soon |
| `python scripts/inspect_inputs.py ...` | PASS | `scripts/inspect_inputs.py` exists |
| `python scripts/run_sanity.py ...` | PASS | `scripts/run_sanity.py` exists |
| `python scripts/evaluate.py mutation_ptm ...` | PASS | Dispatcher and task exist |
| `python scripts/evaluate.py clinvar ...` | PASS | Dispatcher and task exist |
| `python scripts/reproduce_figures.py` | LOCAL-ONLY | Script exists but public `figures/` inputs were intentionally removed |
| `python scripts/evaluate.py --config configs/evaluation.yaml` | NOT USED | Not valid syntax for the task-based dispatcher; README uses task-based syntax |

## Result

Installation, ESM-2 training, input inspection, sanity checks, and task-based evaluation commands match files in the public tree. The AMPLIFY shell launcher is explicitly marked unavailable/Coming soon, and figure reproduction is explicitly local-only rather than presented as a public runnable workflow.
