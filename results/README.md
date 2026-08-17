# Results directory

- `development/` contains the retained FinQA development evidence used in the dissertation analysis.
- `candidate_diagnostic/` contains retained candidate-availability and failure-stage evidence.
- `figures/` contains the retained development plots.
- `execution/` contains the retained development execution log.
- `rerun/` is created by the Colab workflow and keeps newly generated evidence separate from the retained results.

Large or duplicative artefacts are deliberately excluded from the submission package, including selector training-row caches, duplicate JSONL exports for the full development run, interrupted checkpoints and pre-repair logs.
