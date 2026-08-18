# Project Structure

## Active implementation

- `src/numguard_fin/` — core Python package: numeric parsing, evidence representation, retrieval, candidate generation, proof construction, selector logic, trie constraints, evaluation, reporting and calibration.
- `scripts/` — reproducible entry points for dataset download, selector fitting, development evaluation, calibration, engineering verification, counterfactual audit, package validation and rerun packaging.
- `tests/` — unit and integration tests covering the implemented proof language and evaluation pipeline.
- `notebooks/NumGuard_Fin_Colab.ipynb` — clean Colab workflow using the same package and scripts.

## Evidence and outputs

- `results/candidate_diagnostic/` — candidate availability and quality-gate evidence.
- `results/development/` — retained 871-example, 13-method development results including predictions, summaries, paired comparisons and calibration outputs.
- `results/figures/` — figures generated from retained development evidence.
- `results/execution/` — execution log retained from the development experiment.
- `validation/` — fixture evaluation, counterfactual checks, validation summary and test output.
- `reproducibility/` — experiment plan and source-file manifest.

## Documentation

- `docs/dissertation/` — current dissertation.
- `docs/RESULTS_SUMMARY.md` — concise technical result interpretation.
- `docs/SUPERVISOR_DEMO.md` — suggested project demonstration order.
- `docs/technical_validation/` — prior package audit/checksum/validation records.
- `docs/ethics/` — ethics reference note.
- `docs/proposal/` — proposal reference note.

## Historical artefacts

- `development_history/` — five historical implementation release ZIPs, renamed to neutral iteration filenames so they do not interfere with the active package's hygiene checks.
- `evidence_archives/` — frozen final development evidence and the prior validated implementation release.

The historical archives are evidence of the development trail. They are not used by the active implementation when validation or experiments are run.
