# Supervisor Demonstration Guide

A concise project demonstration should follow this order.

## 1. Research question

Explain that the project tests whether derivation-aware provenance constraints applied during decoding can prevent complete numerical outputs outside a declared source-derived certificate policy without fine-tuning the language model.

## 2. Architecture

Show `src/numguard_fin/` and explain:

1. Evidence Ledger
2. Typed candidate/proof lattice
3. Candidate selector
4. Complete-answer trie
5. Separate provenance-policy and semantic evaluation gates

## 3. Reproducibility

Show:

- `notebooks/NumGuard_Fin_Colab.ipynb`
- `scripts/`
- `tests/`
- `reproducibility/experiment_plan.json`
- `reproducibility/source_manifest.sha256`

## 4. Validation

Open `validation/validation_summary.json` and `validation/test_run.txt`.

Key retained engineering checks include:

- at least 145 automated tests;
- 12 transparent fixtures;
- 52 candidate-mechanism counterfactual checks;
- no observed hard-lock constraint escapes in the validated fixture and retained development evidence.

Clarify that this validates implementation of the encoded invariant, not semantic truth.

## 5. Candidate bottleneck

Open `results/candidate_diagnostic/` and `results/figures/candidate_recall.png`.

Explain that the reference answer was present in the final legal candidate lattice for 424/871 development examples (48.68%).

## 6. Main results

Open `results/development/method_summary.csv`, `paired_comparisons.csv` and `predictions.csv`.

Explain:

- baseline: 6/871 correct and 46 certificate-policy-invalid complete numbers;
- selector-guided ProofLock: 14/871 correct and zero certificate-policy-invalid complete numbers;
- fragment lock: 92 policy escapes;
- the main practical finding is the gap between structural control and semantic correctness.

## 7. Calibration and stopping

Open `semantic_calibration.json` and `provenance_calibration.json`.

Explain that the semantic gate failed and therefore the public test was not run.

## 8. Dissertation

Show the final report in `docs/dissertation/` and explain that the paper deliberately presents the results as repeatedly adapted development evidence, not as independent test generalisation.
