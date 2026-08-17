from numguard_fin.cli import main

candidate_mechanism = [
    "counterfactual", "--data-dir", "data/raw", "--split", "test",
    "--shuffle", "--seed", "42", "--limit", "150",
    "--output", "results/rerun/public_test/counterfactual_candidate_mechanism.csv",
]
if main(candidate_mechanism):
    raise SystemExit(1)

model_output = [
    "counterfactual", "--data-dir", "data/raw", "--split", "test",
    "--shuffle", "--seed", "42", "--limit", "50",
    "--model-level", "--method", "risk_controlled_proof_lock",
    "--model-name", "google/flan-t5-base", "--device", "cuda",
    "--selector-file", "models/candidate_selector.json",
    "--risk-threshold-file", "results/rerun/development/semantic_calibration.json",
    "--retrieval-top-k", "16", "--max-direct-candidates", "16",
    "--max-derived-candidates", "48",
    "--output", "results/rerun/public_test/counterfactual_model_output.csv",
]
raise SystemExit(main(model_output))
