from numguard_fin.cli import main

raise SystemExit(main([
    "fit-selector",
    "--data-dir", "data/raw",
    "--split", "train",
    "--retrieval-top-k", "48",
    "--max-direct-candidates", "48",
    "--max-derived-candidates", "192",
    "--folds", "5",
    "--seed", "42",
    "--output", "models/candidate_selector.json",
]))
