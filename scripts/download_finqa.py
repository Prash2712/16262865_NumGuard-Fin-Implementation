from numguard_fin.cli import main

raise SystemExit(main(["download", "--data-dir", "data/raw", "--splits", "train", "dev", "test"]))
