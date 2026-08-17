# Data

`data/fixtures/finqa_structured_fixture.json` contains transparent FinQA-shaped cases used only for engineering verification.

The official FinQA dataset is deliberately not redistributed. Download it with:

```bash
python scripts/download_finqa.py
```

Expected files:

```text
data/raw/train.json
data/raw/dev.json
data/raw/test.json
data/raw/dataset_manifest.json
```

The manifest records source URLs, file sizes and SHA-256 hashes.
