# NumGuard-Fin

**Certified provenance-constrained decoding for verifiable numerical reasoning in financial question answering**

**Student:** Prasanth Balisetty  
**Student ID:** 16262865  
**Programme:** MSc Data Science and Computational Intelligence  
**Module:** 7005SCN Individual Research Project  
**Supervisor:** Dr Kuda Dube

NumGuard-Fin is an inference-time financial question-answering research artefact that separates **certificate-policy membership** from **semantic correctness**. A typed Evidence Ledger and bounded proof lattice generate direct and derived numeric candidates, a ranking component orders legal candidates, and a complete-answer trie restricts committed outputs to certified candidates or refusal.

The central research finding is deliberately bounded: the whole-answer lock successfully instantiated the implemented admissibility invariant, but this structural property did not produce reliable financial reasoning.

## Key retained development results

- FinQA numerical development examples: **871**
- Evaluated methods: **13**
- Paired prediction rows: **11,323**
- Baseline correct answers: **6/871 (0.69%)**
- Selector-guided ProofLock correct answers: **14/871 (1.61%)**
- Baseline complete numerical outputs rejected by the certificate policy: **46/871**
- Selector-guided ProofLock rejected complete numerical outputs: **0/871**
- Candidate recall: **424/871 (48.68%)**
- Correct selection conditional on recall: **14/424 (3.30%)**
- Fragment-lock certificate-policy escapes: **92**
- Hard-lock certificate-policy escapes in retained development evidence: **0**
- Semantic release gate: **not feasible**
- Public test: **not executed** under the predefined stopping protocol

The repository therefore should not be interpreted as a deployable financial-QA system. It is a research artefact for investigating the boundary between structural provenance control and semantic correctness.

## Repository structure

```text
NumGuard-Fin-GitHub-Complete/
├── README.md
├── RUN_IN_COLAB.md
├── THIRD_PARTY_NOTICES.md
├── CITATION.cff
├── pyproject.toml
├── requirements.txt
├── src/                         # reusable NumGuard-Fin package
├── scripts/                     # data, training, evaluation and validation commands
├── tests/                       # automated test suite
├── notebooks/                   # clean Colab execution notebook
├── data/                        # fixture data and download instructions
├── models/                      # fitted selector and training summary
├── results/                     # retained development evidence and figures
├── validation/                  # engineering checks and validation outputs
├── reproducibility/             # experiment plan and SHA-256 manifests
├── docs/                        # dissertation and project documentation
├── development_history/         # preserved historical release archives
└── evidence_archives/           # frozen evidence and validated release archives
```

See [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md) for a detailed guide.

## Reproduce the local validation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
bash run_validation.sh
```

The validated package contains at least **145 automated tests**, 12 transparent fixture examples and 52 candidate-mechanism counterfactual checks. The retained development summaries and paired comparisons are regenerated from the underlying prediction records during validation.

## Google Colab rerun

The clean notebook is:

```text
notebooks/NumGuard_Fin_Colab.ipynb
```

Upload the repository to Google Drive, use a CUDA-enabled Colab runtime, and follow [`RUN_IN_COLAB.md`](RUN_IN_COLAB.md). Fresh outputs are written separately from the retained development evidence. The public-test command must remain blocked unless the semantic release criterion genuinely becomes feasible.



## Development history

Five historical implementation archives are preserved in `development_history/`. They are retained as evidence of the design-science iteration trail; the active implementation is the code at repository root. See [`development_history/README.md`](development_history/README.md).

## Evidence archives

`evidence_archives/` contains the frozen development-evidence archive and the validated clean implementation release used to construct this repository. These archives are included for traceability and should not be confused with the active source tree.

## Data and model redistribution

Raw FinQA benchmark files and Hugging Face model weights are **not redistributed**. They are downloaded through the supplied scripts. See `data/README.md` and `THIRD_PARTY_NOTICES.md`.



## Claim boundary

The strongest justified conclusion is narrow: whole-answer constrained decoding can enforce membership in the implemented numeric certificate language when the implementation behaves as designed. That result does **not** independently establish exhaustive documentary provenance, semantic financial correctness, or deployment readiness.
