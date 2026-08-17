# Google Colab rerun guide

1. Upload the extracted `NumGuard-Fin-Implementation` folder to the top level of Google Drive.
2. Open `notebooks/NumGuard_Fin_Colab.ipynb` in Google Colab.
3. Select **Runtime → Change runtime type → T4 GPU**.
4. Run every cell in order.
5. Download `NumGuard-Fin-Rerun-Evidence.zip` and its checksum after the packaging cell.
6. Do not force the public-test command when `results/rerun/development/semantic_calibration.json` reports `"feasible": false`.

The notebook performs:

```text
install → download FinQA → preflight → clean local validation
→ selector training → candidate gate → development evaluation
→ semantic/provenance calibration → protected public-test decision
→ compact evidence package
```

Fresh outputs are written to:

- `models/candidate_selector.json`
- `models/candidate_selector_training_report.json`
- `results/rerun/candidate_diagnostic/`
- `results/rerun/development/`
- `results/rerun/figures/`
- `results/rerun/execution/`
- `validation/`
- `reproducibility/preflight_report.json`
