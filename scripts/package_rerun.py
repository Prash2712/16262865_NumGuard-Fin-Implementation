from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "NumGuard-Fin-Rerun-Evidence.zip"
CHECKSUM = ROOT / "NumGuard-Fin-Rerun-Evidence.sha256"
FIXED_TIME = (1980, 1, 1, 0, 0, 0)

INCLUDE = [
    ROOT / "models/candidate_selector.json",
    ROOT / "models/candidate_selector_training_report.json",
    ROOT / "results/rerun/candidate_diagnostic",
    ROOT / "results/rerun/development",
    ROOT / "results/rerun/figures",
    ROOT / "results/rerun/execution",
    ROOT / "validation",
    ROOT / "reproducibility/experiment_plan.json",
    ROOT / "reproducibility/source_manifest.sha256",
    ROOT / "reproducibility/preflight_report.json",
]

EXCLUDED_NAMES = {".DS_Store"}
EXCLUDED_PARTS = {".run_state", "__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
EXCLUDED_FILES = {"predictions.jsonl", "selector_training_rows.csv"}


def selected_files() -> list[Path]:
    files: list[Path] = []
    for item in INCLUDE:
        if not item.exists():
            continue
        candidates = [item] if item.is_file() else item.rglob("*")
        for path in candidates:
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT)
            if path.name in EXCLUDED_NAMES or path.name in EXCLUDED_FILES:
                continue
            if any(part in EXCLUDED_PARTS for part in rel.parts):
                continue
            files.append(path)
    return sorted(set(files), key=lambda path: path.relative_to(ROOT).as_posix())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    required = [
        ROOT / "results/rerun/candidate_diagnostic/candidate_quality_gate.json",
        ROOT / "results/rerun/development/predictions.csv",
        ROOT / "results/rerun/development/semantic_calibration.json",
        ROOT / "results/rerun/development/provenance_calibration.json",
    ]
    missing = [path.relative_to(ROOT).as_posix() for path in required if not path.exists()]
    if missing:
        raise SystemExit(
            "Rerun evidence is incomplete. Missing: " + ", ".join(missing)
        )

    files = selected_files()
    if not files:
        raise SystemExit("No rerun evidence was found. Run the Colab workflow first.")

    manifest = {
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ]
    }
    manifest_bytes = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode("utf-8")

    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
        info = zipfile.ZipInfo("rerun_manifest.json", FIXED_TIME)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, manifest_bytes)

    checksum = sha256(OUTPUT)
    CHECKSUM.write_text(f"{checksum}  {OUTPUT.name}\n", encoding="utf-8")
    print(f"Created {OUTPUT.name} with {len(files)} evidence files")
    print(f"SHA-256: {checksum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
