from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the empirical-run environment.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    problems: list[str] = []
    if sys.version_info < (3, 10):
        problems.append("Python 3.10 or newer is required.")

    try:
        import torch
        import transformers
    except ImportError as exc:
        problems.append(f"Missing empirical dependency: {exc}")
        torch = None
        transformers = None

    data_files = {split: args.data_dir / f"{split}.json" for split in ("train", "dev", "test")}
    for split, path in data_files.items():
        if not path.exists() or path.stat().st_size == 0:
            problems.append(f"Missing FinQA {split} file: {path}")
        else:
            try:
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if not isinstance(payload, list) or not payload:
                    problems.append(f"FinQA {split} file is not a non-empty JSON list: {path}")
            except (OSError, json.JSONDecodeError) as exc:
                problems.append(f"Invalid FinQA {split} JSON: {exc}")

    cuda_available = bool(torch is not None and torch.cuda.is_available())
    if args.require_cuda and not cuda_available:
        problems.append("CUDA is required for this run but torch.cuda.is_available() is False.")

    free_bytes = shutil.disk_usage(Path.cwd()).free
    if free_bytes < 4 * 1024**3:
        problems.append("Less than 4 GiB of free disk space is available.")

    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": None if torch is None else torch.__version__,
        "transformers": None if transformers is None else transformers.__version__,
        "cuda_available": cuda_available,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "free_disk_gib": round(free_bytes / 1024**3, 2),
        "data_files": {split: str(path) for split, path in data_files.items()},
        "status": "failed" if problems else "passed",
        "problems": problems,
    }
    print(json.dumps(report, indent=2))
    Path("reproducibility").mkdir(parents=True, exist_ok=True)
    Path("reproducibility/preflight_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
