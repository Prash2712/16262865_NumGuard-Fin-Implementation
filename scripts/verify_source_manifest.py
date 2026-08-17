from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reproducibility" / "source_manifest.sha256"
EXCLUDED_TOP_LEVEL = {"validation", "results", "models"}
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".run_state", ".venv"}
EXCLUDED_FILES = {MANIFEST.resolve(), (ROOT / "reproducibility" / "preflight_report.json").resolve()}


def source_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if relative.parts[0] in EXCLUDED_TOP_LEVEL:
            continue
        if any(
            part in EXCLUDED_PARTS or part.endswith(".egg-info")
            for part in relative.parts
        ):
            continue
        if relative.parts[:2] == ("data", "raw"):
            continue
        if path.resolve() in EXCLUDED_FILES or path.suffix == ".zip":
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_manifest() -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{digest(path)}  {path.relative_to(ROOT).as_posix()}"
        for path in source_files()
    ]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} source hashes to {MANIFEST.relative_to(ROOT)}")


def verify_manifest() -> int:
    if not MANIFEST.exists():
        print(f"Missing manifest: {MANIFEST}")
        return 1
    expected: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        checksum, relative = line.split("  ", 1)
        expected[relative] = checksum

    actual_files = {
        path.relative_to(ROOT).as_posix(): path for path in source_files()
    }
    problems: list[str] = []
    for relative, checksum in expected.items():
        path = actual_files.get(relative)
        if path is None:
            problems.append(f"missing: {relative}")
        elif digest(path) != checksum:
            problems.append(f"changed: {relative}")
    for relative in sorted(set(actual_files) - set(expected)):
        problems.append(f"unrecorded: {relative}")

    if problems:
        print("SOURCE MANIFEST FAILED")
        for problem in problems:
            print(f"- {problem}")
        return 1
    print(f"SOURCE MANIFEST PASSED ({len(expected)} files)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Write or verify source-file SHA-256 hashes.")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        write_manifest()
        return 0
    return verify_manifest()


if __name__ == "__main__":
    raise SystemExit(main())
