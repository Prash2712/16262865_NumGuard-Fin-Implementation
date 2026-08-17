from __future__ import annotations

import ast
import hashlib
import json
import random
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Optional

from .schemas import FinQAExample


OFFICIAL_URLS = {
    "train": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/train.json",
    "dev": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/dev.json",
    "test": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json",
}


def _as_text_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return (str(value).strip(),)


def _coerce_row(row: Any) -> tuple[str, ...]:
    if isinstance(row, (list, tuple)):
        return tuple(str(cell).strip() for cell in row)
    if isinstance(row, str):
        stripped = row.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, (list, tuple)):
                    return tuple(str(cell).strip() for cell in parsed)
            except (SyntaxError, ValueError):
                pass
        if "|" in stripped:
            return tuple(cell.strip() for cell in stripped.split("|"))
        return (stripped,)
    return (str(row).strip(),)


def _coerce_table(raw: dict[str, Any]) -> tuple[tuple[str, ...], ...]:
    table = raw.get("table_ori") or raw.get("table") or []
    if isinstance(table, dict):
        headers = table.get("header") or table.get("headers") or []
        rows = table.get("rows") or table.get("data") or []
        table = [headers, *rows] if headers else rows
    if not isinstance(table, (list, tuple)):
        return ()
    rows = tuple(_coerce_row(row) for row in table)
    width = max((len(row) for row in rows), default=0)
    if width == 0:
        return ()
    return tuple(row + ("",) * (width - len(row)) for row in rows)


def coerce_example(raw: dict[str, Any]) -> FinQAExample:
    qa = raw.get("qa") if isinstance(raw.get("qa"), dict) else {}
    question = raw.get("question") or qa.get("question") or ""
    reference_answer = (
        raw.get("reference_answer")
        or raw.get("answer")
        or qa.get("answer")
        or qa.get("exe_ans")
    )
    program = raw.get("reference_program") or qa.get("program")
    if isinstance(program, list):
        program = ", ".join(str(token) for token in program)
    explanation = raw.get("reference_explanation") or qa.get("explanation")
    support = qa.get("gold_inds") if isinstance(qa.get("gold_inds"), dict) else {}
    example_id = str(raw.get("id") or raw.get("uid") or qa.get("id") or "").strip()
    return FinQAExample(
        example_id=example_id,
        question=str(question).strip(),
        pre_text=_as_text_tuple(raw.get("pre_text")),
        post_text=_as_text_tuple(raw.get("post_text")),
        table=_coerce_table(raw),
        reference_answer=None if reference_answer is None else str(reference_answer).strip(),
        reference_program=None if program is None else str(program).strip(),
        reference_explanation=None if explanation is None else str(explanation).strip(),
        reference_support=support,
        filename=str(raw.get("filename") or "").strip() or None,
    )


def load_examples(
    path: str | Path,
    *,
    limit: Optional[int] = None,
    seed: int = 42,
    shuffle: bool = False,
    require_references: bool = True,
) -> list[FinQAExample]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload if isinstance(payload, list) else payload.get("data", [])
    examples: list[FinQAExample] = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        example = coerce_example(raw)
        if not example.example_id or not example.question or not example.table:
            continue
        if require_references and not example.reference_answer:
            continue
        examples.append(example)
    if shuffle:
        random.Random(seed).shuffle(examples)
    if limit is not None:
        examples = examples[:limit]
    return examples


def resolve_split_file(data_dir: str | Path, split: str) -> Path:
    data_dir = Path(data_dir)
    candidates = [
        data_dir / f"{split}.json",
        data_dir / f"finqa_{split}.json",
        data_dir / f"finqa_{'validation' if split == 'dev' else split}.json",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    names = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No FinQA {split!r} file found. Checked: {names}")


def download_official_finqa(data_dir: str | Path, splits: Iterable[str] = ("train", "dev", "test")) -> dict[str, dict[str, str]]:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, str]] = {}
    for split in splits:
        if split not in OFFICIAL_URLS:
            raise ValueError(f"Unsupported split: {split}")
        destination = data_dir / f"{split}.json"
        temporary = destination.with_suffix(".json.part")
        request = urllib.request.Request(
            OFFICIAL_URLS[split],
            headers={"User-Agent": "NumGuard-Fin research artefact"},
        )
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        # Validate JSON before replacing an existing file.
        with temporary.open("r", encoding="utf-8") as handle:
            json.load(handle)
        temporary.replace(destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        manifest[split] = {
            "path": str(destination),
            "bytes": str(destination.stat().st_size),
            "sha256": digest,
            "source": OFFICIAL_URLS[split],
        }
    manifest_path = data_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
