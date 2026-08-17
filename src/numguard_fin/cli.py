from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import re
import sys
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd
import numpy as np

from .calibration import calibration_curve, select_risk_threshold
from .candidates import build_candidate_lattice
from .counterfactual import audit_counterfactual_predictions, audit_counterfactuals
from .dataset import download_official_finqa, load_examples, resolve_split_file
from .evidence import build_evidence_ledger
from .intent import parse_question_intent
from .model import HuggingFaceGenerator
from .numeric import parse_single_number
from .pipeline import DEFAULT_METHODS, PipelineConfig, run_example, run_examples
from .reporting import generate_research_outputs
from .retrieval import retrieve_evidence, score_evidence
from .selector import candidate_training_rows, fit_linear_selector, load_selector


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()




def _portable_path(path: str | Path) -> str:
    """Return a project-relative path for manifests without exposing local directories."""
    value = Path(path)
    try:
        return value.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return value.name if value.is_absolute() else value.as_posix()


def _portable_command(arguments: list[str]) -> list[str]:
    output: list[str] = []
    for argument in arguments:
        text = str(argument)
        candidate = Path(text)
        output.append(_portable_path(candidate) if candidate.is_absolute() else text)
    return output

def _write_example_ids(path: Path, examples) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{example.example_id}\n" for example in examples),
        encoding="utf-8",
    )
    return _sha256(path)


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in ("torch", "transformers", "pandas", "numpy", "scipy", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _hardware_info() -> dict[str, object]:
    details: dict[str, object] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    try:
        import torch

        details.update(
            {
                "torch_cuda_available": bool(torch.cuda.is_available()),
                "torch_cuda_version": getattr(torch.version, "cuda", None),
                "cudnn_version": (
                    torch.backends.cudnn.version()
                    if hasattr(torch.backends, "cudnn")
                    else None
                ),
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "gpu_names": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ]
                if torch.cuda.is_available()
                else [],
            }
        )
    except ImportError:
        details["torch_cuda_available"] = False
    return details


def _dataset_path(args) -> Path:
    return Path(args.path) if getattr(args, "path", None) else resolve_split_file(args.data_dir, args.split)


def _load_split(args) -> tuple[list, Path]:
    path = _dataset_path(args)
    examples = load_examples(
        path,
        limit=None,
        seed=args.seed,
        shuffle=False,
        require_references=True,
    )
    if getattr(args, "numeric_only", True):
        examples = [
            example
            for example in examples
            if parse_single_number(example.reference_answer or "") is not None
        ]
    if getattr(args, "shuffle", False):
        random.Random(args.seed).shuffle(examples)
    if args.limit is not None:
        examples = examples[: args.limit]
    if not examples:
        raise RuntimeError("No eligible numeric FinQA examples were loaded.")
    return examples, path


def _set_reproducibility_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def command_download(args) -> int:
    manifest = download_official_finqa(args.data_dir, args.splits)
    print(json.dumps(manifest, indent=2))
    return 0


def command_fit_selector(args) -> int:
    examples, dataset_path = _load_split(args)
    rows = candidate_training_rows(
        examples,
        retrieval_top_k=args.retrieval_top_k,
        max_direct=args.max_direct_candidates,
        max_derived=args.max_derived_candidates,
    )
    selector, report = fit_linear_selector(rows, seed=args.seed, folds=args.folds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = selector.to_dict()
    payload["metadata"].update({
        "dataset_path": _portable_path(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "split": args.split,
        "reference_usage": "training labels only; references are never exposed to retrieval, candidate construction or decoding",
        "selector_training_candidate_pool": {
            "retrieval_top_k": args.retrieval_top_k,
            "direct_pool_limit": args.max_direct_candidates,
            "derived_pool_limit": args.max_derived_candidates,
            "deployment_direct_limit": args.max_direct_candidates,
            "deployment_derived_limit": args.max_derived_candidates,
            "policy": "fit on the frozen deployment population; learned influence is chosen by grouped out-of-fold safe blending",
        },
    })
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    report_path = args.output.with_name("candidate_selector_training_report.json")
    report_path.write_text(json.dumps(payload["metadata"], indent=2, allow_nan=False), encoding="utf-8")
    if args.rows_output:
        pd.DataFrame(rows).to_csv(args.rows_output, index=False)
    print(json.dumps(payload["metadata"], indent=2))
    print(f"Saved selector: {args.output}")
    return 0


REFERENCE_NUMBER_RE = re.compile(r"(?<![#\w.])[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?![\w.])")
REFERENCE_OPERATION_RE = re.compile(r"([a-z_]+)\s*\(", flags=re.IGNORECASE)


def _reference_operations(program: str | None) -> tuple[str, ...]:
    if not program:
        return ()
    return tuple(dict.fromkeys(match.lower() for match in REFERENCE_OPERATION_RE.findall(program)))


def _reference_operands(program: str | None) -> tuple[Decimal, ...]:
    """Extract only literal evidence operands from a FinQA program.

    Intermediate references (``#0``) and symbolic constants (``const_100``) are excluded.
    The routine is diagnostic only: it never participates in retrieval, candidate construction,
    selector training, decoding or public-test inference.
    """
    if not program:
        return ()
    operations = _reference_operations(program)
    if operations and operations[0] in {"table_lookup", "text_lookup"}:
        return ()
    cleaned = re.sub(r"const_[-+]?\d+(?:\.\d+)?", " ", program, flags=re.IGNORECASE)
    cleaned = re.sub(r"#\d+", " ", cleaned)
    output: list[Decimal] = []
    for raw in REFERENCE_NUMBER_RE.findall(cleaned):
        try:
            value = Decimal(raw)
        except InvalidOperation:
            continue
        if value not in output:
            output.append(value)
    return tuple(output)


def _value_present(value: Decimal, entries) -> bool:
    for entry in entries:
        item = entry.item if hasattr(entry, "item") else entry
        for observed in (item.value, item.base_value):
            scale = max(abs(value), abs(observed), Decimal("1"))
            if abs(value - observed) <= max(Decimal("0.01"), Decimal("0.0005") * scale):
                return True
    return False


def _operand_coverage(operands: tuple[Decimal, ...], entries) -> tuple[int, float]:
    if not operands:
        return 0, float("nan")
    present = sum(_value_present(value, entries) for value in operands)
    return present, present / len(operands)


def _intent_reference_operation_match(intent, reference_operations: tuple[str, ...]) -> bool:
    if not reference_operations:
        return False
    family_to_primitives = {
        "direct_lookup": {"table_lookup", "text_lookup"},
        "difference": {"subtract"},
        "percentage_change": {"subtract", "divide", "multiply"},
        "indexed_return": {"subtract"},
        "sum": {"add", "table_sum"},
        "table_sum": {"add", "table_sum"},
        "average": {"add", "divide", "average", "table_average"},
        "table_average": {"add", "divide", "average", "table_average"},
        "ratio": {"divide"},
        "margin": {"divide", "multiply"},
        "product": {"multiply"},
        "table_min": {"table_min"},
        "table_max": {"table_max"},
    }
    licensed: set[str] = set()
    for family in (intent.operation, *intent.alternative_operations):
        licensed.update(family_to_primitives.get(family, {family}))
    return bool(licensed & set(reference_operations))


def _failure_stage(
    *,
    final_recall: bool,
    preselector_recall: bool,
    unpruned_recall: bool,
    expanded_recall: bool,
    operand_count: int,
    ledger_operand_rate: float,
    retrieval_operand_rate: float,
    operation_match: bool,
) -> str:
    if final_recall:
        return "success"
    if operand_count and not math.isnan(ledger_operand_rate) and ledger_operand_rate < 1.0:
        return "extraction_or_program_constant"
    if operand_count and not math.isnan(retrieval_operand_rate) and retrieval_operand_rate < 1.0:
        return "retrieval"
    if expanded_recall and not unpruned_recall:
        return "retrieval_or_bounded_context"
    if not operation_match:
        return "operation_schema"
    if not expanded_recall:
        return "candidate_generation"
    if unpruned_recall and not preselector_recall:
        return "bounded_pruning"
    if preselector_recall and not final_recall:
        return "selector_ranking"
    return "unresolved"


def command_inspect(args) -> int:
    examples, path = _load_split(args)
    selector = load_selector(args.selector_file)
    scorer = None if selector is None else selector.score
    rows = []
    operation_counts: dict[str, int] = {}
    diagnostic_direct_limit = max(256, args.max_direct_candidates * 8)
    diagnostic_derived_limit = max(768, args.max_derived_candidates * 8)

    for example in examples:
        intent = parse_question_intent(example.question)
        operation_counts[intent.operation] = operation_counts.get(intent.operation, 0) + 1
        ledger = build_evidence_ledger(example)
        retrieved = retrieve_evidence(
            ledger, example.question, intent, top_k=args.retrieval_top_k
        )

        # Frozen deployment lattice: learned selector followed by frozen candidate bounds.
        lattice = build_candidate_lattice(
            retrieved,
            intent,
            include_derived=True,
            max_direct=args.max_direct_candidates,
            max_derived=args.max_derived_candidates,
            candidate_scorer=scorer,
        )
        # Same evidence and frozen bounds, but before learned rescoring.
        preselector_lattice = build_candidate_lattice(
            retrieved,
            intent,
            include_derived=True,
            max_direct=args.max_direct_candidates,
            max_derived=args.max_derived_candidates,
            candidate_scorer=None,
        )
        # Same retrieved evidence with effectively unbounded pruning. This isolates candidate
        # construction from the frozen lattice limits.
        unpruned_lattice = build_candidate_lattice(
            retrieved,
            intent,
            include_derived=True,
            max_direct=diagnostic_direct_limit,
            max_derived=diagnostic_derived_limit,
            minimum_direct_confidence=0.0,
            minimum_derived_confidence=0.0,
            candidate_scorer=None,
        )
        # Full-ledger oracle diagnostic. Gold references are still not exposed to the mechanism.
        full_scored = sorted(
            [score_evidence(item, example.question, intent) for item in ledger],
            key=lambda entry: (-entry.score, entry.item.evidence_id),
        )
        full_lattice = build_candidate_lattice(
            full_scored,
            intent,
            include_derived=True,
            max_direct=max(diagnostic_direct_limit, len(full_scored)),
            max_derived=diagnostic_derived_limit,
            minimum_direct_confidence=0.0,
            minimum_derived_confidence=0.0,
            candidate_scorer=None,
        )

        final_recall = bool(lattice.contains_reference(example.reference_answer))
        preselector_recall = bool(preselector_lattice.contains_reference(example.reference_answer))
        unpruned_recall = bool(unpruned_lattice.contains_reference(example.reference_answer))
        expanded_recall = bool(full_lattice.contains_reference(example.reference_answer))
        reference_operations = _reference_operations(example.reference_program)
        reference_operands = _reference_operands(example.reference_program)
        ledger_present, ledger_rate = _operand_coverage(reference_operands, ledger)
        retrieved_present, retrieved_rate = _operand_coverage(reference_operands, retrieved)
        operation_match = _intent_reference_operation_match(intent, reference_operations)

        rows.append(
            {
                "example_id": example.example_id,
                "question": example.question,
                "reference_answer": example.reference_answer,
                "reference_program": example.reference_program,
                "operation": intent.operation,
                "alternative_operations": "|".join(intent.alternative_operations),
                "reference_operation": reference_operations[0] if reference_operations else "",
                "reference_operations": "|".join(reference_operations),
                "intent_reference_operation_match": operation_match,
                "reference_operand_count": len(reference_operands),
                "reference_operands_in_ledger": ledger_present,
                "reference_operands_in_ledger_rate": ledger_rate,
                "reference_operands_retrieved": retrieved_present,
                "reference_operands_retrieved_rate": retrieved_rate,
                "ledger_items": len(ledger),
                "retrieved_items": len(retrieved),
                "direct_candidates": lattice.direct_count,
                "derived_candidates": lattice.derived_count,
                "candidate_recall": final_recall,
                "candidate_recall_preselector": preselector_recall,
                "candidate_recall_unpruned_same_retrieval": unpruned_recall,
                "expanded_ledger_candidate_recall_diagnostic": expanded_recall,
                "failure_stage": _failure_stage(
                    final_recall=final_recall,
                    preselector_recall=preselector_recall,
                    unpruned_recall=unpruned_recall,
                    expanded_recall=expanded_recall,
                    operand_count=len(reference_operands),
                    ledger_operand_rate=ledger_rate,
                    retrieval_operand_rate=retrieved_rate,
                    operation_match=operation_match,
                ),
            }
        )

    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.head(10).to_string(index=False))
    print("\nOperation counts:", operation_counts)

    overall_recall = float(frame["candidate_recall"].mean())
    preselector_recall = float(frame["candidate_recall_preselector"].mean())
    unpruned_recall = float(frame["candidate_recall_unpruned_same_retrieval"].mean())
    expanded_recall = float(frame["expanded_ledger_candidate_recall_diagnostic"].mean())
    direct_mask = frame["reference_operation"].isin(
        ["table_lookup", "text_lookup", "table_sum", "table_average", "table_min", "table_max"]
    )
    direct_recall = (
        float(frame.loc[direct_mask, "candidate_recall"].mean())
        if direct_mask.any()
        else float("nan")
    )
    failure_counts = {
        str(key): int(value)
        for key, value in frame["failure_stage"].value_counts(dropna=False).items()
    }

    print(f"Candidate recall (final frozen lattice): {overall_recall:.3f}")
    print(f"Candidate recall (before selector): {preselector_recall:.3f}")
    print(f"Candidate recall (unpruned same retrieval): {unpruned_recall:.3f}")
    print(f"Direct/aggregate reference recall: {direct_recall:.3f}")
    print(f"Expanded-ledger candidate recall diagnostic: {expanded_recall:.3f}")
    print("Failure-stage counts:", failure_counts)
    print(f"Dataset SHA-256: {_sha256(path)}")
    print(f"Saved: {args.output}")

    passed = overall_recall >= args.minimum_candidate_recall and (
        math.isnan(direct_recall) or direct_recall >= args.minimum_direct_recall
    )
    gate = {
        "candidate_recall": overall_recall,
        "candidate_recall_preselector": preselector_recall,
        "candidate_recall_unpruned_same_retrieval": unpruned_recall,
        "expanded_ledger_candidate_recall_diagnostic": expanded_recall,
        "minimum_candidate_recall": args.minimum_candidate_recall,
        "direct_reference_recall": None if math.isnan(direct_recall) else direct_recall,
        "minimum_direct_recall": args.minimum_direct_recall,
        "failure_stage_counts": failure_counts,
        "examples": int(len(frame)),
        "dataset_path": _portable_path(path),
        "dataset_sha256": _sha256(path),
        "selector_file": None if args.selector_file is None else _portable_path(args.selector_file),
        "reference_usage": (
            "references are used only after lattice construction for diagnostic labels and gate metrics"
        ),
        "passed": passed,
    }
    gate_path = args.output.with_name("candidate_quality_gate.json")
    gate_path.write_text(json.dumps(gate, indent=2, allow_nan=False), encoding="utf-8")
    if args.enforce_gates and not gate["passed"]:
        print("CANDIDATE QUALITY GATE FAILED — do not run the expensive model experiment.")
        return 2
    print(
        "CANDIDATE QUALITY GATE PASSED"
        if gate["passed"]
        else "Candidate diagnostic completed; gates were not enforced."
    )
    return 0


def _read_risk_threshold(args) -> float:
    if args.risk_threshold_file:
        payload = json.loads(Path(args.risk_threshold_file).read_text(encoding="utf-8"))
        if payload.get("feasible") is False:
            raise RuntimeError(
                "The development calibration is infeasible under the frozen risk/coverage gates. "
                "Public-test execution is blocked; revise the method on train/dev only."
            )
        return float(payload["threshold"])
    return float(args.risk_threshold)


def command_run(args) -> int:
    _set_reproducibility_seed(args.seed)
    examples, dataset_path = _load_split(args)
    methods = tuple(part.strip() for part in args.methods.split(",") if part.strip()) if args.methods else DEFAULT_METHODS
    risk_threshold = _read_risk_threshold(args)
    selector = load_selector(args.selector_file)
    selector_id = _sha256(Path(args.selector_file))[:12] if args.selector_file else "heuristic"
    config = PipelineConfig(
        retrieval_top_k=args.retrieval_top_k,
        max_direct_candidates=args.max_direct_candidates,
        max_derived_candidates=args.max_derived_candidates,
        risk_threshold=risk_threshold,
        seed=args.seed,
        dataset_split=args.split,
        model_name="heuristic-engineering-check" if args.engineering_check else args.model_name,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        methods=methods,
        selector_id=selector_id,
    )

    generator = None
    model_revision = None
    resolved_device = "none"
    if not args.engineering_check:
        generator = HuggingFaceGenerator(
            args.model_name,
            device=args.device,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
        resolved_device = generator.device
        model_revision = getattr(generator.model.config, "_commit_hash", None)

    output_dir = Path(args.output_dir)
    state_dir = output_dir / ".run_state"
    checkpoint_path = state_dir / "predictions_checkpoint.jsonl"
    checkpoint_rows: dict[tuple[str, str], dict] = {}
    if args.resume and checkpoint_path.exists():
        with checkpoint_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("configuration_id") != config.configuration_id:
                    raise RuntimeError(
                        "The checkpoint configuration does not match the current run. "
                        "Remove the output directory or rerun without --resume."
                    )
                checkpoint_rows[(str(row["example_id"]), str(row["method"]))] = row
    elif checkpoint_path.exists():
        checkpoint_path.unlink()

    complete_methods = set(methods)
    completed_examples = {
        example.example_id
        for example in examples
        if {
            method
            for (example_id, method) in checkpoint_rows
            if example_id == example.example_id
        }
        == complete_methods
    }
    state_dir.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("a", encoding="utf-8") as checkpoint:
        for index, example in enumerate(examples, start=1):
            if example.example_id in completed_examples:
                continue
            example_records = run_example(example, generator=generator, config=config, selector=selector)
            for record in example_records:
                row = record.to_dict()
                checkpoint_rows[(record.example_id, record.method)] = row
                checkpoint.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            checkpoint.flush()
            print(f"Completed {index}/{len(examples)} examples", flush=True)

    records = [
        checkpoint_rows[(example.example_id, method)]
        for example in examples
        for method in methods
        if (example.example_id, method) in checkpoint_rows
    ]
    expected_rows = len(examples) * len(methods)
    if len(records) != expected_rows:
        raise RuntimeError(
            f"Run is incomplete: expected {expected_rows} paired rows, found {len(records)}."
        )
    paths = generate_research_outputs(
        records,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        seed=args.seed,
    )
    manifest = {
        "run_type": "engineering_validation" if args.engineering_check else "empirical_experiment",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_split": args.split,
        "dataset_path": _portable_path(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "examples": len(examples),
        "numeric_only": args.numeric_only,
        "model_name": "heuristic-engineering-check" if args.engineering_check else args.model_name,
        "model_revision": model_revision,
        "requested_device": "none" if args.engineering_check else args.device,
        "resolved_device": resolved_device,
        "seed": args.seed,
        "configuration_id": config.configuration_id,
        "risk_threshold": risk_threshold,
        "selector_file": None if args.selector_file is None else _portable_path(args.selector_file),
        "selector_sha256": None if args.selector_file is None else _sha256(Path(args.selector_file)),
        "methods": methods,
        "python": sys.version,
        "hardware": _hardware_info(),
        "dependencies": _package_versions(),
        "command": _portable_command(sys.argv),
        "resumed_from_checkpoint": bool(args.resume and completed_examples),
        "checkpoint_path": str(checkpoint_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    example_ids_path = output_dir / "example_ids.txt"
    manifest["example_ids_file"] = str(example_ids_path)
    manifest["example_ids_sha256"] = _write_example_ids(example_ids_path, examples)
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    print(f"Configuration ID: {config.configuration_id}")
    return 0


def command_calibrate(args) -> int:
    frame = pd.read_csv(args.predictions)
    method_rows = frame[frame["method"] == args.method].to_dict(orient="records")
    if not method_rows:
        raise RuntimeError(f"No rows found for method {args.method!r}.")
    decision = select_risk_threshold(
        method_rows,
        target_risk=args.target_risk,
        confidence_level=args.confidence,
        risk_type=args.risk_type,
        minimum_coverage=args.minimum_coverage,
        minimum_accepted=args.minimum_accepted,
    )
    curve = calibration_curve(
        method_rows, confidence_level=args.confidence, risk_type=args.risk_type
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    configuration_ids = sorted(
        {str(row.get("configuration_id")) for row in method_rows if row.get("configuration_id")}
    )
    payload = decision.to_dict()
    payload.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "method": args.method,
            "source_predictions": _portable_path(args.predictions),
            "source_predictions_sha256": _sha256(Path(args.predictions)),
            "source_configuration_ids": configuration_ids,
            "confidence_field": "selected_candidate_confidence",
        }
    )
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    curve_path = args.output.with_name(f"{args.output.stem}_curve.csv")
    curve.to_csv(curve_path, index=False)
    print(json.dumps(payload, indent=2))
    return 0


def command_counterfactual(args) -> int:
    examples, dataset_path = _load_split(args)
    rows = []
    manifest: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_split": args.split,
        "dataset_path": _portable_path(dataset_path),
        "dataset_sha256": _sha256(dataset_path),
        "examples": len(examples),
        "seed": args.seed,
        "shuffle": bool(args.shuffle),
        "audit_layer": "model_output" if args.model_level else "candidate_mechanism",
        "example_ids": [example.example_id for example in examples],
    }
    if args.model_level:
        _set_reproducibility_seed(args.seed)
        risk_threshold = _read_risk_threshold(args)
        selector = load_selector(args.selector_file)
        selector_id = _sha256(Path(args.selector_file))[:12] if args.selector_file else "heuristic"
        config = PipelineConfig(
            retrieval_top_k=args.retrieval_top_k,
            max_direct_candidates=args.max_direct_candidates,
            max_derived_candidates=args.max_derived_candidates,
            risk_threshold=risk_threshold,
            seed=args.seed,
            dataset_split=args.split,
            model_name=args.model_name,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            methods=(args.method,),
            selector_id=selector_id,
        )
        generator = HuggingFaceGenerator(
            args.model_name,
            device=args.device,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )

        def predictor(example):
            return run_example(example, generator=generator, config=config, selector=selector)[0]

        for example in examples:
            rows.extend(audit_counterfactual_predictions(example, predictor))
        manifest.update(
            {
                "method": args.method,
                "model_name": args.model_name,
                "model_revision": getattr(generator.model.config, "_commit_hash", None),
                "resolved_device": generator.device,
                "configuration_id": config.configuration_id,
                "risk_threshold": risk_threshold,
                "dependencies": _package_versions(),
                "hardware": _hardware_info(),
            }
        )
    else:
        for example in examples:
            rows.extend(audit_counterfactuals(example))
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    if frame.empty:
        print("No eligible counterfactual cases were constructed.")
    else:
        print(frame.groupby("perturbation")["passed"].agg(["count", "mean"]).to_string())
    print(f"Saved: {args.output}")
    return 0


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna("").astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def command_validate(args) -> int:
    predictions = pd.read_csv(args.predictions)
    required = {
        "example_id",
        "method",
        "question",
        "reference_answer",
        "model_response",
        "valid_numeric_response",
        "abstained",
        "invalid_response",
        "numeric_exact_match",
        "candidate_recall",
        "proof_valid",
        "unsupported_complete_answer",
        "proof_certificate",
        "candidate_count",
        "configuration_id",
        "model_name",
        "verification_seconds",
        "total_method_seconds",
        "input_token_count",
        "input_truncated",
    }
    errors: list[str] = []
    missing = sorted(required - set(predictions.columns))
    if missing:
        errors.append(f"Missing columns: {missing}")
    forbidden = {"gold_answer", "generated_answer", "synthetic_id", "feature_12"}
    present_forbidden = sorted(forbidden & set(predictions.columns))
    if present_forbidden:
        errors.append(f"Submission-facing output contains legacy columns: {present_forbidden}")
    if "example_id" in predictions and predictions["example_id"].astype(str).str.contains("synthetic", case=False).any():
        errors.append("Synthetic example IDs are present in empirical outputs.")
    if "method" in predictions and "example_id" in predictions:
        counts = predictions.groupby("method")["example_id"].nunique()
        if counts.nunique() != 1:
            errors.append(f"Methods have unequal example counts: {counts.to_dict()}")
        duplicated = predictions.duplicated(["method", "example_id"]).sum()
        if duplicated:
            errors.append(f"Duplicate method/example pairs: {duplicated}")
    else:
        counts = pd.Series(dtype=int)

    if not missing:
        valid = _bool_series(predictions["valid_numeric_response"])
        proof_valid = _bool_series(predictions["proof_valid"])
        unsupported = _bool_series(predictions["unsupported_complete_answer"])
        hard = predictions["method"].isin(
            [
                "direct_proof_lock",
                "derivation_proof_lock",
                "risk_controlled_proof_lock",
            ]
        )
        abstained = _bool_series(predictions["abstained"])
        invalid = _bool_series(predictions["invalid_response"])
        if (hard & unsupported).any():
            errors.append("A hard proof-lock method emitted an unsupported complete answer.")
        if (hard & valid & ~proof_valid).any():
            errors.append("A hard proof-lock numeric response lacks a valid proof certificate.")
        if (hard & invalid).any():
            errors.append("A hard proof-lock method emitted an invalid or fragmentary response.")
        if (hard & ~proof_valid & ~abstained).any():
            errors.append("A hard proof-lock response was neither certified nor an explicit abstention.")
        posthoc = predictions["method"].eq("posthoc_verifier")
        if (posthoc & valid & ~proof_valid).any():
            errors.append("The post-hoc verifier retained an uncertified numeric response.")
        if predictions["candidate_count"].max() > args.maximum_candidates + 1:
            errors.append("Candidate lattice exceeded the configured bound.")
        if predictions["model_response"].isna().any():
            errors.append("Blank model responses are present.")
        if predictions["configuration_id"].nunique() != 1:
            errors.append("More than one configuration ID appears in the same result file.")
        if not args.allow_engineering and predictions["model_name"].astype(str).str.contains(
            "engineering", case=False
        ).any():
            errors.append("Engineering-check outputs cannot be validated as empirical results.")

    if errors:
        print("VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print("VALIDATION PASSED")
    print(f"Rows: {len(predictions)}")
    print(f"Methods: {counts.to_dict()}")
    print(f"Configuration ID: {predictions['configuration_id'].iloc[0]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="numguard-fin")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download official FinQA files.")
    download.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    download.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    download.set_defaults(func=command_download)

    fit_selector = subparsers.add_parser(
        "fit-selector", help="Fit the lightweight candidate correctness selector on FinQA train."
    )
    # The selector must be trained on train; dev remains calibration-only.
    fit_selector.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    fit_selector.add_argument("--path", type=Path)
    fit_selector.add_argument("--split", choices=["train"], default="train")
    fit_selector.add_argument("--limit", type=int)
    fit_selector.add_argument("--seed", type=int, default=42)
    fit_selector.add_argument("--shuffle", action="store_true")
    fit_selector.add_argument("--numeric-only", action=argparse.BooleanOptionalAction, default=True)
    fit_selector.add_argument("--retrieval-top-k", type=int, default=48)
    fit_selector.add_argument("--max-direct-candidates", type=int, default=48)
    fit_selector.add_argument("--max-derived-candidates", type=int, default=192)
    fit_selector.add_argument("--folds", type=int, default=5)
    fit_selector.add_argument("--output", type=Path, default=Path("models/candidate_selector.json"))
    fit_selector.add_argument("--rows-output", type=Path, default=Path("results/selector_training_rows.csv"))
    fit_selector.set_defaults(func=command_fit_selector)

    def add_dataset_arguments(command):
        command.add_argument("--data-dir", type=Path, default=Path("data/raw"))
        command.add_argument("--path", type=Path)
        command.add_argument("--split", choices=["train", "dev", "test"], default="test")
        command.add_argument("--limit", type=int)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--shuffle", action="store_true")
        command.add_argument("--numeric-only", action=argparse.BooleanOptionalAction, default=True)

    inspect = subparsers.add_parser("inspect", help="Audit parsing, retrieval and candidate recall.")
    add_dataset_arguments(inspect)
    inspect.add_argument("--retrieval-top-k", type=int, default=48)
    inspect.add_argument("--max-direct-candidates", type=int, default=48)
    inspect.add_argument("--max-derived-candidates", type=int, default=192)
    inspect.add_argument("--selector-file", type=Path)
    inspect.add_argument("--minimum-candidate-recall", type=float, default=0.30)
    inspect.add_argument("--minimum-direct-recall", type=float, default=0.75)
    inspect.add_argument("--enforce-gates", action="store_true")
    inspect.add_argument("--output", type=Path, default=Path("results/dataset_audit.csv"))
    inspect.set_defaults(func=command_inspect)

    run = subparsers.add_parser("run", help="Run paired FinQA experiments.")
    add_dataset_arguments(run)
    run.add_argument("--model-name", default="google/flan-t5-base")
    run.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    run.add_argument("--engineering-check", action="store_true")
    run.add_argument("--methods", help="Comma-separated method names.")
    run.add_argument("--retrieval-top-k", type=int, default=48)
    run.add_argument("--max-direct-candidates", type=int, default=48)
    run.add_argument("--max-derived-candidates", type=int, default=192)
    run.add_argument("--risk-threshold", type=float, default=0.55)
    run.add_argument("--risk-threshold-file", type=Path)
    run.add_argument("--selector-file", type=Path)
    run.add_argument("--max-input-tokens", type=int, default=512)
    run.add_argument("--max-new-tokens", type=int, default=24)
    run.add_argument("--num-beams", type=int, default=4)
    run.add_argument("--output-dir", type=Path, default=Path("results/current_run"))
    run.add_argument("--figures-dir", type=Path, default=Path("figures/current_run"))
    run.add_argument(
        "--resume",
        action="store_true",
        help="Resume a matching run from the per-example checkpoint in the output directory.",
    )
    run.set_defaults(func=command_run)

    calibrate = subparsers.add_parser("calibrate", help="Select a dev-set risk threshold.")
    calibrate.add_argument("--predictions", type=Path, required=True)
    calibrate.add_argument("--method", default="selector_guided_proof_lock")
    calibrate.add_argument("--target-risk", type=float, default=0.20)
    calibrate.add_argument("--confidence", type=float, default=0.95)
    calibrate.add_argument("--risk-type", choices=["semantic", "provenance"], default="semantic")
    calibrate.add_argument("--minimum-coverage", type=float, default=0.05)
    calibrate.add_argument("--minimum-accepted", type=int, default=30)
    calibrate.add_argument("--output", type=Path, default=Path("results/calibration.json"))
    calibrate.set_defaults(func=command_calibrate)

    counterfactual = subparsers.add_parser("counterfactual", help="Run structured provenance perturbation audit.")
    add_dataset_arguments(counterfactual)
    counterfactual.add_argument("--model-level", action="store_true")
    counterfactual.add_argument(
        "--method",
        choices=["direct_proof_lock", "derivation_proof_lock", "selector_guided_proof_lock", "risk_controlled_proof_lock"],
        default="derivation_proof_lock",
    )
    counterfactual.add_argument("--model-name", default="google/flan-t5-base")
    counterfactual.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    counterfactual.add_argument("--retrieval-top-k", type=int, default=48)
    counterfactual.add_argument("--max-direct-candidates", type=int, default=48)
    counterfactual.add_argument("--max-derived-candidates", type=int, default=192)
    counterfactual.add_argument("--risk-threshold", type=float, default=0.55)
    counterfactual.add_argument("--risk-threshold-file", type=Path)
    counterfactual.add_argument("--selector-file", type=Path)
    counterfactual.add_argument("--max-input-tokens", type=int, default=512)
    counterfactual.add_argument("--max-new-tokens", type=int, default=24)
    counterfactual.add_argument("--num-beams", type=int, default=4)
    counterfactual.add_argument("--output", type=Path, default=Path("results/counterfactual_audit.csv"))
    counterfactual.set_defaults(func=command_counterfactual)

    validate = subparsers.add_parser("validate", help="Validate result integrity.")
    validate.add_argument("--predictions", type=Path, required=True)
    validate.add_argument("--maximum-candidates", type=int, default=160)
    validate.add_argument("--allow-engineering", action="store_true")
    validate.set_defaults(func=command_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
