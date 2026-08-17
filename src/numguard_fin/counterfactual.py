from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional

from .candidates import AnswerCandidate, build_candidate_lattice
from .evidence import build_evidence_ledger, evidence_by_id
from .intent import parse_question_intent
from .retrieval import retrieve_evidence
from .schemas import EvidenceItem, FinQAExample, PredictionRecord


@dataclass(frozen=True)
class CounterfactualCase:
    perturbation: str
    example: FinQAExample
    expected_relation: str
    changed_locations: tuple[str, ...]
    notes: str


SCALE_SWAP = {
    "thousand": "million",
    "thousands": "millions",
    "million": "thousand",
    "millions": "thousands",
    "billion": "million",
    "billions": "millions",
}


def _analysis(example: FinQAExample):
    intent = parse_question_intent(example.question)
    ledger = build_evidence_ledger(example)
    retrieved = retrieve_evidence(ledger, example.question, intent, top_k=32)
    lattice = build_candidate_lattice(retrieved, intent, include_derived=True)
    numeric = sorted(
        lattice.numeric_candidates(), key=lambda candidate: (-candidate.confidence, candidate.candidate_id)
    )
    top = numeric[0] if numeric else None
    return intent, ledger, retrieved, top


def _operand_items(example: FinQAExample) -> tuple[Optional[AnswerCandidate], list[EvidenceItem]]:
    _, ledger, _, top = _analysis(example)
    if top is None:
        return None, []
    mapping = evidence_by_id(ledger)
    return top, [mapping[evidence_id] for evidence_id in top.evidence_ids if evidence_id in mapping]


def irrelevant_number_injection(example: FinQAExample) -> CounterfactualCase:
    table = [list(row) for row in example.table]
    width = len(table[0]) if table else 2
    injected = ["Unrelated diagnostic value", "987654321"] + [""] * max(0, width - 2)
    table.append(injected[:width])
    return CounterfactualCase(
        perturbation="irrelevant_number_injection",
        example=replace(
            example,
            example_id=f"{example.example_id}::cf_irrelevant",
            table=tuple(tuple(row) for row in table),
        ),
        expected_relation="answer_and_proof_invariant",
        changed_locations=(f"table:r{len(table)-1}:c1",),
        notes="Adds a plausible but question-irrelevant number in a new row.",
    )


def year_value_swap(example: FinQAExample) -> Optional[CounterfactualCase]:
    _, operands = _operand_items(example)
    target = next(
        (
            item
            for item in operands
            if item.source_type == "table" and item.row_index is not None and item.period
        ),
        None,
    )
    if target is None:
        return None
    _, ledger, _, _ = _analysis(example)
    alternatives = [
        item
        for item in ledger
        if item.source_type == "table"
        and item.row_index == target.row_index
        and item.column_index != target.column_index
        and item.period
    ]
    if not alternatives:
        return None
    alternative = sorted(alternatives, key=lambda item: item.evidence_id)[0]
    table = [list(row) for row in example.table]
    row = target.row_index
    left_col, right_col = target.column_index, alternative.column_index
    table[row][left_col], table[row][right_col] = table[row][right_col], table[row][left_col]
    return CounterfactualCase(
        perturbation="year_value_swap",
        example=replace(
            example,
            example_id=f"{example.example_id}::cf_year",
            table=tuple(tuple(row_values) for row_values in table),
        ),
        expected_relation="answer_or_proof_sensitive",
        changed_locations=(f"table:r{row}:c{left_col}", f"table:r{row}:c{right_col}"),
        notes="Swaps the selected value with another period in the same metric row.",
    )


def metric_value_swap(example: FinQAExample) -> Optional[CounterfactualCase]:
    _, operands = _operand_items(example)
    target = next(
        (
            item
            for item in operands
            if item.source_type == "table" and item.row_index is not None
        ),
        None,
    )
    if target is None:
        return None
    _, ledger, _, _ = _analysis(example)
    distractors = [
        item
        for item in ledger
        if item.source_type == "table"
        and item.row_index != target.row_index
        and item.column_index == target.column_index
        and item.metric_text != target.metric_text
    ]
    if not distractors:
        return None
    distractor = sorted(distractors, key=lambda item: item.evidence_id)[0]
    table = [list(row) for row in example.table]
    column = target.column_index
    row_a, row_b = target.row_index, distractor.row_index
    table[row_a][column], table[row_b][column] = table[row_b][column], table[row_a][column]
    return CounterfactualCase(
        perturbation="metric_value_swap",
        example=replace(
            example,
            example_id=f"{example.example_id}::cf_metric",
            table=tuple(tuple(row_values) for row_values in table),
        ),
        expected_relation="answer_or_proof_sensitive",
        changed_locations=(f"table:r{row_a}:c{column}", f"table:r{row_b}:c{column}"),
        notes="Swaps the selected metric value with a different row in the same period.",
    )


def _mask_text_operand(example: FinQAExample, target: EvidenceItem) -> Optional[CounterfactualCase]:
    pre = list(example.pre_text)
    post = list(example.post_text)
    for label, paragraphs in (("pre_text", pre), ("post_text", post)):
        for index, paragraph in enumerate(paragraphs):
            if target.raw_number in paragraph and target.text in paragraph:
                paragraphs[index] = paragraph.replace(target.raw_number, "—", 1)
                return CounterfactualCase(
                    perturbation="support_mask",
                    example=replace(
                        example,
                        example_id=f"{example.example_id}::cf_mask",
                        pre_text=tuple(pre),
                        post_text=tuple(post),
                    ),
                    expected_relation="answer_changes_or_abstains",
                    changed_locations=(f"{label}:{index}",),
                    notes="Removes a selected narrative value without changing the question.",
                )
    return None


def support_mask(example: FinQAExample) -> Optional[CounterfactualCase]:
    _, operands = _operand_items(example)
    if not operands:
        return None
    target = operands[0]
    if target.source_type == "text":
        return _mask_text_operand(example, target)
    if target.row_index is None or target.column_index is None:
        return None
    table = [list(row) for row in example.table]
    table[target.row_index][target.column_index] = "—"
    return CounterfactualCase(
        perturbation="support_mask",
        example=replace(
            example,
            example_id=f"{example.example_id}::cf_mask",
            table=tuple(tuple(row_values) for row_values in table),
        ),
        expected_relation="answer_changes_or_abstains",
        changed_locations=(f"table:r{target.row_index}:c{target.column_index}",),
        notes="Removes a selected supporting value without changing the question.",
    )


def scale_flip(example: FinQAExample) -> Optional[CounterfactualCase]:
    _, operands = _operand_items(example)
    if not operands or not any(item.scale for item in operands):
        return None
    pattern = re.compile(r"\b(thousands?|millions?|billions?)\b", flags=re.IGNORECASE)

    def flip(text: str) -> tuple[str, bool]:
        match = pattern.search(text)
        if not match:
            return text, False
        original = match.group(1)
        replacement = SCALE_SWAP[original.lower()]
        if original[0].isupper():
            replacement = replacement.capitalize()
        return text[: match.start()] + replacement + text[match.end() :], True

    table = [list(row) for row in example.table]
    for row_index, row in enumerate(table):
        for column_index, cell in enumerate(row):
            updated, changed = flip(cell)
            if changed:
                row[column_index] = updated
                return CounterfactualCase(
                    perturbation="scale_flip",
                    example=replace(
                        example,
                        example_id=f"{example.example_id}::cf_scale",
                        table=tuple(tuple(row_values) for row_values in table),
                    ),
                    expected_relation="scale_certificate_sensitive",
                    changed_locations=(f"table:r{row_index}:c{column_index}",),
                    notes="Changes the declared unit scale while preserving displayed digits.",
                )

    pre, post = list(example.pre_text), list(example.post_text)
    for label, paragraphs in (("pre_text", pre), ("post_text", post)):
        for index, paragraph in enumerate(paragraphs):
            updated, changed = flip(paragraph)
            if changed:
                paragraphs[index] = updated
                return CounterfactualCase(
                    perturbation="scale_flip",
                    example=replace(
                        example,
                        example_id=f"{example.example_id}::cf_scale",
                        pre_text=tuple(pre),
                        post_text=tuple(post),
                    ),
                    expected_relation="scale_certificate_sensitive",
                    changed_locations=(f"{label}:{index}",),
                    notes="Changes the declared unit scale while preserving displayed digits.",
                )
    return None


def generate_counterfactuals(example: FinQAExample) -> list[CounterfactualCase]:
    cases: list[Optional[CounterfactualCase]] = [
        irrelevant_number_injection(example),
        year_value_swap(example),
        metric_value_swap(example),
        support_mask(example),
        scale_flip(example),
    ]
    return [case for case in cases if case is not None]


def _top_candidate_state(example: FinQAExample) -> dict[str, Any]:
    _, ledger, _, top = _analysis(example)
    if top is None:
        return {"answer": None, "proof": None, "scale_signature": ()}
    mapping = evidence_by_id(ledger)
    operand_signature = tuple(
        (
            evidence_id,
            mapping[evidence_id].canonical,
            str(mapping[evidence_id].base_value),
            mapping[evidence_id].scale,
            mapping[evidence_id].period,
            mapping[evidence_id].metric_text,
        )
        for evidence_id in top.evidence_ids
        if evidence_id in mapping
    )
    return {
        "answer": top.answer,
        "proof": (top.proof.operation, operand_signature),
        "scale_signature": tuple(item[3] for item in operand_signature),
    }


def audit_counterfactuals(example: FinQAExample) -> list[dict[str, Any]]:
    original = _top_candidate_state(example)
    rows: list[dict[str, Any]] = []
    for case in generate_counterfactuals(example):
        changed = _top_candidate_state(case.example)
        if case.expected_relation == "answer_and_proof_invariant":
            passed = changed["answer"] == original["answer"] and changed["proof"] == original["proof"]
        elif case.expected_relation == "answer_or_proof_sensitive":
            passed = changed["answer"] != original["answer"] or changed["proof"] != original["proof"]
        elif case.expected_relation == "answer_changes_or_abstains":
            passed = changed["answer"] is None or changed["answer"] != original["answer"]
        elif case.expected_relation == "scale_certificate_sensitive":
            passed = changed["scale_signature"] != original["scale_signature"]
        else:
            passed = False
        rows.append(
            {
                "example_id": example.example_id,
                "audit_layer": "candidate_mechanism",
                "perturbation": case.perturbation,
                "expected_relation": case.expected_relation,
                "passed": passed,
                "original_answer": original["answer"],
                "perturbed_answer": changed["answer"],
                "original_proof": repr(original["proof"]),
                "perturbed_proof": repr(changed["proof"]),
                "changed_locations": "|".join(case.changed_locations),
                "notes": case.notes,
            }
        )
    return rows


def _certificate_signature(record: PredictionRecord) -> tuple[Any, ...]:
    certificate = record.proof_certificate or {}
    claims = certificate.get("evidence_claims") or []
    claim_signature = tuple(
        (
            claim.get("evidence_id"),
            claim.get("canonical"),
            claim.get("base_value"),
            claim.get("scale"),
            claim.get("period"),
            claim.get("metric_text"),
            claim.get("row_index"),
            claim.get("column_index"),
        )
        for claim in claims
    )
    return (
        certificate.get("decision"),
        certificate.get("proof_type"),
        certificate.get("operation"),
        certificate.get("expression"),
        claim_signature,
    )


def _prediction_state(record: PredictionRecord) -> dict[str, Any]:
    certificate = record.proof_certificate or {}
    claims = certificate.get("evidence_claims") or []
    return {
        "response": record.model_response,
        "abstained": record.abstained,
        "proof": _certificate_signature(record),
        "scale_signature": tuple(claim.get("scale") for claim in claims),
        "confidence": record.selected_candidate_confidence,
    }


def audit_counterfactual_predictions(
    example: FinQAExample,
    predictor: Callable[[FinQAExample], PredictionRecord],
) -> list[dict[str, Any]]:
    """Audit actual proof-lock predictions under controlled evidence perturbations."""

    original_record = predictor(example)
    original = _prediction_state(original_record)
    rows: list[dict[str, Any]] = []
    for case in generate_counterfactuals(example):
        perturbed_record = predictor(case.example)
        changed = _prediction_state(perturbed_record)
        if case.expected_relation == "answer_and_proof_invariant":
            passed = (
                changed["response"] == original["response"]
                and changed["proof"] == original["proof"]
            )
        elif case.expected_relation == "answer_or_proof_sensitive":
            passed = (
                changed["response"] != original["response"]
                or changed["proof"] != original["proof"]
            )
        elif case.expected_relation == "answer_changes_or_abstains":
            passed = bool(changed["abstained"]) or changed["response"] != original["response"]
        elif case.expected_relation == "scale_certificate_sensitive":
            passed = changed["scale_signature"] != original["scale_signature"]
        else:
            passed = False
        rows.append(
            {
                "example_id": example.example_id,
                "audit_layer": "model_output",
                "method": original_record.method,
                "perturbation": case.perturbation,
                "expected_relation": case.expected_relation,
                "passed": passed,
                "original_response": original["response"],
                "perturbed_response": changed["response"],
                "original_proof": repr(original["proof"]),
                "perturbed_proof": repr(changed["proof"]),
                "original_confidence": original["confidence"],
                "perturbed_confidence": changed["confidence"],
                "changed_locations": "|".join(case.changed_locations),
                "notes": case.notes,
            }
        )
    return rows
