from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Iterable, Optional

from .candidates import ABSTAIN_RESPONSE, CandidateLattice, build_candidate_lattice
from .evidence import build_evidence_ledger
from .intent import parse_question_intent
from .model import (
    GenerationResult,
    Generator,
    HeuristicSelector,
    build_baseline_prompt,
    build_evidence_prompt,
)
from .numeric import numeric_equal, parse_single_number
from .proof import is_abstention, is_valid_numeric_response, verify_response
from .retrieval import retrieve_evidence
from .schemas import FinQAExample, PredictionRecord
from .selector import LinearCandidateSelector


DEFAULT_METHODS = (
    "baseline",
    "provenance_prompt",
    "candidate_menu_prompt",
    "posthoc_verifier",
    "fragment_token_lock_ablation",
    "soft_prefix_bias_1",
    "soft_prefix_bias_3",
    "soft_prefix_bias_5",
    "soft_prefix_bias_10",
    "direct_proof_lock",
    "derivation_proof_lock",
    "selector_guided_proof_lock",
    "risk_controlled_proof_lock",
)


@dataclass(frozen=True)
class PipelineConfig:
    retrieval_top_k: int = 48
    max_direct_candidates: int = 48
    max_derived_candidates: int = 192
    risk_threshold: float = 0.55
    seed: int = 42
    dataset_split: str = "test"
    model_name: str = "google/flan-t5-base"
    max_input_tokens: int = 512
    max_new_tokens: int = 24
    num_beams: int = 4
    methods: tuple[str, ...] = DEFAULT_METHODS
    selector_id: str = "safe-blend"

    @property
    def configuration_id(self) -> str:
        payload = json.dumps(
            {
                "retrieval_top_k": self.retrieval_top_k,
                "max_direct_candidates": self.max_direct_candidates,
                "max_derived_candidates": self.max_derived_candidates,
                "risk_threshold": self.risk_threshold,
                "seed": self.seed,
                "dataset_split": self.dataset_split,
                "model_name": self.model_name,
                "max_input_tokens": self.max_input_tokens,
                "max_new_tokens": self.max_new_tokens,
                "num_beams": self.num_beams,
                "methods": self.methods,
                "selector_id": self.selector_id,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _numeric_match(
    response: str,
    reference: Optional[str],
    *,
    expected_answer_type: Optional[str] = None,
) -> Optional[bool]:
    if reference is None:
        return None
    generated = parse_single_number(response)
    expected = parse_single_number(reference)
    if generated is None or expected is None:
        return False
    strict = numeric_equal(generated, expected, require_percent_compatibility=True)
    marker_omission = (
        expected_answer_type == "percentage"
        and numeric_equal(generated, expected, require_percent_compatibility=False)
    )
    return strict or marker_omission


def _candidate_confidence(lattice: CandidateLattice, response: str) -> Optional[float]:
    candidate = lattice.find_by_answer(response)
    return None if candidate is None else candidate.confidence


def _predicted_program(lattice: CandidateLattice, response: str) -> Optional[str]:
    candidate = lattice.find_by_answer(response)
    if candidate is None or candidate.proof.proof_type == "abstain":
        return None
    if candidate.proof.proof_type == "direct":
        return f"lookup({candidate.evidence_ids[0]})"
    return f"{candidate.proof.operation}({','.join(candidate.evidence_ids)})"


def _normalise_predicted_operation(operation: Optional[str]) -> Optional[str]:
    mapping = {
        "lookup": "direct_lookup",
        "subtraction": "difference",
        "addition": "sum",
        "division": "ratio",
        "multiplication": "product",
        "table_sum": "sum",
        "table_average": "average",
        "table_min": "table_min",
        "table_max": "table_max",
        "percentage_of_total": "margin",
        "aggregate_difference": "difference",
        "indexed_return": "indexed_return",
    }
    return mapping.get(operation or "", operation)


def infer_reference_operation(
    reference_program: Optional[str],
    *,
    question_operation: Optional[str] = None,
) -> Optional[str]:
    """Map a FinQA program to a coarse operation family.

    The diagnostic is intentionally coarser than official exact program accuracy. It
    records whether the emitted proof uses the same operation family as the annotated
    FinQA program, without claiming token-level program equivalence.
    """

    if not reference_program:
        return None
    operations = re.findall(
        r"\b(add|subtract|multiply|divide|average|table_average|table_lookup|text_lookup|exp|greater)\s*\(",
        reference_program.lower(),
    )
    if not operations:
        return "direct_lookup"
    operation_set = set(operations)
    if operation_set & {"table_lookup", "text_lookup"}:
        return "direct_lookup"
    if question_operation == "percentage_change" and {"subtract", "divide"}.issubset(
        operation_set
    ):
        return "percentage_change"
    if question_operation == "margin" and "divide" in operation_set:
        return "margin"
    if "average" in operation_set or "table_average" in operation_set:
        return "average"
    if {"subtract", "divide", "multiply"}.issubset(operation_set):
        return "percentage_change"
    if {"divide", "multiply"}.issubset(operation_set) and question_operation in {
        "margin",
        "percentage_change",
    }:
        return question_operation
    if "add" in operation_set and "divide" in operation_set:
        return "average"
    if operations == ["add"] or operation_set == {"add"}:
        return "sum"
    if operations == ["subtract"] or operation_set == {"subtract"}:
        return "difference"
    if operations == ["divide"] or operation_set == {"divide"}:
        return "ratio"
    if operations == ["multiply"] or operation_set == {"multiply"}:
        return "product"
    return "+".join(operations)


def _make_record(
    *,
    example: FinQAExample,
    method: str,
    result: GenerationResult,
    lattice: CandidateLattice,
    evidence,
    intent,
    config: PipelineConfig,
    model_name: str,
) -> PredictionRecord:
    response = result.response.strip()
    verification_start = time.perf_counter()
    certificate = verify_response(response, lattice, evidence)
    verification_seconds = result.control_seconds + (time.perf_counter() - verification_start)
    valid_numeric = is_valid_numeric_response(response)
    abstained = is_abstention(response)
    invalid = not valid_numeric and not abstained
    selected = lattice.find_by_answer(response)
    numeric_match = _numeric_match(
        response,
        example.reference_answer,
        expected_answer_type=intent.expected_answer_type,
    )
    candidate_recall = lattice.contains_reference(example.reference_answer)
    predicted_program = _predicted_program(lattice, response)
    reference_operation = infer_reference_operation(
        example.reference_program,
        question_operation=intent.operation,
    )
    predicted_operation = _normalise_predicted_operation(
        None if selected is None else selected.proof.operation
    )
    operation_match = (
        None
        if reference_operation is None or predicted_operation is None
        else reference_operation == predicted_operation
    )
    selected_confidence = None if selected is None else selected.confidence
    selected_margin = lattice.confidence_margin(response)
    provenance_risk = (1.0 if valid_numeric and certificate.decision == "rejected" else 0.0)
    semantic_risk = None if selected_confidence is None else 1.0 - selected_confidence
    return PredictionRecord(
        example_id=example.example_id,
        method=method,
        question=example.question,
        reference_answer=example.reference_answer,
        model_response=response,
        valid_numeric_response=valid_numeric,
        abstained=abstained,
        invalid_response=invalid,
        numeric_exact_match=numeric_match,
        candidate_recall=candidate_recall,
        proof_valid=certificate.decision == "accepted",
        unsupported_complete_answer=valid_numeric and certificate.decision == "rejected",
        selected_candidate_id=None if selected is None else selected.candidate_id,
        selected_candidate_confidence=selected_confidence,
        selected_candidate_margin=selected_margin,
        provenance_risk_score=provenance_risk,
        semantic_risk_score=semantic_risk,
        proof_certificate=certificate.to_dict(),
        candidate_count=len(lattice.candidates),
        direct_candidate_count=lattice.direct_count,
        derived_candidate_count=lattice.derived_count,
        retrieval_count=lattice.retrieval_count,
        inference_seconds=result.inference_seconds,
        verification_seconds=verification_seconds,
        total_method_seconds=result.inference_seconds + verification_seconds,
        input_token_count=result.input_token_count,
        input_truncated=result.input_truncated,
        model_sequence_score=result.sequence_score,
        model_name=model_name,
        dataset_split=config.dataset_split,
        seed=config.seed,
        configuration_id=config.configuration_id,
        reference_program=example.reference_program,
        predicted_program=predicted_program,
        reference_operation=reference_operation,
        predicted_operation=predicted_operation,
        proof_operation_match=operation_match,
        notes=None,
    )


def run_example(
    example: FinQAExample,
    *,
    generator: Optional[Generator],
    config: PipelineConfig,
    selector: Optional[LinearCandidateSelector] = None,
) -> list[PredictionRecord]:
    intent = parse_question_intent(example.question)
    evidence = build_evidence_ledger(example)
    retrieved = retrieve_evidence(
        evidence,
        example.question,
        intent,
        top_k=config.retrieval_top_k,
    )
    direct_lattice = build_candidate_lattice(
        retrieved,
        intent,
        include_derived=False,
        max_direct=config.max_direct_candidates,
        max_derived=0,
        candidate_scorer=None if selector is None else selector.score,
    )
    full_lattice = build_candidate_lattice(
        retrieved,
        intent,
        include_derived=True,
        max_direct=config.max_direct_candidates,
        max_derived=config.max_derived_candidates,
        candidate_scorer=None if selector is None else selector.score,
    )

    def generator_for(lattice: CandidateLattice) -> Generator:
        return generator if generator is not None else HeuristicSelector(lattice, threshold=0.55)

    model_name = generator.model_name if generator is not None else HeuristicSelector.model_name
    baseline_prompt = build_baseline_prompt(example.question, retrieved)
    provenance_prompt = build_evidence_prompt(example.question, retrieved)
    candidate_prompt = build_evidence_prompt(
        example.question,
        retrieved,
        lattice=full_lattice,
        include_candidates=True,
    )
    direct_prompt = build_evidence_prompt(
        example.question,
        retrieved,
        lattice=direct_lattice,
        include_candidates=True,
    )

    records: list[PredictionRecord] = []
    baseline_result: Optional[GenerationResult] = None
    derivation_result: Optional[GenerationResult] = None
    ranked_result: Optional[GenerationResult] = None

    for method in config.methods:
        lattice = full_lattice
        active_generator = generator_for(full_lattice)
        if method == "baseline":
            baseline_result = active_generator.free_generate(baseline_prompt)
            result = baseline_result
        elif method == "provenance_prompt":
            result = active_generator.free_generate(provenance_prompt)
        elif method == "candidate_menu_prompt":
            result = active_generator.free_generate(candidate_prompt)
        elif method == "posthoc_verifier":
            if baseline_result is None:
                baseline_result = active_generator.free_generate(baseline_prompt)
            decision_start = time.perf_counter()
            certificate = verify_response(baseline_result.response, full_lattice, evidence)
            decision_seconds = time.perf_counter() - decision_start
            response = baseline_result.response if certificate.decision == "accepted" else ABSTAIN_RESPONSE
            result = GenerationResult(
                response,
                baseline_result.sequence_score,
                baseline_result.inference_seconds,
                baseline_result.input_token_count,
                baseline_result.input_truncated,
                decision_seconds,
            )
        elif method == "fragment_token_lock_ablation":
            result = active_generator.fragment_generate(candidate_prompt, full_lattice.answer_strings)
        elif method.startswith("soft_prefix_bias_"):
            bias = float(method.rsplit("_", 1)[1])
            result = active_generator.soft_generate(candidate_prompt, full_lattice.answer_strings, bias)
        elif method == "direct_proof_lock":
            lattice = direct_lattice
            result = generator_for(direct_lattice).constrained_generate(
                direct_prompt, direct_lattice.answer_strings
            )
        elif method == "derivation_proof_lock":
            derivation_result = active_generator.constrained_generate(
                candidate_prompt, full_lattice.answer_strings
            )
            result = derivation_result
        elif method == "selector_guided_proof_lock":
            ranked_result = active_generator.ranked_constrained_generate(
                candidate_prompt, full_lattice.candidates, prior_strength=2.0
            )
            result = ranked_result
        elif method == "risk_controlled_proof_lock":
            if ranked_result is None:
                ranked_result = active_generator.ranked_constrained_generate(
                    candidate_prompt, full_lattice.candidates, prior_strength=2.0
                )
            decision_start = time.perf_counter()
            confidence = _candidate_confidence(full_lattice, ranked_result.response)
            response = (
                ranked_result.response
                if confidence is not None and confidence >= config.risk_threshold
                else ABSTAIN_RESPONSE
            )
            decision_seconds = time.perf_counter() - decision_start
            result = GenerationResult(
                response,
                ranked_result.sequence_score,
                ranked_result.inference_seconds,
                ranked_result.input_token_count,
                ranked_result.input_truncated,
                decision_seconds,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        records.append(
            _make_record(
                example=example,
                method=method,
                result=result,
                lattice=lattice,
                evidence=evidence,
                intent=intent,
                config=config,
                model_name=model_name,
            )
        )
    return records


def run_examples(
    examples: Iterable[FinQAExample],
    *,
    generator: Optional[Generator],
    config: PipelineConfig,
    selector: Optional[LinearCandidateSelector] = None,
) -> list[PredictionRecord]:
    records: list[PredictionRecord] = []
    for example in examples:
        records.extend(run_example(example, generator=generator, config=config, selector=selector))
    return records
