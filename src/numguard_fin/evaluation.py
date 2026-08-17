from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .schemas import PredictionRecord


def _as_dict(record: PredictionRecord | dict[str, Any]) -> dict[str, Any]:
    return record.to_dict() if isinstance(record, PredictionRecord) else record


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    repetitions: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(repetitions, array.size))
    means = array[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(array.mean()),
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
    )


def risk_coverage_curve(rows: list[dict[str, Any]]) -> pd.DataFrame:
    scored = [
        row
        for row in rows
        if bool(row.get("valid_numeric_response"))
        and row.get("selected_candidate_confidence") is not None
    ]
    if not scored:
        return pd.DataFrame(columns=["threshold", "coverage", "risk", "accuracy", "accepted"])
    thresholds = sorted(
        {0.0, 1.0, *(float(row["selected_candidate_confidence"]) for row in scored)}
    )
    total = len(rows)
    points = []
    for threshold in thresholds:
        accepted = [
            row
            for row in scored
            if float(row["selected_candidate_confidence"]) >= threshold
        ]
        coverage = len(accepted) / total if total else 0.0
        accuracy = (
            mean(float(bool(row.get("numeric_exact_match"))) for row in accepted)
            if accepted
            else math.nan
        )
        points.append(
            {
                "threshold": threshold,
                "coverage": coverage,
                "risk": 1.0 - accuracy if accepted else math.nan,
                "accuracy": accuracy,
                "accepted": len(accepted),
            }
        )
    return pd.DataFrame(points).sort_values(
        ["coverage", "threshold"], ascending=[False, True]
    )


def area_under_risk_coverage(curve: pd.DataFrame) -> float:
    valid = curve.dropna(subset=["coverage", "risk"]).sort_values("coverage")
    if len(valid) < 2:
        return math.nan
    return float(np.trapezoid(valid["risk"].to_numpy(), valid["coverage"].to_numpy()))


def _rate_ci(values: list[float], seed: int) -> tuple[float, float, float]:
    return bootstrap_mean_ci(values, seed=seed)


def summarise_records(
    records: Iterable[PredictionRecord | dict[str, Any]], seed: int = 42
) -> pd.DataFrame:
    rows = [_as_dict(record) for record in records]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)

    summaries: list[dict[str, Any]] = []
    for method, method_rows in grouped.items():
        n = len(method_rows)
        accuracy_values = [
            float(bool(row.get("numeric_exact_match")))
            for row in method_rows
            if row.get("numeric_exact_match") is not None
        ]
        accuracy, accuracy_lo, accuracy_hi = _rate_ci(accuracy_values, seed)

        valid_values = [float(bool(row.get("valid_numeric_response"))) for row in method_rows]
        valid_rate, valid_lo, valid_hi = _rate_ci(valid_values, seed)
        unsupported_values = [
            float(bool(row.get("unsupported_complete_answer"))) for row in method_rows
        ]
        unsupported_rate, unsupported_lo, unsupported_hi = _rate_ci(
            unsupported_values, seed
        )
        recall_values = [
            float(bool(row.get("candidate_recall")))
            for row in method_rows
            if row.get("candidate_recall") is not None
        ]
        candidate_recall, recall_lo, recall_hi = _rate_ci(recall_values, seed)
        operation_values = []
        for row in method_rows:
            value = row.get("proof_operation_match")
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            operation_values.append(float(bool(value)))
        operation_match, operation_lo, operation_hi = _rate_ci(operation_values, seed)

        valid_numeric = [row for row in method_rows if bool(row.get("valid_numeric_response"))]
        proof_valid_numeric = [row for row in valid_numeric if bool(row.get("proof_valid"))]
        unsupported = [
            row for row in method_rows if bool(row.get("unsupported_complete_answer"))
        ]
        recalled = [row for row in method_rows if row.get("candidate_recall") is True]
        impossible = [row for row in method_rows if row.get("candidate_recall") is False]
        abstained = [row for row in method_rows if bool(row.get("abstained"))]

        answered_accuracy = (
            mean(float(bool(row.get("numeric_exact_match"))) for row in valid_numeric)
            if valid_numeric
            else math.nan
        )
        selection_given_recall = (
            mean(float(bool(row.get("numeric_exact_match"))) for row in recalled)
            if recalled
            else math.nan
        )
        abstention_precision = (
            sum(row.get("candidate_recall") is False for row in abstained) / len(abstained)
            if abstained
            else math.nan
        )
        abstention_recall = (
            sum(bool(row.get("abstained")) for row in impossible) / len(impossible)
            if impossible
            else math.nan
        )
        curve = risk_coverage_curve(method_rows)
        inference_latencies = [
            float(row.get("inference_seconds") or 0.0) for row in method_rows
        ]
        verification_latencies = [
            float(row.get("verification_seconds") or 0.0) for row in method_rows
        ]
        total_latencies = [
            float(row.get("total_method_seconds") or 0.0) for row in method_rows
        ]

        summaries.append(
            {
                "method": method,
                "examples": n,
                "numeric_answer_accuracy": accuracy,
                "numeric_answer_accuracy_ci_low": accuracy_lo,
                "numeric_answer_accuracy_ci_high": accuracy_hi,
                "valid_numeric_response_rate": valid_rate,
                "valid_numeric_response_rate_ci_low": valid_lo,
                "valid_numeric_response_rate_ci_high": valid_hi,
                "non_abstention_rate": 1.0
                - mean(float(bool(row.get("abstained"))) for row in method_rows),
                "abstention_rate": mean(
                    float(bool(row.get("abstained"))) for row in method_rows
                ),
                "invalid_response_rate": mean(
                    float(bool(row.get("invalid_response"))) for row in method_rows
                ),
                "unsupported_complete_answer_rate": unsupported_rate,
                "unsupported_complete_answer_rate_ci_low": unsupported_lo,
                "unsupported_complete_answer_rate_ci_high": unsupported_hi,
                "unsupported_rate_given_numeric_response": (
                    len(unsupported) / len(valid_numeric) if valid_numeric else math.nan
                ),
                "proof_validity_rate_given_numeric_response": (
                    len(proof_valid_numeric) / len(valid_numeric)
                    if valid_numeric
                    else math.nan
                ),
                "candidate_recall": candidate_recall,
                "candidate_recall_ci_low": recall_lo,
                "candidate_recall_ci_high": recall_hi,
                "proof_operation_match_rate": operation_match,
                "proof_operation_match_rate_ci_low": operation_lo,
                "proof_operation_match_rate_ci_high": operation_hi,
                "selection_accuracy_given_candidate_recall": selection_given_recall,
                "accuracy_given_numeric_response": answered_accuracy,
                "risk_given_numeric_response": (
                    1.0 - answered_accuracy
                    if not math.isnan(answered_accuracy)
                    else math.nan
                ),
                "abstention_precision_for_unrecalled_answers": abstention_precision,
                "abstention_recall_for_unrecalled_answers": abstention_recall,
                "area_under_risk_coverage": area_under_risk_coverage(curve),
                "mean_candidate_count": mean(
                    float(row.get("candidate_count", 0)) for row in method_rows
                ),
                "mean_direct_candidate_count": mean(
                    float(row.get("direct_candidate_count", 0)) for row in method_rows
                ),
                "mean_derived_candidate_count": mean(
                    float(row.get("derived_candidate_count", 0)) for row in method_rows
                ),
                "mean_retrieval_count": mean(
                    float(row.get("retrieval_count", 0)) for row in method_rows
                ),
                "input_truncation_rate": mean(
                    float(bool(row.get("input_truncated"))) for row in method_rows
                ),
                "mean_input_token_count": mean(
                    float(row.get("input_token_count", 0)) for row in method_rows
                ),
                "mean_inference_seconds": (
                    mean(inference_latencies) if inference_latencies else math.nan
                ),
                "mean_verification_seconds": (
                    mean(verification_latencies) if verification_latencies else math.nan
                ),
                "mean_total_method_seconds": (
                    mean(total_latencies) if total_latencies else math.nan
                ),
                "p95_total_method_seconds": (
                    float(np.quantile(total_latencies, 0.95))
                    if total_latencies
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(summaries).sort_values("method").reset_index(drop=True)


def mcnemar_exact(
    records: Iterable[PredictionRecord | dict[str, Any]],
    method_a: str,
    method_b: str,
    outcome: str = "numeric_exact_match",
) -> dict[str, Any]:
    rows = [_as_dict(record) for record in records]
    by_key = {(row["example_id"], row["method"]): row for row in rows}
    ids = sorted({row["example_id"] for row in rows})
    a_only = b_only = paired = 0
    for example_id in ids:
        a = by_key.get((example_id, method_a))
        b = by_key.get((example_id, method_b))
        if a is None or b is None:
            continue
        av, bv = bool(a.get(outcome)), bool(b.get(outcome))
        paired += 1
        if av and not bv:
            a_only += 1
        elif bv and not av:
            b_only += 1
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        try:
            from scipy.stats import binomtest

            p_value = float(
                binomtest(
                    min(a_only, b_only), discordant, 0.5, alternative="two-sided"
                ).pvalue
            )
        except ImportError:
            from math import comb

            tail = sum(
                comb(discordant, k) for k in range(0, min(a_only, b_only) + 1)
            ) / (2**discordant)
            p_value = min(1.0, 2.0 * tail)
    return {
        "method_a": method_a,
        "method_b": method_b,
        "outcome": outcome,
        "paired_examples": paired,
        "a_only": a_only,
        "b_only": b_only,
        "p_value": p_value,
    }


def paired_bootstrap_difference(
    records: Iterable[PredictionRecord | dict[str, Any]],
    method_a: str,
    method_b: str,
    *,
    outcome: str = "numeric_exact_match",
    repetitions: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    rows = [_as_dict(record) for record in records]
    by_key = {(row["example_id"], row["method"]): row for row in rows}
    pairs: list[tuple[float, float]] = []
    for example_id in sorted({row["example_id"] for row in rows}):
        a = by_key.get((example_id, method_a))
        b = by_key.get((example_id, method_b))
        if a is not None and b is not None:
            pairs.append((float(bool(a.get(outcome))), float(bool(b.get(outcome)))))
    if not pairs:
        return {"difference": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    array = np.asarray(pairs)
    observed = float((array[:, 1] - array[:, 0]).mean())
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(repetitions, len(array)))
    differences = (array[indices, 1] - array[indices, 0]).mean(axis=1)
    return {
        "difference": observed,
        "ci_low": float(np.quantile(differences, 0.025)),
        "ci_high": float(np.quantile(differences, 0.975)),
    }
