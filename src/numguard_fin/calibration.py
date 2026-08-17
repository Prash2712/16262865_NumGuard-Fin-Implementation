from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import pandas as pd


@dataclass(frozen=True)
class CalibrationDecision:
    threshold: float
    target_risk: float
    observed_risk: float | None
    risk_upper_bound: float
    coverage: float
    accepted_examples: int
    calibration_examples: int
    confidence_level: float
    risk_type: str = "semantic"
    feasible: bool = True
    minimum_coverage: float = 0.0
    minimum_accepted: int = 1
    reason: str = "target achieved"
    best_available_threshold: float | None = None
    best_available_risk_upper_bound: float | None = None
    best_available_coverage: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clopper_pearson_upper(errors: int, total: int, confidence: float = 0.95) -> float:
    if total <= 0:
        return 1.0
    if errors >= total:
        return 1.0
    alpha = 1.0 - confidence
    try:
        from scipy.stats import beta

        return float(beta.ppf(1.0 - alpha, errors + 1, total - errors))
    except ImportError:
        z = 1.959963984540054
        p = errors / total
        denominator = 1.0 + z * z / total
        centre = p + z * z / (2 * total)
        radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        return min(1.0, (centre + radius) / denominator)


def _is_error(row: dict[str, Any], risk_type: str) -> bool:
    if risk_type == "semantic":
        return not bool(row.get("numeric_exact_match"))
    if risk_type == "provenance":
        return bool(row.get("unsupported_complete_answer")) or not bool(row.get("proof_valid"))
    raise ValueError(f"Unsupported risk_type: {risk_type!r}")


def calibration_curve(
    rows: Iterable[dict[str, Any]],
    *,
    confidence_level: float = 0.95,
    risk_type: str = "semantic",
) -> pd.DataFrame:
    rows = list(rows)
    scored = [
        row
        for row in rows
        if row.get("selected_candidate_confidence") is not None
        and bool(row.get("valid_numeric_response"))
    ]
    thresholds = sorted(
        {0.0, 1.0, *(float(row["selected_candidate_confidence"]) for row in scored)}
    )
    output: list[dict[str, Any]] = []
    for threshold in thresholds:
        accepted = [
            row
            for row in scored
            if float(row["selected_candidate_confidence"]) >= threshold
        ]
        errors = sum(_is_error(row, risk_type) for row in accepted)
        total = len(accepted)
        risk = errors / total if total else math.nan
        output.append(
            {
                "risk_type": risk_type,
                "threshold": threshold,
                "accepted_examples": total,
                "coverage": total / len(rows) if rows else 0.0,
                "errors": errors,
                "observed_risk": risk,
                "risk_upper_bound": clopper_pearson_upper(
                    errors, total, confidence_level
                ),
            }
        )
    return pd.DataFrame(output).sort_values(["threshold"]).reset_index(drop=True)


def select_risk_threshold(
    rows: Iterable[dict[str, Any]],
    *,
    target_risk: float = 0.20,
    confidence_level: float = 0.95,
    risk_type: str = "semantic",
    minimum_coverage: float = 0.0,
    minimum_accepted: int = 1,
) -> CalibrationDecision:
    rows = list(rows)
    curve = calibration_curve(
        rows, confidence_level=confidence_level, risk_type=risk_type
    )
    eligible = curve[
        (curve["accepted_examples"] >= minimum_accepted)
        & (curve["coverage"] >= minimum_coverage)
    ]
    feasible = eligible[eligible["risk_upper_bound"] <= target_risk]

    if not feasible.empty:
        best = feasible.sort_values(
            ["coverage", "threshold"], ascending=[False, True]
        ).iloc[0]
        return CalibrationDecision(
            threshold=float(best["threshold"]),
            target_risk=target_risk,
            observed_risk=float(best["observed_risk"]),
            risk_upper_bound=float(best["risk_upper_bound"]),
            coverage=float(best["coverage"]),
            accepted_examples=int(best["accepted_examples"]),
            calibration_examples=len(rows),
            confidence_level=confidence_level,
            risk_type=risk_type,
            feasible=True,
            minimum_coverage=minimum_coverage,
            minimum_accepted=minimum_accepted,
            reason="target achieved with the required non-zero coverage gate",
        )

    best_available = None
    if not eligible.empty:
        best_available = eligible.sort_values(
            ["risk_upper_bound", "coverage"], ascending=[True, False]
        ).iloc[0]
    return CalibrationDecision(
        threshold=1.000001,
        target_risk=target_risk,
        observed_risk=None,
        risk_upper_bound=1.0,
        coverage=0.0,
        accepted_examples=0,
        calibration_examples=len(rows),
        confidence_level=confidence_level,
        risk_type=risk_type,
        feasible=False,
        minimum_coverage=minimum_coverage,
        minimum_accepted=minimum_accepted,
        reason=(
            "No threshold achieved the requested risk bound while satisfying the "
            "minimum coverage and accepted-example gates. Public-test execution is blocked."
        ),
        best_available_threshold=(
            None if best_available is None else float(best_available["threshold"])
        ),
        best_available_risk_upper_bound=(
            None
            if best_available is None
            else float(best_available["risk_upper_bound"])
        ),
        best_available_coverage=(
            None if best_available is None else float(best_available["coverage"])
        ),
    )
