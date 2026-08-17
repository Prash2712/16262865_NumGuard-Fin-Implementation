from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd

from .evaluation import (
    mcnemar_exact,
    paired_bootstrap_difference,
    risk_coverage_curve,
    summarise_records,
)
from .file_sanitizer import sanitise_png, sanitise_xlsx
from .schemas import PredictionRecord


def records_to_dataframe(
    records: Iterable[PredictionRecord | dict[str, Any]],
) -> pd.DataFrame:
    rows = [
        record.to_dict() if isinstance(record, PredictionRecord) else record
        for record in records
    ]
    frame = pd.DataFrame(rows)
    if "proof_certificate" in frame.columns:
        frame["proof_certificate"] = frame["proof_certificate"].apply(
            lambda value: json.dumps(value, sort_keys=True)
            if isinstance(value, dict)
            else value
        )
    return frame


def save_records(
    records: Iterable[PredictionRecord | dict[str, Any]], output_dir: str | Path
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = [
        record.to_dict() if isinstance(record, PredictionRecord) else dict(record)
        for record in records
    ]
    frame = records_to_dataframe(raw_rows)
    frame.to_csv(output_dir / "predictions.csv", index=False)
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in raw_rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return frame


def _suggested_failure(row: pd.Series) -> str:
    if bool(row.get("numeric_exact_match")):
        return "correct"
    if bool(row.get("invalid_response")):
        return "invalid_or_fragmentary_response"
    if bool(row.get("abstained")):
        if row.get("candidate_recall") is True:
            return "over_refusal_or_selection_uncertainty"
        return "retrieval_or_candidate_recall_failure"
    if bool(row.get("unsupported_complete_answer")):
        return "constraint_escape_or_unsupported_answer"
    if row.get("candidate_recall") is False:
        return "candidate_recall_failure"
    if row.get("candidate_recall") is True:
        return "ranking_or_selection_error"
    return "requires_manual_review"


def build_failure_review_sample(
    predictions: pd.DataFrame,
    *,
    sample_size: int = 120,
    seed: int = 42,
) -> pd.DataFrame:
    blinded, _ = build_failure_review_package(
        predictions,
        sample_size=sample_size,
        seed=seed,
    )
    return blinded


def build_failure_review_package(
    predictions: pd.DataFrame,
    *,
    sample_size: int = 120,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = predictions.copy().reset_index(drop=True)
    frame["_row_id"] = frame.index
    frame["suggested_category"] = frame.apply(_suggested_failure, axis=1)
    failures = frame[frame["suggested_category"] != "correct"].copy()

    selected_ids: set[int] = set()
    selected_frames: list[pd.DataFrame] = []
    if not failures.empty:
        grouping = ["method", "suggested_category"]
        groups = failures.groupby(grouping, dropna=False)
        per_group = max(1, sample_size // max(1, groups.ngroups))
        for _, group in groups:
            chosen = group.sample(min(len(group), per_group), random_state=seed)
            selected_frames.append(chosen)
            selected_ids.update(int(value) for value in chosen["_row_id"])

        current = sum(len(item) for item in selected_frames)
        if current < sample_size:
            remaining = failures[~failures["_row_id"].isin(selected_ids)]
            if not remaining.empty:
                selected_frames.append(
                    remaining.sample(
                        min(len(remaining), sample_size - current), random_state=seed
                    )
                )

    sample = (
        pd.concat(selected_frames, ignore_index=True).head(sample_size)
        if selected_frames
        else failures.head(0)
    )
    sample = sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sample["review_case_id"] = [
        "FR-" + hashlib.sha256(
            f"{seed}|{row.example_id}|{row.method}|{index}".encode("utf-8")
        ).hexdigest()[:8].upper()
        for index, row in sample.iterrows()
    ]

    manual_keep = [
        "review_case_id",
        "example_id",
        "question",
        "reference_answer",
        "model_response",
        "candidate_recall",
        "proof_valid",
        "unsupported_complete_answer",
        "selected_candidate_id",
        "selected_candidate_confidence",
        "proof_certificate",
    ]
    output = sample[[column for column in manual_keep if column in sample.columns]].copy()
    output["manual_category"] = ""
    output["manual_evidence_location"] = ""
    output["manual_notes"] = ""
    output["reviewer_confidence"] = ""
    output["second_review_status"] = ""
    key_keep = [
        "review_case_id",
        "example_id",
        "method",
        "suggested_category",
        "configuration_id",
    ]
    review_key = sample[[column for column in key_keep if column in sample.columns]].copy()
    return output, review_key


def generate_figures(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    figures_dir: str | Path,
) -> None:
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    ordered = summary.sort_values("unsupported_complete_answer_rate")
    plt.figure(figsize=(11, 5.8))
    plt.bar(ordered["method"], ordered["unsupported_complete_answer_rate"])
    plt.ylabel("Unsupported complete-answer rate")
    plt.xticks(rotation=38, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "unsupported_answer_rate.png", dpi=240)
    plt.close()

    ordered = summary.sort_values("numeric_answer_accuracy", ascending=False)
    plt.figure(figsize=(11, 5.8))
    plt.bar(ordered["method"], ordered["numeric_answer_accuracy"])
    plt.ylabel("Numeric answer accuracy")
    plt.xticks(rotation=38, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "numeric_answer_accuracy.png", dpi=240)
    plt.close()

    plt.figure(figsize=(8, 6.2))
    for method, group in predictions.groupby("method"):
        curve = risk_coverage_curve(group.to_dict(orient="records"))
        if not curve.empty:
            plt.plot(
                curve["coverage"],
                curve["risk"],
                marker="o",
                markersize=2,
                label=method,
            )
    plt.xlabel("Coverage")
    plt.ylabel("Risk among accepted numeric answers")
    plt.legend(fontsize=6.5, loc="best")
    plt.tight_layout()
    plt.savefig(figures_dir / "risk_coverage_curve.png", dpi=240)
    plt.close()

    soft = summary[summary["method"].str.startswith("soft_prefix_bias_")].copy()
    if not soft.empty:
        soft["bias"] = soft["method"].str.rsplit("_", n=1).str[-1].astype(float)
        soft = soft.sort_values("bias")
        plt.figure(figsize=(8, 5.8))
        plt.plot(
            soft["bias"],
            soft["unsupported_complete_answer_rate"],
            marker="o",
            label="Unsupported rate",
        )
        plt.plot(
            soft["bias"],
            soft["numeric_answer_accuracy"],
            marker="o",
            label="Numeric accuracy",
        )
        plt.plot(
            soft["bias"],
            soft["valid_numeric_response_rate"],
            marker="o",
            label="Valid numeric response rate",
        )
        plt.xlabel("Soft proof-prefix bias")
        plt.ylabel("Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "soft_bias_tradeoff.png", dpi=240)
        plt.close()

    plt.figure(figsize=(8.4, 6.2))
    plt.scatter(
        summary["valid_numeric_response_rate"],
        summary["unsupported_complete_answer_rate"],
    )
    for _, row in summary.iterrows():
        plt.annotate(
            row["method"],
            (
                row["valid_numeric_response_rate"],
                row["unsupported_complete_answer_rate"],
            ),
            fontsize=6.5,
        )
    plt.xlabel("Valid numeric response rate")
    plt.ylabel("Unsupported complete-answer rate")
    plt.tight_layout()
    plt.savefig(figures_dir / "safety_coverage_tradeoff.png", dpi=240)
    plt.close()

    plt.figure(figsize=(11, 5.8))
    ordered = summary.sort_values("candidate_recall", ascending=False)
    plt.bar(ordered["method"], ordered["candidate_recall"])
    plt.ylabel("Proof-lattice candidate recall")
    plt.xticks(rotation=38, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "candidate_recall.png", dpi=240)
    plt.close()

    operation = summary.dropna(subset=["proof_operation_match_rate"]).sort_values(
        "proof_operation_match_rate", ascending=False
    )
    if not operation.empty:
        plt.figure(figsize=(11, 5.8))
        plt.bar(operation["method"], operation["proof_operation_match_rate"])
        plt.ylabel("Proof-operation agreement with FinQA annotation")
        plt.xticks(rotation=38, ha="right")
        plt.tight_layout()
        plt.savefig(figures_dir / "proof_operation_agreement.png", dpi=240)
        plt.close()

    latency = summary.sort_values("mean_total_method_seconds")
    plt.figure(figsize=(11, 5.8))
    plt.bar(latency["method"], latency["mean_total_method_seconds"])
    plt.ylabel("Mean end-to-end method time (seconds)")
    plt.xticks(rotation=38, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "method_latency.png", dpi=240)
    plt.close()


def _paired_comparisons(predictions: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = predictions.to_dict(orient="records")
    methods = sorted(set(predictions["method"]))
    comparisons: list[dict[str, Any]] = []
    if "baseline" not in methods:
        return pd.DataFrame()

    for method in [name for name in methods if name != "baseline"]:
        for outcome in ("numeric_exact_match", "unsupported_complete_answer"):
            comparison = mcnemar_exact(rows, "baseline", method, outcome=outcome)
            difference = paired_bootstrap_difference(
                rows,
                "baseline",
                method,
                outcome=outcome,
                seed=seed,
            )
            comparison.update(
                {
                    "method_minus_baseline_difference": difference["difference"],
                    "difference_ci_low": difference["ci_low"],
                    "difference_ci_high": difference["ci_high"],
                }
            )
            comparisons.append(comparison)
    return pd.DataFrame(comparisons)


def generate_research_outputs(
    records: Iterable[PredictionRecord | dict[str, Any]],
    *,
    output_dir: str | Path,
    figures_dir: str | Path,
    seed: int = 42,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = save_records(records, output_dir)
    summary = summarise_records(predictions.to_dict(orient="records"), seed=seed)
    summary.to_csv(output_dir / "method_summary.csv", index=False)

    comparisons = _paired_comparisons(predictions, seed)
    comparisons.to_csv(output_dir / "paired_comparisons.csv", index=False)

    review, review_key = build_failure_review_package(predictions, seed=seed)
    review.to_excel(output_dir / "failure_review_blinded.xlsx", index=False)
    review_key.to_csv(output_dir / "failure_review_key.csv", index=False)

    with pd.ExcelWriter(output_dir / "research_tables.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Method summary", index=False)
        comparisons.to_excel(writer, sheet_name="Paired comparisons", index=False)
        review.to_excel(writer, sheet_name="Blinded manual review", index=False)
        review_key.to_excel(writer, sheet_name="Review key", index=False)

    sanitise_xlsx(output_dir / "failure_review_blinded.xlsx")
    sanitise_xlsx(output_dir / "research_tables.xlsx")
    generate_figures(predictions, summary, figures_dir)
    for image in Path(figures_dir).glob("*.png"):
        sanitise_png(image)
    return {
        "predictions_csv": output_dir / "predictions.csv",
        "summary_csv": output_dir / "method_summary.csv",
        "comparisons_csv": output_dir / "paired_comparisons.csv",
        "review_xlsx": output_dir / "failure_review_blinded.xlsx",
        "review_key_csv": output_dir / "failure_review_key.csv",
        "tables_xlsx": output_dir / "research_tables.xlsx",
    }
