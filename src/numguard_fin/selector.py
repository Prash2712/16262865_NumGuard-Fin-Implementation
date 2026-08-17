from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from .candidates import AnswerCandidate, build_candidate_lattice
from .evidence import build_evidence_ledger
from .intent import parse_question_intent
from .numeric import numeric_equal, parse_single_number
from .retrieval import retrieve_evidence
from .schemas import FinQAExample


FEATURE_NAMES = (
    "heuristic_confidence",
    "retrieval",
    "metric_fit",
    "type_fit",
    "period_fit",
    "role_fit",
    "proof_valid",
    "operation_match",
    "proof_depth",
    "operand_count",
    "direct",
    "percentage_answer",
    "answer_log_magnitude",
    "answer_negative",
    "operation_lookup",
    "operation_table_aggregate",
    "operation_additive",
    "operation_division",
    "operation_multiplication",
    "operation_composite",
    "metric_period_interaction",
    "role_operation_interaction",
    "specific_proof",
)


def candidate_feature_vector(candidate: AnswerCandidate) -> np.ndarray:
    """Return the fixed selector feature vector.

    All features are computable from the question-conditioned candidate and its proof;
    no reference answer or FinQA gold program is consulted at inference. Proof-family
    indicators and interaction terms help distinguish a generic arithmetic coincidence
    from a question-aligned table or composite proof.
    """
    values = dict(candidate.feature_values)
    values["heuristic_confidence"] = float(candidate.confidence)
    magnitude = 0.0
    if candidate.value is not None:
        absolute = abs(float(candidate.value))
        magnitude = math.log10(1.0 + absolute) / 12.0
    operation = candidate.proof.operation
    table_ops = {"table_sum", "table_average", "table_min", "table_max"}
    additive_ops = {"addition", "subtraction", "average"}
    division_ops = {"division", "margin", "percentage_of_total", "percentage_change"}
    multiplication_ops = {"multiplication"}
    composite_ops = {"aggregate_difference", "percentage_change", "indexed_return"}
    specific_ops = table_ops | composite_ops | {"percentage_of_total"}
    values.update({
        "answer_log_magnitude": max(0.0, min(1.0, magnitude)),
        "answer_negative": float(candidate.value is not None and candidate.value < 0),
        "operation_lookup": float(operation == "lookup"),
        "operation_table_aggregate": float(operation in table_ops),
        "operation_additive": float(operation in additive_ops),
        "operation_division": float(operation in division_ops),
        "operation_multiplication": float(operation in multiplication_ops),
        "operation_composite": float(operation in composite_ops),
        "metric_period_interaction": float(values.get("metric_fit", 0.0))
        * float(values.get("period_fit", 0.0)),
        "role_operation_interaction": float(values.get("role_fit", 0.0))
        * float(values.get("operation_match", 0.0)),
        "specific_proof": float(operation in specific_ops),
    })
    return np.asarray([float(values.get(name, 0.0)) for name in FEATURE_NAMES], dtype=float)



def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


@dataclass(frozen=True)
class LinearCandidateSelector:
    coefficients: tuple[float, ...]
    intercept: float
    blend_weight: float = 0.0
    feature_names: tuple[str, ...] = FEATURE_NAMES
    metadata: dict[str, Any] | None = None

    def learned_score(self, candidate: AnswerCandidate) -> float:
        vector = candidate_feature_vector(candidate)
        if len(vector) != len(self.coefficients):
            raise ValueError("Selector feature vector and coefficient lengths differ.")
        return _sigmoid(
            float(np.dot(vector, np.asarray(self.coefficients))) + self.intercept
        )

    def score(self, candidate: AnswerCandidate) -> float:
        """Return a cross-validated safe blend of heuristic and learned ranking.

        The learned ranker is allowed to influence deployment only when grouped
        out-of-fold validation shows that it improves within-question top-1 selection.
        A blend weight of zero is an explicit, reproducible fallback to the stronger
        heuristic ranker rather than silently allowing a weak model to destroy recall.
        """
        learned = self.learned_score(candidate)
        heuristic = max(0.0, min(1.0, float(candidate.confidence)))
        alpha = max(0.0, min(1.0, float(self.blend_weight)))
        return (1.0 - alpha) * heuristic + alpha * learned

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 3,
            "feature_names": list(self.feature_names),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "blend_weight": self.blend_weight,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LinearCandidateSelector":
        names = tuple(payload.get("feature_names", ()))
        if names != FEATURE_NAMES:
            raise ValueError(
                "Selector feature schema mismatch. Expected "
                f"{FEATURE_NAMES}, found {names}."
            )
        coefficients = tuple(float(value) for value in payload["coefficients"])
        if len(coefficients) != len(FEATURE_NAMES):
            raise ValueError("Invalid selector coefficient count.")
        return cls(
            coefficients=coefficients,
            intercept=float(payload["intercept"]),
            blend_weight=float(payload.get("blend_weight", 0.0)),
            feature_names=names,
            metadata=dict(payload.get("metadata") or {}),
        )


def load_selector(path: str | Path | None) -> Optional[LinearCandidateSelector]:
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Candidate selector not found: {path}")
    return LinearCandidateSelector.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _candidate_label(candidate: AnswerCandidate, reference_answer: str | None) -> int:
    reference = parse_single_number(reference_answer or "")
    predicted = parse_single_number(candidate.answer)
    if reference is None or predicted is None:
        return 0
    strict = numeric_equal(reference, predicted, require_percent_compatibility=True)
    percent_marker_omission = (
        candidate.answer_type == "percentage"
        and numeric_equal(reference, predicted, require_percent_compatibility=False)
    )
    return int(strict or percent_marker_omission)


def candidate_training_rows(
    examples: Iterable[FinQAExample],
    *,
    retrieval_top_k: int = 48,
    max_direct: int = 48,
    max_derived: int = 192,
    training_pool_multiplier: int = 1,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in examples:
        intent = parse_question_intent(example.question)
        ledger = build_evidence_ledger(example)
        retrieved = retrieve_evidence(
            ledger, example.question, intent, top_k=retrieval_top_k
        )
        # The selector fits on the fixed deployment candidate population. The cross-validated safe blend
        # prevents a weak learned ranker from displacing the stronger heuristic, while
        # avoiding the memory cost of materialising a much wider candidate population.
        lattice = build_candidate_lattice(
            retrieved,
            intent,
            include_derived=True,
            max_direct=max(max_direct, max_direct * training_pool_multiplier),
            max_derived=max(max_derived, max_derived * training_pool_multiplier),
            minimum_direct_confidence=0.0,
            minimum_derived_confidence=0.0,
        )
        for candidate in lattice.numeric_candidates():
            vector = candidate_feature_vector(candidate)
            row = {
                "example_id": example.example_id,
                "group": example.filename or example.example_id.split("-")[0],
                "candidate_id": candidate.candidate_id,
                "answer": candidate.answer,
                "label": _candidate_label(candidate, example.reference_answer),
            }
            row.update({name: float(value) for name, value in zip(FEATURE_NAMES, vector)})
            rows.append(row)
    return rows


def _pairwise_rows(
    X: np.ndarray,
    y: np.ndarray,
    example_ids: np.ndarray,
    candidate_scores: np.ndarray,
    indices: np.ndarray,
    *,
    hard_negatives_per_positive: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct balanced within-question preference pairs.

    Candidate correctness is extremely sparse. Row-wise classification therefore
    optimises candidate-level AUC rather than the operational objective: ranking the
    correct proof above alternatives for the same question. Each positive is paired
    with the strongest heuristic negatives from its own question, and the reversed
    difference is added to keep the pair labels exactly balanced.
    """
    selected = set(int(index) for index in indices)
    by_example: dict[str, list[int]] = {}
    for index in indices:
        by_example.setdefault(str(example_ids[index]), []).append(int(index))
    pair_x: list[np.ndarray] = []
    pair_y: list[int] = []
    for candidate_indices in by_example.values():
        positives = [index for index in candidate_indices if y[index] == 1]
        negatives = [index for index in candidate_indices if y[index] == 0]
        if not positives or not negatives:
            continue
        negatives.sort(key=lambda index: (-candidate_scores[index], index))
        negatives = negatives[:hard_negatives_per_positive]
        for positive in positives:
            for negative in negatives:
                difference = X[positive] - X[negative]
                pair_x.append(difference)
                pair_y.append(1)
                pair_x.append(-difference)
                pair_y.append(0)
    if not pair_x:
        return np.empty((0, X.shape[1]), dtype=float), np.empty((0,), dtype=int)
    return np.vstack(pair_x), np.asarray(pair_y, dtype=int)


def _raw_linear_parameters(pipeline) -> tuple[np.ndarray, float]:
    scaler = pipeline.named_steps["scale"]
    model = pipeline.named_steps["rank"]
    scale = np.asarray(scaler.scale_, dtype=float)
    scale = np.where(scale == 0.0, 1.0, scale)
    coefficients = np.asarray(model.coef_[0], dtype=float) / scale
    intercept = float(model.intercept_[0] - np.dot(model.coef_[0], scaler.mean_ / scale))
    return coefficients, intercept


def fit_linear_selector(
    rows: list[dict[str, Any]],
    *,
    seed: int = 42,
    folds: int = 5,
) -> tuple[LinearCandidateSelector, dict[str, Any]]:
    if not rows:
        raise ValueError("No candidate training rows were provided.")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.model_selection import GroupKFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required to fit the candidate selector.") from exc

    X = np.asarray([[float(row[name]) for name in FEATURE_NAMES] for row in rows])
    y = np.asarray([int(row["label"]) for row in rows])
    groups = np.asarray([str(row["group"]) for row in rows])
    example_ids = np.asarray([str(row["example_id"]) for row in rows])
    heuristic_scores = np.asarray([float(row["heuristic_confidence"]) for row in rows])
    if y.sum() == 0:
        raise RuntimeError(
            "The candidate lattice recalled no training references; a selector cannot be fitted."
        )

    def make_pipeline():
        return Pipeline([
            ("scale", StandardScaler()),
            ("rank", LogisticRegression(max_iter=2500, random_state=seed, C=1.0)),
        ])

    unique_groups = np.unique(groups)
    n_splits = min(folds, len(unique_groups))
    out_of_fold = np.full(len(rows), np.nan)
    fold_metrics: list[dict[str, Any]] = []
    all_indices = np.arange(len(rows))
    if n_splits >= 2:
        splitter = GroupKFold(n_splits=n_splits)
        for fold, (train_idx, valid_idx) in enumerate(
            splitter.split(X, y, groups), start=1
        ):
            pair_x, pair_y = _pairwise_rows(
                X, y, example_ids, heuristic_scores, train_idx
            )
            if len(pair_y) == 0 or len(np.unique(pair_y)) < 2:
                continue
            pipeline = make_pipeline()
            pipeline.fit(pair_x, pair_y)
            coefficients, intercept = _raw_linear_parameters(pipeline)
            probabilities = np.asarray([
                _sigmoid(float(np.dot(X[index], coefficients)) + intercept)
                for index in valid_idx
            ])
            out_of_fold[valid_idx] = probabilities
            fold_y = y[valid_idx]
            metrics: dict[str, Any] = {
                "fold": fold,
                "candidate_rows": int(len(valid_idx)),
                "pairwise_training_rows": int(len(pair_y)),
                "positives": int(fold_y.sum()),
            }
            if len(np.unique(fold_y)) == 2:
                metrics["roc_auc"] = float(roc_auc_score(fold_y, probabilities))
                metrics["average_precision"] = float(
                    average_precision_score(fold_y, probabilities)
                )
            valid_examples: dict[str, list[int]] = {}
            for position, index in enumerate(valid_idx):
                valid_examples.setdefault(example_ids[index], []).append(position)
            recalled, correct = 0, 0
            for positions in valid_examples.values():
                labels = fold_y[positions]
                if labels.max() == 0:
                    continue
                recalled += 1
                best_position = max(positions, key=lambda position: probabilities[position])
                correct += int(fold_y[best_position] == 1)
            metrics["top1_accuracy_given_candidate_recall"] = correct / max(1, recalled)
            fold_metrics.append(metrics)

    pair_x, pair_y = _pairwise_rows(
        X, y, example_ids, heuristic_scores, all_indices
    )
    if len(pair_y) == 0:
        raise RuntimeError("No within-question positive/negative selector pairs were available.")
    final_pipeline = make_pipeline()
    final_pipeline.fit(pair_x, pair_y)
    coefficients, intercept = _raw_linear_parameters(final_pipeline)

    fallback_scores = np.asarray([
        _sigmoid(float(np.dot(vector, coefficients)) + intercept) for vector in X
    ])
    learned_oof_scores = np.where(np.isnan(out_of_fold), fallback_scores, out_of_fold)
    by_example: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_example.setdefault(str(row["example_id"]), []).append(index)

    def top1_accuracy(scores: np.ndarray) -> tuple[int, int, float]:
        recalled, correct = 0, 0
        for indices in by_example.values():
            if y[indices].max() == 0:
                continue
            recalled += 1
            best = max(indices, key=lambda index: (scores[index], -index))
            correct += int(y[best] == 1)
        return recalled, correct, correct / max(1, recalled)

    recalled_examples, heuristic_correct, heuristic_top1 = top1_accuracy(heuristic_scores)
    _, learned_correct, learned_top1 = top1_accuracy(learned_oof_scores)

    # Choose the amount of learned influence strictly by grouped out-of-fold top-1
    # performance. Ties prefer the lower learned weight so a weak ranker cannot
    # displace a stronger deterministic heuristic merely because of numerical noise.
    blend_grid = [index / 20.0 for index in range(21)]
    blend_results: list[dict[str, float | int]] = []
    best_alpha = 0.0
    best_accuracy = heuristic_top1
    best_correct = heuristic_correct
    for alpha in blend_grid:
        blended = (1.0 - alpha) * heuristic_scores + alpha * learned_oof_scores
        _, correct, accuracy = top1_accuracy(blended)
        blend_results.append({
            "blend_weight": alpha,
            "top1_correct": correct,
            "top1_accuracy_given_candidate_recall": accuracy,
        })
        if accuracy > best_accuracy + 1e-12:
            best_alpha, best_accuracy, best_correct = alpha, accuracy, correct

    selector = LinearCandidateSelector(
        coefficients=tuple(float(value) for value in coefficients),
        intercept=float(intercept),
        blend_weight=float(best_alpha),
        metadata={},
    )
    report = {
        "selector_objective": "cross-validated safe blend of heuristic and within-question pairwise proof ranking",
        "training_candidate_rows": len(rows),
        "pairwise_training_rows": int(len(pair_y)),
        "hard_negatives_per_positive": 24,
        "training_examples": len(by_example),
        "positive_candidate_rows": int(y.sum()),
        "candidate_recall_on_training_examples": recalled_examples / max(1, len(by_example)),
        "heuristic_top1_accuracy_given_candidate_recall": heuristic_top1,
        "learned_out_of_fold_top1_accuracy_given_candidate_recall": learned_top1,
        "selected_blend_weight": best_alpha,
        "out_of_fold_top1_accuracy_given_candidate_recall": best_accuracy,
        "top1_accuracy_given_candidate_recall": best_accuracy,
        "blend_grid_results": blend_results,
        "safe_fallback_used": bool(best_alpha == 0.0),
        "grouped_cross_validation": fold_metrics,
        "seed": seed,
        "feature_names": list(FEATURE_NAMES),
    }
    payload = selector.to_dict()
    payload["metadata"] = report
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    report["selector_sha256"] = digest
    selector = LinearCandidateSelector(
        coefficients=selector.coefficients,
        intercept=selector.intercept,
        blend_weight=selector.blend_weight,
        metadata=report,
    )
    return selector, report

