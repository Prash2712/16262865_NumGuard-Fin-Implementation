from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Optional, Protocol

from .candidates import ABSTAIN_RESPONSE, CandidateLattice
from .schemas import AnswerCandidate, QuestionIntent, RetrievedEvidence
from .trie import (
    FragmentTokenMask,
    ProofPriorLogitsProcessor,
    SoftProofPrefixBias,
    make_prefix_allowed_tokens_fn,
)


@dataclass(frozen=True)
class GenerationResult:
    response: str
    sequence_score: Optional[float]
    inference_seconds: float
    input_token_count: int = 0
    input_truncated: bool = False
    control_seconds: float = 0.0


class Generator(Protocol):
    model_name: str

    def free_generate(self, prompt: str) -> GenerationResult: ...

    def constrained_generate(self, prompt: str, allowed_answers: Iterable[str]) -> GenerationResult: ...

    def soft_generate(self, prompt: str, allowed_answers: Iterable[str], bias: float) -> GenerationResult: ...

    def ranked_constrained_generate(
        self, prompt: str, candidates: Iterable[AnswerCandidate], prior_strength: float
    ) -> GenerationResult: ...

    def fragment_generate(self, prompt: str, allowed_answers: Iterable[str]) -> GenerationResult: ...


def build_evidence_prompt(
    question: str,
    retrieved: Iterable[RetrievedEvidence],
    *,
    lattice: Optional[CandidateLattice] = None,
    include_candidates: bool = False,
    evidence_display_limit: int = 24,
    candidate_display_limit: int = 24,
) -> str:
    evidence_groups: dict[str, list[str]] = {}
    for entry in list(retrieved)[:evidence_display_limit]:
        item = entry.item
        evidence_groups.setdefault(item.text, []).append(item.evidence_id)
    evidence_lines = [
        f"[{','.join(evidence_ids)}] {text}"
        for text, evidence_ids in evidence_groups.items()
    ]
    evidence_block = "\n".join(evidence_lines) or "[none] No relevant evidence was retrieved."
    candidate_block = ""
    if include_candidates and lattice is not None:
        candidate_lines = []
        numeric = list(lattice.numeric_candidates())[:candidate_display_limit]
        for candidate in numeric:
            candidate_lines.append(
                f"- {candidate.answer} | {candidate.proof.operation} | "
                f"support={candidate.confidence:.3f}"
            )
        candidate_lines.append(f"- {ABSTAIN_RESPONSE}")
        hidden = max(0, len(lattice.numeric_candidates()) - len(numeric))
        suffix = (
            f"\n[{hidden} additional certified numeric answers are enforced by the decoder.]"
            if hidden else ""
        )
        candidate_block = (
            "\n\nTOP CERTIFIED ANSWER OPTIONS:\n"
            + "\n".join(candidate_lines)
            + suffix
        )

    return (
        "Answer the financial question using only the retrieved evidence below.\n"
        "Return one complete numeric answer only. Include % only when the answer is a percentage.\n"
        f"Return {ABSTAIN_RESPONSE} when the evidence does not support a complete answer.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"RETRIEVED EVIDENCE:\n{evidence_block}"
        f"{candidate_block}\n\nANSWER:"
    )


def build_baseline_prompt(
    question: str, retrieved: Iterable[RetrievedEvidence], *, evidence_display_limit: int = 24
) -> str:
    evidence_groups: dict[str, list[str]] = {}
    for entry in list(retrieved)[:evidence_display_limit]:
        evidence_groups.setdefault(entry.item.text, []).append(entry.item.evidence_id)
    evidence_lines = [
        f"[{','.join(evidence_ids)}] {text}"
        for text, evidence_ids in evidence_groups.items()
    ]
    evidence_block = "\n".join(evidence_lines) or "[none]"
    return (
        "Answer the financial question. Return the final answer only.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"EVIDENCE:\n{evidence_block}\n\nANSWER:"
    )


class HeuristicSelector:
    """Deterministic engineering-test selector.

    It exercises the candidate lattice, proof verifier and counterfactual audit without
    pretending to be a language-model result. Empirical runs must use
    HuggingFaceGenerator.
    """

    model_name = "heuristic-engineering-check"

    def __init__(self, lattice: CandidateLattice, threshold: float = 0.0) -> None:
        self.lattice = lattice
        self.threshold = threshold

    def _select(self) -> str:
        numeric = sorted(
            self.lattice.numeric_candidates(),
            key=lambda candidate: (-candidate.confidence, candidate.candidate_id),
        )
        if not numeric or numeric[0].confidence < self.threshold:
            return ABSTAIN_RESPONSE
        return numeric[0].answer

    def free_generate(self, prompt: str) -> GenerationResult:
        del prompt
        start = time.perf_counter()
        response = self._select()
        return GenerationResult(response, None, time.perf_counter() - start, 0, False)

    def constrained_generate(self, prompt: str, allowed_answers: Iterable[str]) -> GenerationResult:
        del prompt
        start = time.perf_counter()
        allowed = set(allowed_answers)
        response = self._select()
        if response not in allowed:
            response = ABSTAIN_RESPONSE
        return GenerationResult(response, None, time.perf_counter() - start, 0, False)

    def soft_generate(self, prompt: str, allowed_answers: Iterable[str], bias: float) -> GenerationResult:
        del bias
        return self.constrained_generate(prompt, allowed_answers)

    def ranked_constrained_generate(
        self, prompt: str, candidates: Iterable[AnswerCandidate], prior_strength: float
    ) -> GenerationResult:
        del prior_strength
        return self.constrained_generate(prompt, (candidate.answer for candidate in candidates))

    def fragment_generate(self, prompt: str, allowed_answers: Iterable[str]) -> GenerationResult:
        return self.constrained_generate(prompt, allowed_answers)


class HuggingFaceGenerator:
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        *,
        device: str = "auto",
        max_input_tokens: int = 512,
        max_new_tokens: int = 24,
        num_beams: int = 4,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "HuggingFace generation requires torch and transformers. "
                "Install the project requirements before running empirical experiments."
            ) from exc

        self.torch = torch
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.decoder_start_token_id = self.model.config.decoder_start_token_id

    def _inputs(self, prompt: str):
        return self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.device)

    def _decode_generate(self, prompt: str, **kwargs) -> GenerationResult:
        input_token_count = len(self.tokenizer.encode(prompt, add_special_tokens=True))
        inputs = self._inputs(prompt)
        start = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                **kwargs,
            )
        elapsed = time.perf_counter() - start
        response = self.tokenizer.decode(output.sequences[0], skip_special_tokens=True).strip()
        score = None
        if getattr(output, "sequences_scores", None) is not None:
            score = float(output.sequences_scores[0].detach().cpu())
        return GenerationResult(
            response,
            score,
            elapsed,
            input_token_count,
            input_token_count > self.max_input_tokens,
        )

    def free_generate(self, prompt: str) -> GenerationResult:
        return self._decode_generate(prompt)

    def constrained_generate(self, prompt: str, allowed_answers: Iterable[str]) -> GenerationResult:
        answers = tuple(dict.fromkeys(str(answer) for answer in allowed_answers))
        prefix_fn = make_prefix_allowed_tokens_fn(
            self.tokenizer,
            answers,
            decoder_start_token_id=self.decoder_start_token_id,
        )
        return self._decode_generate(prompt, prefix_allowed_tokens_fn=prefix_fn)

    def ranked_constrained_generate(
        self,
        prompt: str,
        candidates: Iterable[AnswerCandidate],
        prior_strength: float = 2.0,
    ) -> GenerationResult:
        try:
            from transformers import LogitsProcessorList
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for proof-prior decoding."
            ) from exc

        weighted_answers = [
            (
                candidate.answer,
                max(1e-4, float(candidate.confidence)),
            )
            for candidate in candidates
        ]

        processor = ProofPriorLogitsProcessor(
            self.tokenizer,
            weighted_answers,
            prior_strength=prior_strength,
            decoder_start_token_id=self.decoder_start_token_id,
        )

        return self._decode_generate(
            prompt,
            logits_processor=LogitsProcessorList([processor]),
        )

    def soft_generate(self, prompt: str, allowed_answers: Iterable[str], bias: float) -> GenerationResult:
        try:
            from transformers import LogitsProcessorList
        except ImportError as exc:
            raise RuntimeError("transformers is required for soft proof-prefix bias.") from exc
        processor = SoftProofPrefixBias(
            self.tokenizer,
            allowed_answers,
            bias=bias,
            decoder_start_token_id=self.decoder_start_token_id,
        )
        return self._decode_generate(prompt, logits_processor=LogitsProcessorList([processor]))

    def fragment_generate(self, prompt: str, allowed_answers: Iterable[str]) -> GenerationResult:
        try:
            from transformers import LogitsProcessorList
        except ImportError as exc:
            raise RuntimeError("transformers is required for the fragment-token ablation.") from exc
        processor = FragmentTokenMask(self.tokenizer, allowed_answers)
        return self._decode_generate(prompt, logits_processor=LogitsProcessorList([processor]))
