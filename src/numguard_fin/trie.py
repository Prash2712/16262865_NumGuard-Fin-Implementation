from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional


@dataclass
class TrieNode:
    children: dict[int, "TrieNode"] = field(default_factory=dict)
    terminal: bool = False
    max_weight: float = 0.0
    terminal_weight: float = 0.0


class TokenTrie:
    """Prefix trie over complete tokenizer sequences.

    Unlike fragment-level digit masking, this structure admits a token only when the
    entire decoder prefix can still be completed to a certified candidate answer.
    """

    def __init__(self, sequences: Iterable[Iterable[int]] = ()) -> None:
        self.root = TrieNode()
        self.sequence_count = 0
        for sequence in sequences:
            self.insert(sequence)

    def insert(self, sequence: Iterable[int], *, weight: float = 0.0) -> None:
        node = self.root
        node.max_weight = max(node.max_weight, float(weight))
        consumed = False
        for token_id in sequence:
            consumed = True
            node = node.children.setdefault(int(token_id), TrieNode())
            node.max_weight = max(node.max_weight, float(weight))
        if consumed:
            if not node.terminal:
                node.terminal = True
                self.sequence_count += 1
            node.terminal_weight = max(node.terminal_weight, float(weight))

    def node_for_prefix(self, prefix: Iterable[int]) -> Optional[TrieNode]:
        node = self.root
        for token_id in prefix:
            node = node.children.get(int(token_id))
            if node is None:
                return None
        return node

    def allowed_next(self, prefix: Iterable[int], eos_token_id: int) -> list[int]:
        node = self.node_for_prefix(prefix)
        if node is None:
            return [int(eos_token_id)]
        allowed = sorted(node.children)
        if node.terminal:
            allowed.append(int(eos_token_id))
        return allowed or [int(eos_token_id)]

    def accepts(self, sequence: Iterable[int]) -> bool:
        node = self.node_for_prefix(sequence)
        return bool(node and node.terminal)


def build_token_trie(tokenizer, answers: Iterable[str]) -> TokenTrie:
    sequences: list[list[int]] = []
    for answer in answers:
        token_ids = tokenizer.encode(str(answer), add_special_tokens=False)
        if token_ids:
            sequences.append([int(token_id) for token_id in token_ids])
    if not sequences:
        raise ValueError("Cannot build a proof-lock trie without candidate sequences.")
    return TokenTrie(sequences)


def make_prefix_allowed_tokens_fn(
    tokenizer,
    answers: Iterable[str],
    *,
    decoder_start_token_id: Optional[int] = None,
):
    trie = build_token_trie(tokenizer, answers)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id.")

    def prefix_allowed_tokens_fn(batch_id: int, input_ids) -> list[int]:
        del batch_id
        prefix = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
        if prefix and decoder_start_token_id is not None and prefix[0] == decoder_start_token_id:
            prefix = prefix[1:]
        return trie.allowed_next(prefix, eos_token_id)

    prefix_allowed_tokens_fn.trie = trie  # type: ignore[attr-defined]
    return prefix_allowed_tokens_fn


def build_weighted_token_trie(tokenizer, weighted_answers: Iterable[tuple[str, float]]) -> TokenTrie:
    trie = TokenTrie()
    for answer, weight in weighted_answers:
        token_ids = tokenizer.encode(str(answer), add_special_tokens=False)
        if token_ids:
            trie.insert((int(token_id) for token_id in token_ids), weight=float(weight))
    if trie.sequence_count == 0:
        raise ValueError("Cannot build a weighted proof-lock trie without candidate sequences.")
    return trie


class ProofPriorLogitsProcessor:
    """Hard whole-sequence lock with a train-only proof-quality prior.

    Every disallowed token is still assigned negative infinity, preserving the complete
    answer guarantee. Among admissible branches, the maximum frozen selector confidence
    of any completion supplies a bounded log-prior. This separates *safety* (the trie)
    from *selection* (the train-only candidate ranker) and does not alter model weights.
    """

    def __init__(
        self,
        tokenizer,
        weighted_answers: Iterable[tuple[str, float]],
        *,
        prior_strength: float = 2.0,
        decoder_start_token_id: Optional[int] = None,
        minimum_weight: float = 1e-4,
    ) -> None:
        self.tokenizer = tokenizer
        self.trie = build_weighted_token_trie(tokenizer, weighted_answers)
        self.prior_strength = float(prior_strength)
        self.decoder_start_token_id = decoder_start_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.minimum_weight = float(minimum_weight)
        if self.eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id.")

    def __call__(self, input_ids, scores):
        import math
        import torch

        for row_index in range(scores.shape[0]):
            prefix = input_ids[row_index].tolist()
            if (
                prefix
                and self.decoder_start_token_id is not None
                and prefix[0] == self.decoder_start_token_id
            ):
                prefix = prefix[1:]
            node = self.trie.node_for_prefix(prefix)
            mask = torch.full_like(scores[row_index], float("-inf"))
            if node is None:
                mask[int(self.eos_token_id)] = scores[row_index, int(self.eos_token_id)]
                scores[row_index] = mask
                continue
            branches: list[tuple[int, float]] = [
                (int(token_id), child.max_weight)
                for token_id, child in node.children.items()
            ]
            if node.terminal:
                branches.append((int(self.eos_token_id), node.terminal_weight))
            if not branches:
                branches = [(int(self.eos_token_id), self.minimum_weight)]
            for token_id, weight in branches:
                prior = self.prior_strength * math.log(max(self.minimum_weight, float(weight)))
                mask[token_id] = scores[row_index, token_id] + prior
            scores[row_index] = mask
        return scores


class SoftProofPrefixBias:
    """Optional soft ablation that biases, but does not guarantee, proof prefixes.

    This is deliberately kept separate from the hard proof lock. It exists only to
    produce a strength-versus-risk curve and must not be described as a guarantee.
    """

    def __init__(
        self,
        tokenizer,
        answers: Iterable[str],
        *,
        bias: float,
        decoder_start_token_id: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.trie = build_token_trie(tokenizer, answers)
        self.bias = float(bias)
        self.decoder_start_token_id = decoder_start_token_id
        self.eos_token_id = tokenizer.eos_token_id

    def __call__(self, input_ids, scores):
        for row_index in range(scores.shape[0]):
            prefix = input_ids[row_index].tolist()
            if prefix and self.decoder_start_token_id is not None and prefix[0] == self.decoder_start_token_id:
                prefix = prefix[1:]
            node = self.trie.node_for_prefix(prefix)
            if node is None:
                continue
            allowed = set(node.children)
            if node.terminal and self.eos_token_id is not None:
                allowed.add(int(self.eos_token_id))
            if not allowed:
                continue
            scores[row_index, list(allowed)] += self.bias
        return scores

class FragmentTokenMask:
    """Deliberately weak token-fragment constraint used only as an ablation.

    The processor admits any token that occurs anywhere in a certified answer. Because
    token fragments can be recombined into an uncertified sequence, this is not a
    safety guarantee. It is retained to demonstrate the difference between fragment
    masking and the proposed whole-sequence prefix lock.
    """

    def __init__(self, tokenizer, answers: Iterable[str]) -> None:
        token_ids: set[int] = set()
        for answer in answers:
            token_ids.update(
                int(token_id)
                for token_id in tokenizer.encode(str(answer), add_special_tokens=False)
            )
        if tokenizer.eos_token_id is not None:
            token_ids.add(int(tokenizer.eos_token_id))
        if not token_ids:
            raise ValueError("Cannot build a fragment-token ablation without tokens.")
        self.allowed_token_ids = tuple(sorted(token_ids))

    def __call__(self, input_ids, scores):
        del input_ids
        import torch

        mask = torch.full_like(scores, float("-inf"))
        mask[:, list(self.allowed_token_ids)] = scores[:, list(self.allowed_token_ids)]
        return mask
