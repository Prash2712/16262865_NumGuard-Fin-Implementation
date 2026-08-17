import pytest

from numguard_fin.trie import TokenTrie, build_token_trie, make_prefix_allowed_tokens_fn


class DeterministicTokenizer:
    eos_token_id = 0

    def __init__(self):
        self.vocab = {"1": 1, "2": 2, "3": 3, ".": 4, "%": 5, "A": 6, "B": 7}

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [self.vocab[character] for character in text]


@pytest.mark.parametrize(
    "sequence,accepted",
    [
        ([1, 2], True),
        ([1, 3], True),
        ([1], False),
        ([2, 1], False),
        ([1, 2, 3], False),
    ],
)
def test_token_trie_accepts_only_complete_sequences(sequence, accepted):
    trie = TokenTrie([[1, 2], [1, 3]])
    assert trie.accepts(sequence) is accepted


def test_allowed_next_tracks_prefix():
    trie = TokenTrie([[1, 2], [1, 3]])
    assert trie.allowed_next([], 0) == [1]
    assert trie.allowed_next([1], 0) == [2, 3]
    assert trie.allowed_next([1, 2], 0) == [0]


def test_unknown_prefix_can_only_terminate():
    assert TokenTrie([[1, 2]]).allowed_next([9], 0) == [0]


def test_whole_sequence_lock_rejects_fragment_recombination():
    trie = TokenTrie([[1, 2], [3, 4]])
    assert not trie.accepts([1, 4])


def test_build_token_trie():
    trie = build_token_trie(DeterministicTokenizer(), ["12", "13"])
    assert trie.accepts([1, 2])
    assert trie.accepts([1, 3])


def test_prefix_callback_strips_decoder_start():
    fn = make_prefix_allowed_tokens_fn(DeterministicTokenizer(), ["12"], decoder_start_token_id=9)
    assert fn(0, [9]) == [1]
    assert fn(0, [9, 1]) == [2]
    assert fn(0, [9, 1, 2]) == [0]


def test_empty_answer_set_fails():
    with pytest.raises(ValueError):
        build_token_trie(DeterministicTokenizer(), [])


def test_proof_prior_processor_hard_masks_and_prefers_stronger_branch():
    import torch

    from numguard_fin.trie import ProofPriorLogitsProcessor

    tokenizer = DeterministicTokenizer()
    processor = ProofPriorLogitsProcessor(
        tokenizer,
        [("12", 0.9), ("13", 0.1)],
        prior_strength=2.0,
        decoder_start_token_id=9,
    )
    scores = torch.zeros((1, 8), dtype=torch.float32)
    first = processor(torch.tensor([[9, 1]]), scores.clone())
    assert torch.isneginf(first[0, 6])
    assert first[0, 2] > first[0, 3]


def test_weighted_trie_retains_complete_answer_guarantee():
    from numguard_fin.trie import build_weighted_token_trie

    trie = build_weighted_token_trie(
        DeterministicTokenizer(), [("12", 0.9), ("13", 0.1)]
    )
    assert trie.accepts([1, 2])
    assert trie.accepts([1, 3])
    assert not trie.accepts([1, 2, 3])
