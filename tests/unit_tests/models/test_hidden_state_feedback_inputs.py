# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for the functions that prepare the inputs of a hidden-state fusion."""

import pytest
import torch

from megatron.core.models.hybrid.hidden_state_feedback.inputs import (
    document_starts_from_cu_seqlens,
    document_starts_from_eod,
    sample_prefix_mask,
    select_feedback_inputs,
    shift_hidden_states,
)
from megatron.core.models.hybrid.hidden_state_feedback.specs import gated_hidden_state_fusion_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

SEQ, BATCH, HIDDEN = 5, 3, 4
FUSION_HIDDEN = 16
EOD = 9


def _hidden_states(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return states in which every element has a distinct nonzero value."""

    return torch.arange(1, SEQ * BATCH * HIDDEN + 1, dtype=dtype).reshape(SEQ, BATCH, HIDDEN)


def _generator(seed: int = 0) -> torch.Generator:
    """Return a seeded CPU generator."""

    return torch.Generator().manual_seed(seed)


def _row_starts(positions: list[int], num_rows: int, num_tokens: int) -> torch.Tensor:
    """Return ``num_rows`` equal rows of document starts at the given positions."""

    starts = torch.zeros(num_rows, num_tokens, dtype=torch.bool)
    starts[:, positions] = True
    return starts


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_shift_moves_each_state_to_the_next_position(dtype):
    hidden_states = _hidden_states(dtype)
    reference = hidden_states.clone()
    shifted = shift_hidden_states(hidden_states)
    assert shifted.shape == hidden_states.shape
    assert shifted.dtype == dtype
    assert torch.equal(shifted[1:], reference[:-1])
    assert torch.count_nonzero(shifted[0]) == 0
    assert torch.equal(hidden_states, reference)


def test_shift_gradient_returns_to_the_source_position():
    hidden_states = _hidden_states().requires_grad_(True)
    weights = torch.randn(SEQ, BATCH, HIDDEN, generator=_generator())
    (shift_hidden_states(hidden_states) * weights).sum().backward()
    assert torch.equal(hidden_states.grad[:-1], weights[1:])
    # The last state has no successor in the sequence.
    assert torch.count_nonzero(hidden_states.grad[-1]) == 0


def test_eod_starts_mark_the_first_position_and_each_position_after_eod():
    rows = [
        ([1, 2, EOD, 3, 4, EOD, 5], [1, 0, 0, 1, 0, 0, 1]),
        ([1, 2, 3, 4, 5, 6, 7], [1, 0, 0, 0, 0, 0, 0]),
        ([1, EOD, EOD, EOD, 2, 3, 4], [1, 0, 1, 1, 1, 0, 0]),
        ([EOD, 1, 2, 3, 4, 5, EOD], [1, 1, 0, 0, 0, 0, 0]),
    ]
    tokens = torch.tensor([row for row, _ in rows])
    expected = torch.tensor([starts for _, starts in rows], dtype=torch.bool)
    starts = document_starts_from_eod(tokens, EOD)
    assert starts.dtype == torch.bool
    assert torch.equal(starts, expected)


@pytest.mark.parametrize(
    "offsets, num_tokens, expected",
    [
        pytest.param([0, 5, 7, 11], 11, [0, 5, 7], id="offsets_end_at_the_last_token"),
        pytest.param([0, 5, 7, 11], 16, [0, 5, 7, 11], id="tokens_after_the_last_offset"),
        pytest.param([0, 4, 4, 6], 8, [0, 4, 6], id="repeated_offset"),
        pytest.param([0, 4, 12], 8, [0, 4], id="offset_after_the_last_token"),
        pytest.param([3, 6], 8, [0, 3, 6], id="no_offset_at_zero"),
    ],
)
def test_cu_seqlens_starts_mark_each_offset_and_the_first_position(offsets, num_tokens, expected):
    cu_seqlens = torch.tensor(offsets, dtype=torch.int32)
    starts = document_starts_from_cu_seqlens(cu_seqlens, num_tokens)
    assert starts.dtype == torch.bool
    assert torch.equal(starts, _row_starts(expected, 1, num_tokens))


@pytest.mark.parametrize("num_tokens", [11, 16])
def test_cu_seqlens_starts_agree_with_the_packed_sequence_index(num_tokens):
    cu_seqlens = torch.tensor([0, 5, 7, 11], dtype=torch.int32)
    seq_idx = PackedSeqParams(cu_seqlens_q=cu_seqlens, total_tokens=num_tokens).seq_idx
    expected = torch.ones(1, num_tokens, dtype=torch.bool)
    expected[:, 1:] = seq_idx[:, 1:] != seq_idx[:, :-1]
    assert torch.equal(document_starts_from_cu_seqlens(cu_seqlens, num_tokens), expected)


def test_prefix_mask_is_false_on_a_prefix_of_each_document():
    # Each row holds documents of 4, 3 and 1 tokens.
    starts = _row_starts([0, 4, 7], 200, 8)
    mask = sample_prefix_mask(starts, _generator())
    assert mask.shape == starts.shape
    assert mask.dtype == torch.bool
    assert not (mask & starts).any()
    # Inside a document, a True position is never followed by a False position.
    assert (starts[:, 1:] | ~mask[:, :-1] | mask[:, 1:]).all()
    # The prefix of the 4-token document takes each length from 1 to 4.
    assert set((~mask[:, :4]).sum(dim=1).tolist()) == {1, 2, 3, 4}


def test_prefix_mask_depends_only_on_the_generator_state():
    starts = _row_starts([0, 4, 7], 50, 8)
    first = sample_prefix_mask(starts, _generator(1))
    assert torch.equal(sample_prefix_mask(starts, _generator(1)), first)
    assert not torch.equal(sample_prefix_mask(starts, _generator(2)), first)
    # Without a generator argument, the default generator supplies the draws.
    torch.manual_seed(3)
    default = sample_prefix_mask(starts)
    torch.manual_seed(3)
    assert torch.equal(sample_prefix_mask(starts), default)


def test_select_takes_each_position_from_the_input_that_the_mask_names():
    token_embedding = _hidden_states().requires_grad_(True)
    fused = (-_hidden_states()).requires_grad_(True)
    mask = torch.tensor([[0, 1, 1, 0, 1], [0, 0, 1, 1, 1], [0, 0, 0, 0, 1]], dtype=torch.bool)
    selected = select_feedback_inputs(token_embedding, fused, mask)
    assert selected.shape == token_embedding.shape
    for batch in range(BATCH):
        for position in range(SEQ):
            source = fused if mask[batch, position] else token_embedding
            assert torch.equal(selected[position, batch], source[position, batch])
    selected.sum().backward()
    from_fused = mask.t().unsqueeze(-1).expand(SEQ, BATCH, HIDDEN)
    assert torch.equal(fused.grad, from_fused.float())
    assert torch.equal(token_embedding.grad, (~from_fused).float())


def test_input_functions_reject_wrong_layouts():
    hidden_states = _hidden_states()
    mask = torch.zeros(BATCH, SEQ, dtype=torch.bool)
    with pytest.raises(ValueError, match="shift_hidden_states"):
        shift_hidden_states(hidden_states[0])
    with pytest.raises(ValueError, match="document_starts_from_eod"):
        document_starts_from_eod(torch.tensor([1, EOD, 2]), EOD)
    with pytest.raises(ValueError, match="document_starts_from_cu_seqlens"):
        document_starts_from_cu_seqlens(torch.tensor([[0, 4, 8]]), 8)
    with pytest.raises(ValueError, match="sample_prefix_mask"):
        sample_prefix_mask(mask[0])
    with pytest.raises(ValueError, match="select_feedback_inputs"):
        select_feedback_inputs(hidden_states, hidden_states[:-1], mask)
    with pytest.raises(ValueError, match="select_feedback_inputs"):
        select_feedback_inputs(hidden_states, hidden_states, mask.t())


class TestFeedbackInputsWithFusion:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        torch.manual_seed(0)
        config = TransformerConfig(
            num_layers=2,
            hidden_size=FUSION_HIDDEN,
            num_attention_heads=4,
            use_cpu_initialization=True,
        )
        self.fusion = build_module(
            gated_hidden_state_fusion_spec,
            config=config,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp"]),
        ).cuda()
        # Row 0 holds two documents; its first document ends with the EOD token at position 2.
        self.tokens = torch.tensor([[1, 2, EOD, 3, 4, 5], [6, 7, 8, 1, 2, 3]]).cuda()
        self.starts = document_starts_from_eod(self.tokens, EOD)
        self.token_embedding = torch.randn(6, 2, FUSION_HIDDEN).cuda()
        self.hidden_states = torch.randn(6, 2, FUSION_HIDDEN).cuda()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _decoder_input(self, hidden_states, feedback_mask):
        """Return the decoder input of one feedback pass."""

        fused = self.fusion(self.token_embedding, shift_hidden_states(hidden_states))
        return select_feedback_inputs(self.token_embedding, fused, feedback_mask)

    def _changed(self, first, second):
        """Return the ``[sequence, batch]`` positions where two decoder inputs differ."""

        return (first != second).any(dim=-1)

    def test_a_state_changes_only_the_next_position_of_its_batch_entry(self):
        changed_states = self.hidden_states.clone()
        changed_states[0, 0] += 1.0
        reference = self._decoder_input(self.hidden_states, ~self.starts)
        changed = self._decoder_input(changed_states, ~self.starts)
        expected = torch.zeros(6, 2, dtype=torch.bool).cuda()
        expected[1, 0] = True
        assert torch.equal(self._changed(reference, changed), expected)

    def test_a_document_start_ignores_the_last_state_of_the_previous_document(self):
        changed_states = self.hidden_states.clone()
        changed_states[2, 0] += 1.0
        reference = self._decoder_input(self.hidden_states, ~self.starts)
        changed = self._decoder_input(changed_states, ~self.starts)
        assert torch.equal(reference[3, 0], self.token_embedding[3, 0])
        assert not self._changed(reference, changed).any()
        # Without the document starts, the same state reaches the next document.
        no_starts = torch.ones_like(self.starts)
        leaked = self._changed(
            self._decoder_input(self.hidden_states, no_starts),
            self._decoder_input(changed_states, no_starts),
        )
        assert leaked[3, 0]

    def test_prefix_positions_keep_the_plain_token_embedding(self):
        mask = sample_prefix_mask(self.starts)
        decoder_input = self._decoder_input(self.hidden_states, mask)
        assert torch.equal(self._changed(decoder_input, self.token_embedding), mask.t())
