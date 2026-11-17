# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Preparation of the previous hidden state for hidden-state fusion."""

import torch
import torch.nn.functional as F
from torch import Tensor


def shift_hidden_states(hidden_states: Tensor) -> Tensor:
    """Shift ``[sequence, batch, hidden]`` states one position later and zero position 0."""

    if hidden_states.ndim != 3:
        raise ValueError(
            "shift_hidden_states expected [sequence, batch, hidden] states, "
            f"got {tuple(hidden_states.shape)}."
        )
    shifted = torch.roll(hidden_states, shifts=1, dims=0)
    shifted.select(0, 0).zero_()
    return shifted


def document_starts_from_eod(tokens: Tensor, eod_token_id: int) -> Tensor:
    """Return a bool mask that is True at position 0 of each row and after each EOD token."""

    if tokens.ndim != 2:
        raise ValueError(
            "document_starts_from_eod expected [batch, sequence] token ids, "
            f"got {tuple(tokens.shape)}."
        )
    starts = torch.zeros_like(tokens, dtype=torch.bool)
    starts[:, 0] = True
    starts[:, 1:] = tokens[:, :-1] == eod_token_id
    return starts


def document_starts_from_cu_seqlens(cu_seqlens: Tensor, num_tokens: int) -> Tensor:
    """Return a ``[1, num_tokens]`` bool mask of the document starts of one packed row.

    Position 0 is always a start, and tokens after the last offset form one more document.
    """

    if cu_seqlens.ndim != 1:
        raise ValueError(
            "document_starts_from_cu_seqlens expected one-dimensional offsets, "
            f"got {tuple(cu_seqlens.shape)}."
        )
    positions = torch.arange(num_tokens, dtype=cu_seqlens.dtype, device=cu_seqlens.device)
    starts = torch.isin(positions, cu_seqlens).unsqueeze(0)
    starts[:, 0] = True
    return starts


def sample_prefix_mask(document_starts: Tensor, generator: torch.Generator | None = None) -> Tensor:
    """Return a bool mask that is False on a random prefix of each document.

    The prefix length is uniform in ``1..n`` for a document of ``n`` tokens. This is the prefix
    mixin of the Full-bandwidth Transformer (https://arxiv.org/abs/2608.08888).
    """

    if document_starts.ndim != 2:
        raise ValueError(
            "sample_prefix_mask expected a [batch, sequence] mask, "
            f"got {tuple(document_starts.shape)}."
        )
    batch, sequence = document_starts.shape
    device = document_starts.device
    positions = torch.arange(sequence, device=device).expand(batch, sequence)
    document_start = torch.where(document_starts, positions, 0).cummax(dim=1).values
    next_start = torch.where(document_starts, positions, sequence)[:, 1:]
    next_start = F.pad(next_start, (0, 1), value=sequence)
    document_end = next_start.flip(1).cummin(dim=1).values.flip(1)
    # Every token reads the draw at its document start, so one draw serves one document.
    uniform = torch.rand(batch, sequence, generator=generator, device=device)
    uniform = uniform.gather(1, document_start)
    prefix_length = (uniform * (document_end - document_start)).long() + 1
    return positions - document_start >= prefix_length


def select_feedback_inputs(token_embedding: Tensor, fused: Tensor, feedback_mask: Tensor) -> Tensor:
    """Select ``fused`` where the ``[batch, sequence]`` mask is True, else ``token_embedding``."""

    if token_embedding.ndim != 3 or token_embedding.shape != fused.shape:
        raise ValueError(
            "select_feedback_inputs expected two [sequence, batch, hidden] inputs of one shape, "
            f"got {tuple(token_embedding.shape)} and {tuple(fused.shape)}."
        )
    sequence, batch, _ = token_embedding.shape
    if feedback_mask.shape != (batch, sequence):
        raise ValueError(
            f"select_feedback_inputs expected a [{batch}, {sequence}] mask, "
            f"got {tuple(feedback_mask.shape)}."
        )
    return torch.where(feedback_mask.t().unsqueeze(-1), fused, token_embedding)
