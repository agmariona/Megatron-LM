# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Feedback of the previous position's final hidden state into the decoder input."""

from megatron.core.models.common.hidden_state_feedback.fusion import (
    AdditiveHiddenStateFusion,
    AdditiveHiddenStateFusionSubmodules,
    GatedHiddenStateFusion,
    GatedHiddenStateFusionSubmodules,
    HiddenStateFusion,
)
from megatron.core.models.common.hidden_state_feedback.inputs import (
    document_starts_from_cu_seqlens,
    document_starts_from_eod,
    sample_prefix_mask,
    select_feedback_inputs,
    shift_hidden_states,
)
from megatron.core.models.common.hidden_state_feedback.specs import (
    additive_hidden_state_fusion_spec,
    gated_hidden_state_fusion_spec,
)

__all__ = [
    "AdditiveHiddenStateFusion",
    "AdditiveHiddenStateFusionSubmodules",
    "GatedHiddenStateFusion",
    "GatedHiddenStateFusionSubmodules",
    "HiddenStateFusion",
    "additive_hidden_state_fusion_spec",
    "document_starts_from_cu_seqlens",
    "document_starts_from_eod",
    "gated_hidden_state_fusion_spec",
    "sample_prefix_mask",
    "select_feedback_inputs",
    "shift_hidden_states",
]
