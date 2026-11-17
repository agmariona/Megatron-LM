# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Selection of the hidden-state fusion spec.

The selection by config is kept out of ``specs.py``, so that each spec there stays one
static behavior, and out of the generic ``HybridModel`` code, which only calls it at
model construction.
"""

from megatron.core.models.hybrid.hidden_state_feedback.specs import (
    additive_hidden_state_fusion_spec,
    gated_hidden_state_fusion_spec,
)
from megatron.core.transformer.hidden_state_feedback_config import HiddenStateFeedbackConfig
from megatron.core.transformer.spec_utils import ModuleSpec


def resolve_hidden_state_fusion_spec(config: HiddenStateFeedbackConfig) -> ModuleSpec:
    """Return the fusion spec that the feedback config names."""

    if config.fusion == "additive":
        return additive_hidden_state_fusion_spec
    if config.fusion == "gated":
        return gated_hidden_state_fusion_spec
    # The config validates the same names, so this only guards against the two lists diverging.
    raise ValueError(f"hidden state feedback fusion {config.fusion!r} has no spec.")
