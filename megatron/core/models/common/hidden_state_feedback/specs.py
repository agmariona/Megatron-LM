# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Module specs for hidden-state fusion."""

from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.models.common.hidden_state_feedback.fusion import (
    AdditiveHiddenStateFusion,
    AdditiveHiddenStateFusionSubmodules,
    GatedHiddenStateFusion,
    GatedHiddenStateFusionSubmodules,
)
from megatron.core.ops.norm.torch_norm import L2Norm
from megatron.core.transformer.spec_utils import ModuleSpec

# L2Norm is the RMS norm without learned weights.
additive_hidden_state_fusion_spec = ModuleSpec(
    module=AdditiveHiddenStateFusion,
    submodules=AdditiveHiddenStateFusionSubmodules(
        hidden_state_norm=L2Norm, hidden_state_projection=TELinear
    ),
)

gated_hidden_state_fusion_spec = ModuleSpec(
    module=GatedHiddenStateFusion,
    submodules=GatedHiddenStateFusionSubmodules(
        token_embedding_norm=L2Norm,
        output_norm=L2Norm,
        hidden_state_projection=TELinear,
        gate_projection=TELinear,
    ),
)
