# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for the hidden-state feedback configuration."""

from typing import Any

import pytest
import torch

from megatron.core.transformer import HiddenStateFeedbackConfig, TransformerConfig


def _config(**overrides: Any) -> TransformerConfig:
    """Return a small config with the given overrides."""

    values: dict[str, Any] = dict(num_layers=1, hidden_size=8, num_attention_heads=2)
    values.update(overrides)
    return TransformerConfig(**values)


def test_feedback_accepts_tensor_parallelism_without_sequence_parallelism():
    # Each rank keeps the full sequence, so the previous position is local.
    _config(hidden_state_feedback=HiddenStateFeedbackConfig(), tensor_model_parallel_size=2)


def test_feedback_config_rejects_an_unknown_fusion():
    with pytest.raises(ValueError, match="fusion"):
        HiddenStateFeedbackConfig(fusion="concatenated")


@pytest.mark.parametrize(
    ("override", "expected_error"),
    [
        ({"context_parallel_size": 2}, "context_parallel_size"),
        (
            {"num_layers": 2, "pipeline_model_parallel_size": 2, "pipeline_dtype": torch.bfloat16},
            "pipeline_model_parallel_size",
        ),
        ({"sequence_parallel": True, "tensor_model_parallel_size": 2}, "sequence_parallel"),
        (
            {"sequence_packing_scheduler": "dp_balanced", "max_seqlen_per_dp_cp_rank": 64},
            "sequence_packing_scheduler",
        ),
        (
            {
                "dynamic_context_parallel": True,
                "max_seqlen_per_dp_cp_rank": 64,
                "calculate_per_token_loss": True,
            },
            "dynamic_context_parallel",
        ),
    ],
)
def test_transformer_config_rejects_unsupported_feedback_modes(override, expected_error):
    with pytest.raises(NotImplementedError, match=expected_error):
        _config(hidden_state_feedback=HiddenStateFeedbackConfig(), **override)
