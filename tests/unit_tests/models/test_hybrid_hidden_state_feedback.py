# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for the construction of hidden-state feedback in HybridModel."""

import pytest

from megatron.core.models.hybrid.hidden_state_feedback.fusion import (
    AdditiveHiddenStateFusion,
    GatedHiddenStateFusion,
)
from megatron.core.models.hybrid.hidden_state_feedback.resolve import (
    resolve_hidden_state_fusion_spec,
)
from megatron.core.models.hybrid.hidden_state_feedback.specs import gated_hidden_state_fusion_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import HiddenStateFeedbackConfig, TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(autouse=True)
def model_parallel():
    """Initialize the process groups that the model and the fusion projections use."""
    Utils.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    yield
    Utils.destroy_model_parallel()


def _model(feedback=None, **model_kwargs):
    """Return a small attention and MLP model with the given feedback config."""
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        use_cpu_initialization=True,
        hidden_state_feedback=feedback,
    )
    return HybridModel(
        config=config,
        vocab_size=100,
        max_sequence_length=8,
        hybrid_layer_pattern="*-",
        **model_kwargs,
    )


@pytest.mark.parametrize(
    ("fusion", "custom_spec", "module_type"),
    [
        pytest.param("additive", None, AdditiveHiddenStateFusion, id="additive"),
        pytest.param("gated", None, GatedHiddenStateFusion, id="gated"),
        pytest.param(
            "additive", gated_hidden_state_fusion_spec, GatedHiddenStateFusion, id="custom_spec"
        ),
    ],
)
def test_model_builds_the_configured_fusion(fusion, custom_spec, module_type):
    model = _model(HiddenStateFeedbackConfig(fusion=fusion), hidden_state_fusion_spec=custom_spec)
    assert type(model.hidden_state_fusion) is module_type


def test_feedback_adds_only_fusion_parameters():
    plain = _model()
    feedback = _model(HiddenStateFeedbackConfig())
    assert plain.hidden_state_fusion is None
    # A stage without the embedding has no fusion. The config rejects pipeline parallelism
    # with feedback for now, so only a direct caller builds such a stage.
    assert _model(HiddenStateFeedbackConfig(), pre_process=False).hidden_state_fusion is None
    plain_keys, feedback_keys = set(plain.state_dict()), set(feedback.state_dict())
    added = feedback_keys - plain_keys
    assert added and all(key.startswith("hidden_state_fusion.") for key in added)
    assert plain_keys <= feedback_keys
    # The distributed checkpoint must hold the same new parameters.
    sharded_added = set(feedback.sharded_state_dict()) - set(plain.sharded_state_dict())
    assert sharded_added == added


def test_feedback_construction_rejects_inconsistent_inputs():
    with pytest.raises(ValueError, match="hidden_state_fusion_spec"):
        _model(hidden_state_fusion_spec=gated_hidden_state_fusion_spec)
    # The config rejects unknown names, so bypass its validation to reach the resolver's guard.
    config = HiddenStateFeedbackConfig()
    config.fusion = "concatenated"
    with pytest.raises(ValueError, match="concatenated"):
        resolve_hidden_state_fusion_spec(config)
