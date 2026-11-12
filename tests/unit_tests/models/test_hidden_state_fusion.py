# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for the hidden-state fusion modules and their specs."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.models.common.hidden_state_feedback import (
    AdditiveHiddenStateFusion,
    GatedHiddenStateFusion,
    additive_hidden_state_fusion_spec,
    gated_hidden_state_fusion_spec,
)
from megatron.core.ops.norm.torch_norm import L2Norm
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

SEQ, BATCH, HIDDEN = 5, 3, 16
SPECS = {"additive": additive_hidden_state_fusion_spec, "gated": gated_hidden_state_fusion_spec}


def _reference_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference RMS norm without learned weights."""

    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class TestHiddenStateFusion:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        torch.manual_seed(0)
        self.config = TransformerConfig(
            num_layers=2, hidden_size=HIDDEN, num_attention_heads=4, use_cpu_initialization=True
        )
        self.token_embedding = torch.randn(SEQ, BATCH, HIDDEN).cuda()
        self.previous_hidden = torch.randn(SEQ, BATCH, HIDDEN).cuda()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _build(self, fusion: str, config: TransformerConfig | None = None, **kwargs):
        return build_module(SPECS[fusion], config=config or self.config, **kwargs).cuda()

    def test_specs_build_the_documented_modules(self):
        additive = self._build("additive")
        gated = self._build("gated", name="hidden_state_feedback.fusion")
        assert isinstance(additive, AdditiveHiddenStateFusion)
        assert isinstance(gated, GatedHiddenStateFusion)
        assert isinstance(additive.hidden_state_norm, L2Norm)
        assert isinstance(gated.token_embedding_norm, L2Norm)
        assert isinstance(gated.output_norm, L2Norm)
        assert isinstance(additive.hidden_state_projection, TELinear)
        assert isinstance(gated.hidden_state_projection, TELinear)
        assert isinstance(gated.gate_projection, TELinear)
        assert sum(p.numel() for p in additive.parameters()) == HIDDEN * HIDDEN
        assert sum(p.numel() for p in gated.parameters()) == 2 * HIDDEN * HIDDEN
        # The projection weights are replicated across tensor-parallel ranks.
        for parameter in [*additive.parameters(), *gated.parameters()]:
            assert parameter.tensor_model_parallel is False
            assert parameter.allreduce is True

    def test_additive_module_is_the_identity_at_init_and_still_receives_gradient(self):
        fusion = self._build("additive")
        assert torch.count_nonzero(fusion.hidden_state_projection.weight) == 0
        fused = fusion(self.token_embedding, self.previous_hidden)
        assert torch.equal(fused, self.token_embedding)
        fused.sum().backward()
        gradient = fusion.hidden_state_projection.weight.grad
        assert gradient is not None and torch.count_nonzero(gradient) > 0

    def test_additive_matches_the_equation(self):
        fusion = self._build("additive")
        torch.nn.init.normal_(fusion.hidden_state_projection.weight)
        eps = self.config.layernorm_epsilon
        reference = self.token_embedding + F.linear(
            _reference_rms_norm(self.previous_hidden, eps), fusion.hidden_state_projection.weight
        )
        fused = fusion(self.token_embedding, self.previous_hidden)
        torch.testing.assert_close(fused, reference, rtol=1e-5, atol=1e-5)

    def test_gated_matches_the_equation(self):
        fusion = self._build("gated")
        eps = self.config.layernorm_epsilon
        value = F.linear(self.previous_hidden, fusion.hidden_state_projection.weight)
        gate = torch.sigmoid(
            F.linear(_reference_rms_norm(self.token_embedding, eps), fusion.gate_projection.weight)
        )
        reference = _reference_rms_norm(value * gate, eps)
        fused = fusion(self.token_embedding, self.previous_hidden)
        torch.testing.assert_close(fused, reference, rtol=1e-5, atol=1e-5)

    def test_gated_projections_use_the_config_init_method(self):
        fusion = self._build("gated")
        for weight in (fusion.hidden_state_projection.weight, fusion.gate_projection.weight):
            assert torch.count_nonzero(weight) > 0
            # The default init_method is a zero-mean normal with standard deviation 0.02.
            assert 0.01 < weight.float().std().item() < 0.03

    @pytest.mark.parametrize("name", list(SPECS))
    def test_rejects_mismatched_shapes(self, name):
        fusion = self._build(name)
        with pytest.raises(ValueError):
            fusion(self.token_embedding, self.previous_hidden[:-1])
        with pytest.raises(ValueError):
            fusion(self.token_embedding[..., :-1], self.previous_hidden[..., :-1])

    @pytest.mark.parametrize("name", list(SPECS))
    def test_gradients_reach_both_inputs_and_the_weights(self, name):
        fusion = self._build(name)
        if name == "additive":
            torch.nn.init.normal_(fusion.hidden_state_projection.weight)
        token_embedding = self.token_embedding.clone().requires_grad_(True)
        previous_hidden = self.previous_hidden.clone().requires_grad_(True)
        fusion(token_embedding, previous_hidden).sum().backward()
        assert token_embedding.grad is not None and torch.count_nonzero(token_embedding.grad) > 0
        assert previous_hidden.grad is not None and torch.count_nonzero(previous_hidden.grad) > 0
        for parameter in fusion.parameters():
            assert parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0

    @pytest.mark.parametrize("name", list(SPECS))
    def test_sharded_state_dict_names_the_projections(self, name):
        fusion = self._build(name)
        sharded = fusion.sharded_state_dict(prefix="fusion.")
        assert "fusion.hidden_state_projection.weight" in sharded
        if name == "gated":
            assert "fusion.gate_projection.weight" in sharded
        assert all(key.startswith("fusion.") for key in sharded)

    @pytest.mark.parametrize("name", list(SPECS))
    def test_bf16_forward(self, name):
        config = TransformerConfig(
            num_layers=2,
            hidden_size=HIDDEN,
            num_attention_heads=4,
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        fusion = self._build(name, config)
        token_embedding = self.token_embedding.to(torch.bfloat16)
        previous_hidden = self.previous_hidden.to(torch.bfloat16)
        fused = fusion(token_embedding, previous_hidden)
        assert fused.shape == token_embedding.shape
        assert fused.dtype == torch.bfloat16
        assert torch.isfinite(fused).all()
