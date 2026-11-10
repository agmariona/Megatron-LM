# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Fusion of token embeddings with the previous position's final hidden state."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from megatron.core.ops.norm.torch_norm import LayerNormBuilder
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig


class HiddenStateFusion(MegatronModule, ABC):
    """Combine each token embedding with the previous position's final hidden state.

    Inputs and output have shape ``[sequence, batch, hidden]``. The caller shifts
    ``previous_hidden`` so that position ``t`` holds the state of position ``t - 1`` and
    masks sequence boundaries. Concrete fusions own their normalization and initialization.
    """

    def __init__(self, config: TransformerConfig, name: str | None = None) -> None:
        super().__init__(config=config)
        self.hidden_size = config.hidden_size
        self.name = name

    def _build_projection(
        self,
        linear: ModuleSpec | type,
        projection_name: str,
        init_method: Callable[[Tensor], Tensor | None],
    ) -> torch.nn.Module:
        """Build a bias-free square projection whose weight is replicated across TP ranks."""

        return build_module(
            linear,
            self.hidden_size,
            self.hidden_size,
            config=self.config,
            init_method=init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
            name=f"{self.name}.{projection_name}" if self.name is not None else None,
        )

    def _validate_inputs(self, token_embedding: Tensor, previous_hidden: Tensor) -> None:
        """Validate that both inputs share one ``[sequence, batch, hidden]`` shape."""

        if token_embedding.shape != previous_hidden.shape:
            raise ValueError(
                f"{type(self).__name__} expected inputs of one shape, got "
                f"{tuple(token_embedding.shape)} and {tuple(previous_hidden.shape)}."
            )
        if token_embedding.ndim != 3 or token_embedding.shape[-1] != self.hidden_size:
            raise ValueError(
                f"{type(self).__name__} expected [sequence, batch, {self.hidden_size}] inputs, "
                f"got {tuple(token_embedding.shape)}."
            )

    @abstractmethod
    def forward(self, token_embedding: Tensor, previous_hidden: Tensor) -> Tensor:
        """Return the fused decoder input for position-aligned embeddings and states."""


@dataclass
class AdditiveHiddenStateFusionSubmodules:
    """Module specs for the submodules of the additive fusion."""

    hidden_state_norm: LayerNormBuilder
    """Norm applied to the previous hidden state before its projection."""

    hidden_state_projection: ModuleSpec | type
    """Replicated linear that projects the normalized previous hidden state."""


class AdditiveHiddenStateFusion(HiddenStateFusion):
    """Add a projected previous hidden state to the token embedding.

    Computes ``e + W(norm(h))``, the adapter of Recycler:
    https://seed.bytedance.com/en/public_papers/recycler-cost-free-latent-reasoning-via-hidden-state-recycling
    ``W`` starts at zero, so a new module returns the token embedding unchanged.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: AdditiveHiddenStateFusionSubmodules,
        name: str | None = None,
    ) -> None:
        super().__init__(config=config, name=name)
        self.hidden_state_norm = submodules.hidden_state_norm(
            config=config, hidden_size=config.hidden_size, eps=config.layernorm_epsilon
        )
        self.hidden_state_projection = self._build_projection(
            submodules.hidden_state_projection, "hidden_state_projection", torch.nn.init.zeros_
        )

    def forward(self, token_embedding: Tensor, previous_hidden: Tensor) -> Tensor:
        """Return the token embedding plus the projected, normalized previous state."""

        self._validate_inputs(token_embedding, previous_hidden)
        update, _ = self.hidden_state_projection(self.hidden_state_norm(previous_hidden))
        return token_embedding + update


@dataclass
class GatedHiddenStateFusionSubmodules:
    """Module specs for the submodules of the gated fusion."""

    token_embedding_norm: LayerNormBuilder
    """Norm applied to the token embedding before the gate projection."""

    output_norm: LayerNormBuilder
    """Norm applied to the gated product before it enters the decoder."""

    hidden_state_projection: ModuleSpec | type
    """Replicated linear that projects the previous hidden state."""

    gate_projection: ModuleSpec | type
    """Replicated linear that projects the normalized token embedding into gate logits."""


class GatedHiddenStateFusion(HiddenStateFusion):
    """Gate a projected previous hidden state with the token embedding.

    Computes ``norm(W_h(h) * sigmoid(W_g(norm(e))))``, the gated fusion of the Full-bandwidth
    Transformer (https://arxiv.org/abs/2608.08888). The hidden state is the value path; the
    token embedding enters only through the gate. Both projections use ``config.init_method``.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: GatedHiddenStateFusionSubmodules,
        name: str | None = None,
    ) -> None:
        super().__init__(config=config, name=name)
        self.token_embedding_norm = submodules.token_embedding_norm(
            config=config, hidden_size=config.hidden_size, eps=config.layernorm_epsilon
        )
        self.output_norm = submodules.output_norm(
            config=config, hidden_size=config.hidden_size, eps=config.layernorm_epsilon
        )
        self.hidden_state_projection = self._build_projection(
            submodules.hidden_state_projection, "hidden_state_projection", config.init_method
        )
        self.gate_projection = self._build_projection(
            submodules.gate_projection, "gate_projection", config.init_method
        )

    def forward(self, token_embedding: Tensor, previous_hidden: Tensor) -> Tensor:
        """Return the normalized product of the projected state and the token gate."""

        self._validate_inputs(token_embedding, previous_hidden)
        value, _ = self.hidden_state_projection(previous_hidden)
        gate_logits, _ = self.gate_projection(self.token_embedding_norm(token_embedding))
        return self.output_norm(value * torch.sigmoid(gate_logits))
