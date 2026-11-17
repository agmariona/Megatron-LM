# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Configuration of hidden-state feedback."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class HiddenStateFeedbackConfig:
    """Configuration for feedback of the previous position's final hidden state.

    A fusion module combines each token embedding with the final hidden state of the
    previous position before the decoder runs.
    """

    fusion: Literal["additive", "gated"] = "additive"
    """Fusion that combines the token embedding with the previous hidden state."""

    def __post_init__(self) -> None:
        if self.fusion not in ("additive", "gated"):
            raise ValueError(
                f"hidden state feedback fusion must be 'additive' or 'gated', got {self.fusion!r}."
            )
