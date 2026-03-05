"""Reservoir computing sub-package.

Public API
----------
ESN               : Echo State Network with sparse topology.
MultiReservoir    : Dual fast/slow timescale reservoir pair.
MultiReservoirConfig : Configuration dataclass for MultiReservoir.
ReadProjection    : Trainable projection from reservoir state to LLM hidden dim.
WriteHead         : Trainable projection from LLM hidden state to reservoir input.
CrossAttentionSidecar : Multi-head cross-attention sidecar (reservoir K/V × LLM Q).
FiLMModulation    : FiLM-style gated residual modulation by reservoir state.
"""

from src.reservoir.esn import ESN
from src.reservoir.interface import (
    CrossAttentionSidecar,
    FiLMModulation,
    ReadProjection,
    WriteHead,
)
from src.reservoir.multi_reservoir import MultiReservoir, MultiReservoirConfig

__all__ = [
    "ESN",
    "MultiReservoir",
    "MultiReservoirConfig",
    "ReadProjection",
    "WriteHead",
    "CrossAttentionSidecar",
    "FiLMModulation",
]
