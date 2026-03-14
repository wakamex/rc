"""Read/Write interface & cross-attention sidecar for ESN ↔ LLM bridge.

Modules:
- ReadProjection: reservoir state → LLM hidden dim (trainable linear projection)
- WriteHead: LLM hidden state → reservoir input dim (trainable linear projection)
- CrossAttentionSidecar: reservoir K/V × LLM hidden Q (multi-head cross-attention)
- FiLMModulation: FiLM-style gated residual modulation

Gradient boundary:
- Reservoir weights (W, W_in) are frozen numpy arrays → no gradient.
- Interface modules are trainable PyTorch nn.Modules.
- Gradient stops at the numpy ↔ torch boundary automatically (from_numpy / detach).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_tensor(
    x: np.ndarray | torch.Tensor,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert numpy array or tensor to a detached tensor.

    Gradient stops here: numpy arrays carry no gradient tracking, and any
    incoming torch.Tensor is explicitly detached so reservoir state updates
    cannot leak gradients into reservoir weights.
    """
    if isinstance(x, torch.Tensor):
        t = x.detach()
    else:
        t = torch.from_numpy(np.asarray(x, dtype=np.float32))
    if device is not None:
        t = t.to(device)
    return t.to(dtype)


class ReadProjection(nn.Module):
    """Project reservoir state to LLM hidden dimension.

    Trainable linear projection: reservoir_dim → hidden_dim.
    Gradient flows through this module's parameters but not into the reservoir.

    Args:
        reservoir_dim: Reservoir state size (ESN ``n``).
        hidden_dim: LLM hidden dimension.
        bias: Whether to include a bias term.
    """

    def __init__(self, reservoir_dim: int, hidden_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.reservoir_dim = reservoir_dim
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(reservoir_dim, hidden_dim, bias=bias)

    def forward(
        self,
        reservoir_state: np.ndarray | torch.Tensor,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Project reservoir state to LLM hidden dimension.

        Args:
            reservoir_state: shape ``(reservoir_dim,)`` or ``(batch, reservoir_dim)``.
            device: target device (defaults to the projection weight's device).

        Returns:
            Tensor of shape ``(hidden_dim,)`` or ``(batch, hidden_dim)``.
        """
        if device is None:
            device = self.proj.weight.device
        t = _to_tensor(reservoir_state, device=device, dtype=self.proj.weight.dtype)
        return self.proj(t)


class WriteHead(nn.Module):
    """Project LLM hidden state to reservoir input dimension.

    Trainable linear projection: hidden_dim → input_dim.
    Use :meth:`to_numpy` to obtain a detached numpy array for ``ESN.step()``.

    Args:
        hidden_dim: LLM hidden dimension.
        input_dim: Reservoir input dimension (ESN ``input_dim``).
        bias: Whether to include a bias term.
    """

    def __init__(self, hidden_dim: int, input_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.proj = nn.Linear(hidden_dim, input_dim, bias=bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project LLM hidden state to reservoir input dimension.

        Args:
            hidden: shape ``(..., hidden_dim)``.

        Returns:
            Tensor of shape ``(..., input_dim)``.
            Detach before passing to ``ESN.step()`` to stop gradients.
        """
        return self.proj(hidden)

    def to_numpy(self, hidden: torch.Tensor) -> np.ndarray:
        """Project and convert to numpy for direct use with ``ESN.step()``.

        Gradient stops here (detach + numpy conversion).

        Args:
            hidden: shape ``(..., hidden_dim)``.

        Returns:
            numpy float32 array of shape ``(..., input_dim)``.
        """
        with torch.no_grad():
            out = self.proj(hidden)
        return out.detach().cpu().numpy().astype(np.float32)


class CrossAttentionSidecar(nn.Module):
    """Cross-attention sidecar: LLM hidden (Q) × reservoir states (K, V).

    Reservoir states serve as keys and values; LLM hidden states as queries.
    Inserted at configurable transformer layers to let the model attend over
    the reservoir's temporal memory.

    Gradient flows through Q/K/V/out projections and layer norms.
    Gradient stops at reservoir states (numpy → tensor boundary).

    Args:
        hidden_dim: LLM hidden dimension (must be divisible by ``num_heads``).
        reservoir_dim: Reservoir state dimension.
        num_heads: Number of attention heads.
        dropout: Attention dropout probability.
        bias: Whether to use bias in projections.
    """

    def __init__(
        self,
        hidden_dim: int,
        reservoir_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.hidden_dim = hidden_dim
        self.reservoir_dim = reservoir_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Layer norm applied to queries (LLM hidden) before attention
        self.q_norm = nn.LayerNorm(hidden_dim)

        # Q from LLM hidden, K/V from reservoir states
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.k_proj = nn.Linear(reservoir_dim, hidden_dim, bias=bias)
        self.v_proj = nn.Linear(reservoir_dim, hidden_dim, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

        # Flamingo-style tanh gate (Alayrac et al. NeurIPS 2022).
        # alpha=0 → tanh(0)=0 (identity at init, no noise injection).
        # tanh'(0)=1 → perfect gradient flow from step 1.
        # tanh bounds output to [-1,1] → prevents explosion.
        self.gate_alpha = nn.Parameter(torch.tensor([gate_init]))

        # Post-norm for residual connection
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        reservoir_states: np.ndarray | torch.Tensor,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Apply cross-attention with reservoir states as K/V.

        Args:
            hidden: LLM hidden states.
                    Shape ``(batch, seq_len, hidden_dim)`` or ``(seq_len, hidden_dim)``.
            reservoir_states: Reservoir state sequence.
                    Shape ``(batch, res_seq, reservoir_dim)``,
                    ``(res_seq, reservoir_dim)``, or ``(reservoir_dim,)``.
            device: target device (defaults to ``hidden.device``).

        Returns:
            Updated hidden states, same shape as ``hidden``.
        """
        squeeze_batch = hidden.ndim == 2
        if squeeze_batch:
            hidden = hidden.unsqueeze(0)  # (1, T, H)

        B, T, H = hidden.shape

        # Convert reservoir states — gradient stops here
        r = _to_tensor(reservoir_states, device=hidden.device, dtype=hidden.dtype)

        # Ensure shape is (B, S, reservoir_dim)
        if r.ndim == 1:
            r = r.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        elif r.ndim == 2:
            r = r.unsqueeze(0).expand(B, -1, -1)
        S = r.shape[1]

        # Compute Q from normalized LLM hidden, K/V from reservoir
        Q = self.q_proj(self.q_norm(hidden))  # (B, T, H)
        K = self.k_proj(r)                    # (B, S, H)
        V = self.v_proj(r)                    # (B, S, H)

        # Reshape to (B, num_heads, seq, head_dim)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, heads, T, S)

        # Causal mask: position t can only attend to reservoir states at positions <= t.
        # Without this, queries can see future reservoir states (causal leakage).
        if S > 1:
            causal_mask = torch.ones(T, S, device=attn.device, dtype=torch.bool).triu(diagonal=1)
            attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)  # (B, heads, T, head_dim)

        # Merge heads → (B, T, H) and project
        out = out.transpose(1, 2).contiguous().view(B, T, H)
        out = self.out_proj(out)
        out = self.out_norm(out)

        # Flamingo-style tanh-gated residual: tanh(0)=0 → identity at init,
        # tanh'(0)=1 → full gradient flow, tanh bounds to [-1,1].
        out = hidden + torch.tanh(self.gate_alpha) * out

        if squeeze_batch:
            out = out.squeeze(0)
        return out


class GatedLinearSidecar(nn.Module):
    """Per-token gated linear projection of reservoir states.

    At each position t, projects reservoir_state[t] (which encodes input history
    0..t) and adds to hidden with a tanh gate.  Inherently causal — no mask needed.

    Lighter than CrossAttentionSidecar: no Q/K/V projections, no attention weights.

    Args:
        hidden_dim: LLM hidden dimension.
        reservoir_dim: Reservoir state dimension.
        gate_init: Initial value for tanh gate parameter.
        use_delta: If True, concatenate state deltas (state[t] - state[t-1]) as
                   additional features, doubling the input to the projection.
    """

    def __init__(
        self,
        hidden_dim: int,
        reservoir_dim: int,
        gate_init: float = 0.0,
        use_delta: bool = False,
        proj_hidden: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.reservoir_dim = reservoir_dim
        self.use_delta = use_delta
        input_dim = reservoir_dim * 2 if use_delta else reservoir_dim
        if proj_hidden > 0:
            self.proj = nn.Sequential(
                nn.Linear(input_dim, proj_hidden, bias=True),
                nn.GELU(),
                nn.Linear(proj_hidden, hidden_dim, bias=True),
            )
        else:
            self.proj = nn.Linear(input_dim, hidden_dim, bias=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.gate_alpha = nn.Parameter(torch.tensor([gate_init]))

    def forward(
        self,
        hidden: torch.Tensor,
        reservoir_states: np.ndarray | torch.Tensor,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Apply gated linear injection from reservoir states.

        Args:
            hidden: LLM hidden states, shape (batch, seq_len, hidden_dim) or (seq_len, hidden_dim).
            reservoir_states: shape (batch, seq_len, reservoir_dim) or (seq_len, reservoir_dim).
            device: target device.

        Returns:
            Updated hidden states, same shape as input.
        """
        squeeze_batch = hidden.ndim == 2
        if squeeze_batch:
            hidden = hidden.unsqueeze(0)

        r = _to_tensor(reservoir_states, device=hidden.device, dtype=hidden.dtype)
        if r.ndim == 2:
            r = r.unsqueeze(0)

        # Align reservoir sequence length to hidden (truncate or pad)
        B, T, H = hidden.shape
        S = r.shape[1]
        if S > T:
            r = r[:, :T, :]
        elif S < T:
            pad = torch.zeros(B, T - S, r.shape[2], device=r.device, dtype=r.dtype)
            r = torch.cat([r, pad], dim=1)

        if self.use_delta:
            # Compute state deltas: delta[t] = state[t] - state[t-1], delta[0] = state[0]
            delta = r[:, 1:, :] - r[:, :-1, :]
            delta = torch.cat([r[:, :1, :], delta], dim=1)  # (B, T, reservoir_dim)
            r = torch.cat([r, delta], dim=-1)  # (B, T, 2*reservoir_dim)

        projected = self.proj(r)       # (B, T, H)
        projected = self.norm(projected)
        out = hidden + torch.tanh(self.gate_alpha) * projected

        if squeeze_batch:
            out = out.squeeze(0)
        return out


class FiLMModulation(nn.Module):
    """FiLM-style (Feature-wise Linear Modulation) gated residual injection.

    Computes:
        γ, β  = film_proj(reservoir_state)      # scale and shift
        gate  = sigmoid(gate_proj(reservoir_state))
        output = gate * (γ * norm(hidden) + β) + (1 − gate) * hidden

    The gated residual ensures near-identity behaviour when the reservoir
    signal is weak (gate → 0).

    Gradient flows through ``film_proj``, ``gate_proj``, and ``norm``.
    Gradient stops at ``reservoir_state`` (numpy → tensor boundary).

    Args:
        reservoir_dim: Reservoir state dimension.
        hidden_dim: LLM hidden dimension.
        bias: Whether to use bias in linear projections.
    """

    def __init__(self, reservoir_dim: int, hidden_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.reservoir_dim = reservoir_dim
        self.hidden_dim = hidden_dim

        # Projects reservoir → γ and β jointly
        self.film_proj = nn.Linear(reservoir_dim, 2 * hidden_dim, bias=bias)
        # Projects reservoir → element-wise gate
        self.gate_proj = nn.Linear(reservoir_dim, hidden_dim, bias=bias)
        # Layer norm applied to hidden before modulation
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden: torch.Tensor,
        reservoir_state: np.ndarray | torch.Tensor,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Apply FiLM-style gated modulation.

        Args:
            hidden: shape ``(..., hidden_dim)``.
            reservoir_state: shape ``(reservoir_dim,)`` or ``(batch, reservoir_dim)``.
            device: target device (defaults to ``hidden.device``).

        Returns:
            Modulated hidden states, same shape as ``hidden``.
        """
        r = _to_tensor(reservoir_state, device=hidden.device, dtype=hidden.dtype)

        # Project to γ, β, and gate
        film_out = self.film_proj(r)               # (..._r, 2*H)
        gamma, beta = film_out.chunk(2, dim=-1)    # each (..._r, H)
        gate = torch.sigmoid(self.gate_proj(r))    # (..._r, H)

        # Unsqueeze reservoir-derived tensors to broadcast over hidden's
        # middle (sequence) dimensions.  E.g. gamma (B, H) → (B, 1, H) for
        # hidden (B, T, H).
        while gamma.ndim < hidden.ndim:
            gamma = gamma.unsqueeze(-2)
            beta = beta.unsqueeze(-2)
            gate = gate.unsqueeze(-2)

        normed = self.norm(hidden)
        modulated = gamma * normed + beta

        return gate * modulated + (1.0 - gate) * hidden
