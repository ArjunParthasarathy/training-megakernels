"""Qwen3 model — one shared eager implementation used by every variant.

All fusible ops go through ``megakernels.kernels`` so a variant swaps kernels just
by selecting a backend; the module graph is identical across baseline / modded /
custom_backward. This is deliberate: a comparison then isolates exactly what each
variant changes (optimizer, kernel backend, backward pass) and nothing else.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from . import kernels
from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return kernels.rms_norm(x, self.weight, self.eps)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.nh, self.nkv, self.hd = (
            cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim,
        )
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.q_dim, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.kv_dim, bias=cfg.attention_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.kv_dim, bias=cfg.attention_bias)
        self.o_proj = nn.Linear(cfg.q_dim, cfg.hidden_size, bias=cfg.attention_bias)
        if cfg.qk_norm:
            self.q_norm = nn.Parameter(torch.ones(cfg.head_dim))
            self.k_norm = nn.Parameter(torch.ones(cfg.head_dim))

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        if self.cfg.qk_norm:
            q, k = kernels.qk_norm(q, k, self.q_norm, self.k_norm, self.cfg.rms_norm_eps)
        q, k = kernels.apply_rope(q, k, cos, sin)
        o = kernels.attention(q, k, v, causal=True)           # [B, H, T, D]
        o = o.transpose(1, 2).reshape(B, T, self.cfg.q_dim)
        return self.o_proj(o)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(kernels.swiglu(self.gate_proj(x), self.up_proj(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        if cfg.tie_word_embeddings:
            self.lm_head = None                                # ties to embed_tokens.weight
        else:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    @property
    def head_weight(self) -> Tensor:
        return self.embed_tokens.weight if self.lm_head is None else self.lm_head.weight

    def backbone(self, input_ids: Tensor) -> Tensor:
        B, T = input_ids.shape
        x = self.embed_tokens(input_ids)
        pos = torch.arange(T, device=input_ids.device)
        cos, sin = kernels.rotary_cos_sin(
            pos, self.cfg.head_dim, self.cfg.rope_theta, x.dtype, x.device)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.norm(x)

    def forward(self, input_ids: Tensor, labels: Tensor | None = None,
                fused_ce: bool = False) -> dict:
        """Returns {'loss': Tensor|None, 'logits': Tensor|None}.

        fused_ce=True (modded) routes through kernels.linear_cross_entropy and
        never materializes the [N, vocab] logits. fused_ce=False (baseline)
        computes explicit logits + F.cross_entropy.
        """
        h = self.backbone(input_ids)
        if labels is None:
            logits = F.linear(h, self.head_weight)
            return {"loss": None, "logits": logits}

        # next-token shift
        h_shift = h[:, :-1].reshape(-1, self.cfg.hidden_size)
        tgt = labels[:, 1:].reshape(-1)
        if fused_ce:
            loss = kernels.linear_cross_entropy(h_shift, self.head_weight, tgt)
            return {"loss": loss, "logits": None}
        logits = F.linear(h_shift, self.head_weight)
        loss = F.cross_entropy(logits, tgt)
        return {"loss": loss, "logits": logits}

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed_tokens.weight.numel()
            if self.lm_head is not None:
                n -= self.lm_head.weight.numel()
        return n
