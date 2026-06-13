"""Model + run configuration shared by every training variant.

`ModelConfig` is the architecture (identical across baseline / modded / custom-
backward, so a comparison isolates *only* what each variant changes). `RunConfig`
is the training knobs. Variants differ in their optimizer + kernel backend +
backward pass, never in the architecture here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # Qwen3-1.7B (huggingface.co/Qwen/Qwen3-1.7B/config.json)
    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 6144          # SwiGLU
    num_hidden_layers: int = 28
    num_attention_heads: int = 16          # query heads
    num_key_value_heads: int = 8           # GQA 2:1
    head_dim: int = 128
    max_position_embeddings: int = 40960
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    qk_norm: bool = True                   # Qwen3: per-head RMSNorm on Q/K before RoPE
    tie_word_embeddings: bool = True
    attention_bias: bool = False

    @property
    def q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim


def qwen3_1p7b() -> ModelConfig:
    return ModelConfig()


def tiny() -> ModelConfig:
    """Small config for CPU tests / pipeline smoke tests."""
    return ModelConfig(
        vocab_size=4096,
        hidden_size=256,
        intermediate_size=768,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,
        max_position_embeddings=2048,
    )


@dataclass
class RunConfig:
    variant: str = "baseline"              # baseline | modded | custom_backward
    kernels: str = "auto"                  # auto | eager | cute  (kernel backend)
    seq_len: int = 2048
    batch_size: int = 4
    max_steps: int = 64
    lr: float = 3e-4
    muon_lr: float = 0.02
    weight_decay: float = 0.1
    warmup_steps: int = 10
    seed: int = 0
    compile: bool = False
    tiny: bool = False
    log_path: str = ""                     # jsonl per-step metrics; empty = stdout only
    model: ModelConfig = field(default_factory=qwen3_1p7b)
