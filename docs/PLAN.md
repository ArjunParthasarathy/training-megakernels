# Plan: CuTeDSL megakernels for a Qwen3-1.7B training loop

This is the Step-1 deliverable: where the optimized-Qwen3 / nanochat-speedrun
landscape stands, which kernels are worth writing in **CuTeDSL**, and how to graft
them into a candidate training repo. Profiler (Step 2) lives in `profiling/`;
Vast.ai launcher (Step 3) in `vast/`; both summarized in `../CLAUDE.md`.

---

## 0. Why CuTeDSL

CuTe DSL (CUTLASS 4.x, `pip install nvidia-cutlass-dsl`) is NVIDIA's Python-native
kernel DSL: same CuTe layout/tensor/atom abstractions as CUTLASS C++, JIT-compiled
(~100× faster compile than NVCC), perf "on par with C++", and — the reason it fits
this project — **one source compiles for Hopper (sm90) and Blackwell (sm100/sm120)**.
You profile on H100 today and the same kernels run on Blackwell later. Public beta
mid-2026 (graduating ~summer 2026); already shipping production kernels
(FlashAttention-4, QuACK, Tri Dao's Gram Newton-Schulz).

---

## 1. Baseline repos: what's already optimized

Two reference speedruns, both relevant because they define "the bar" a megakernel
must beat. The **math is already near-roofline**; the slack is in *HBM traffic and
redundant FLOPs across many small kernels* (launch overhead is mostly already paid
down by CUDA graphs under `torch.compile` — see §3).

### modded-nanoGPT (Keller Jordan) — GPT-2 124M → 3.28 val loss, 8×H100
Record went 45 min → **~1.33 min** by stacking: **Muon** optimizer (5-step bf16
Newton-Schulz orthogonalization) for hidden matrices + AdamW for embeds/scalars;
RoPE; **QK-norm**; **ReLU²** MLP; untied embeddings; tanh logit softcap; value
embeddings; U-net skip connections; **FlexAttention** (sliding-window + block-causal
+ document masking); bf16 activations + **fp8** head matmuls; **fused Triton**
linear-ReLU²-MLP and softcapped-CE kernels; trapezoidal LR; batch-size + seq-len
warmup; vocab padded to ×128; `torch.compile`; overlapped data loading + comms.

### nanochat (Karpathy) — full pipeline, "$100 / ~4h on 8×H100"
Borrows modded-nanoGPT pretraining but **cleaner / less exotic**: a single
`--depth` dial (d20 ≈ 560M is the speedrun), RoPE + RMSNorm + QK-norm + ReLU² +
untied embeds + softcap, **FlashAttention-3 (not FlexAttention)**, **Muon + AdamW**,
custom 65k BPE tokenizer, stages base→mid→SFT→RL. Deliberately **omits**
FlexAttention/value-embeds/U-net/fp8.

**Recommended baseline: nanochat.** It's the cleaner, more reproducible harness, it
already uses the Muon+AdamW split (which is where our best CuTeDSL win lives), and
its `--depth` dial lets us scale toward Qwen3-1.7B's shape. Keep modded-nanoGPT's
record log as the menu of tricks to optionally pull in.

> Note: "Qwen3-1.7B training loop" and "nanochat speedrun" are two axes of the same
> project — adopt **nanochat's training infrastructure** (Muon+AdamW, schedules,
> data pipeline) but with the **Qwen3-1.7B architecture** (§2). They share RoPE,
> RMSNorm, QK-norm, GQA, SwiGLU/ReLU²-family MLP, so the port is mechanical.

---

## 2. Qwen3-1.7B architecture (what we actually fuse)

From `huggingface.co/Qwen/Qwen3-1.7B/config.json`:

```
hidden=2048  layers=28  q_heads=16  kv_heads=8 (GQA 2:1)  head_dim=128
intermediate=6144 (SwiGLU, silu)   vocab=151936   RMSNorm eps=1e-6
RoPE theta=1e6   tie_word_embeddings=true   bf16
```

Distinctive: **QK-norm** = a per-head RMSNorm over the 128-dim applied to Q and K
*before RoPE* (replaces Qwen2's QKV bias). Two extra tiny norms per layer that
stock fused-kernel libraries (Liger) don't natively cover → a clean CuTeDSL target.
The **151936 vocab** makes the logits+cross-entropy the dominant activation-memory
cost → the other prime target.

---

## 3. Which kernels to write in CuTeDSL (prioritized by ROI)

Profiling rule of thumb confirmed by the research: **don't rewrite the big GEMMs or
attention** (cuBLAS/CUTLASS and FA3/FlexAttention are already ~85–100% of roofline).
Target the **memory-bound glue** and the **optimizer**.

> **Frame the win as HBM-traffic + redundant-FLOP elimination, not "kill kernel
> launches."** `torch.compile`'s `reduce-overhead` path wraps the step in **CUDA
> graphs** (CUDAGraph Trees), and both baselines use `torch.compile` — so most of
> the ~5µs/kernel host-side *launch* overhead is **already paid down** (graph replay
> is one `cudaGraphLaunch`, ~10× cheaper per kernel). What CUDA graphs do **not**
> fix: replaying N small kernels still does N HBM write-then-read round-trips of
> intermediates, and still pays redundant FLOPs. A megakernel that keeps data in
> SMEM/registers across fused steps removes exactly that — graph-immune. (Caveat:
> nanoGPT/nanochat often fall back to `max-autotune-no-cudagraphs` and hit graph
> **breaks** on dynamic shapes / the Muon NS loop / optimizer mutation, which
> re-expose eager launches at break boundaries — so collapsing an op-sequence into
> one node also removes a break point, a smaller secondary launch-side win.)

### P0 — Muon Newton-Schulz orthogonalization  ⭐ best-documented win
The Muon step orthogonalizes each 2D weight grad via a 5-step quintic Newton-Schulz
iteration: a chain of small/rectangular matmuls (`A=XXᵀ`, `B=bA+cA²`, `BX`) with a
kernel launch + HBM round-trip **between every step**. Prior art proves the gap:
- **Tri Dao, "Gram Newton-Schulz"** (tridao.me/blog/2026/gram-newton-schulz/) — uses
  **custom CuTeDSL symmetric-GEMM kernels** with a triangular scheduler: iterate on
  the small Gram matrix XXᵀ → **42–58% FLOP reduction** on rectangular weights,
  **25–29% speedup just from symmetric GEMM on Hopper**, up to **2×**. NS is 2–17% of
  total training time. **This is directly reusable / adaptable CuTeDSL code.**
- **Flash-Muon** (github.com/nil0x9/flash-muon) — CUDA, exploits XXᵀ symmetry, ~1.56×.

Our megakernel adds: **fuse all 5 NS iterations** (keep the Gram matrix in
shared/registers across steps, killing inter-step HBM + ~10 launches/param) and fuse
the Nesterov-momentum update. modded-nanoGPT/nanochat's stock NS does **neither**.

### P1 — Fused logits + cross-entropy (linear-CE) over 151936 vocab
Fuse the tied unembedding GEMM with log-sum-exp/CE so the `[seq×151936]` logits
never hit HBM (Apple cut-cross-entropy / Liger FLCE idea: 40–60% VRAM cut). CuTeDSL
fit: excellent — GEMM with a custom reduction epilogue, its sweet spot. **QuACK**
(`pip install quack-kernels`) already provides the CE primitive to build on. fwd+bwd.

### P2 — RMSNorm + residual (fwd+bwd)
28 layers × (2 block norms) + final norm, each a separate launch; fusing
norm+residual-add removes a full HBM round-trip. Memory-bound reduction = ideal
CuTeDSL. QuACK ships RMSNorm fwd+bwd; add the residual-add fusion.

### P3 — QK-norm + RoPE prologue (Qwen3-specific)
Fuse the two per-head RMSNorm(128) on Q/K with the RoPE rotation in one pass over
the QKV-projection output. Not covered off the shelf; distinctly Qwen3.

### P4 — Fused optimizer-step elementwise (AdamW + cautious weight decay)
Pure memory-bound elementwise over all params, currently one launch per op. Fuse
momentum/variance/bias-correction/weight-decay (+ the Muon scatter) into one kernel.

### P5 — Fused SwiGLU MLP
gate/up GEMMs → `silu(gate)*up` in the GEMM epilogue → down GEMM (Blockscaled FP8
possible on Blackwell). GEMM-epilogue fusion = CuTeDSL sweet spot.

**Reuse, don't rewrite:** attention via **FlashAttention-4** (`flash_attn.cute`,
already CuTeDSL, fwd+bwd, sm90+sm100); norm/CE/softmax primitives via **QuACK**.
Spend novel effort on **P0 (Newton-Schulz)** and **P1 (linear-CE)** — the two
highest-leverage gaps not covered off the shelf.

### On "megakernels" for *training*
Published megakernels (HazyResearch/Megakernels, Mirage MPK, FlashFormer) are all
**inference/forward-only** — no prior art for a single whole-model *training*
megakernel (backward + optimizer state make it much harder). So **don't** start with
a monolithic persistent kernel. Start with **block-level fused kernels** (P0–P5),
each fwd+bwd, wrapped as custom ops in a nanochat/torchtitan loop — the megakernel
*philosophy* (fuse everything fusible, kill launch/HBM overhead) without the
unproven moonshot. The **CODA** paper (arxiv 2605.19269, "transformer blocks as
GEMM-epilogue programs") is the closest training-fusion blueprint: backward has the
same tile structure as forward, so backward elementwise ops fuse as GEMM epilogues.

---

## 4. How to integrate a CuTeDSL kernel into the training loop

Core mechanism: zero-copy DLPack + compile-once-cache.

```python
import torch, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cute.jit
def my_kernel(mA: cute.Tensor, mB: cute.Tensor, mOut: cute.Tensor):
    _kernel(mA, mB, mOut).launch(grid=(...), block=(...), stream=torch.cuda.current_stream().cuda_stream)

# cache the converted wrappers (DLPack conv ~2-3µs) and the compiled fn:
a_ = from_dlpack(a, assumed_align=16)            # shares torch storage, zero-copy
compiled = cute.compile(my_kernel, a_, b_, out_) # compile once, keyed by shape/dtype
compiled(a_, b_, out_)                            # reuse every step
```

**Autograd / torch.compile:** CuTeDSL is **not** yet Dynamo/AOTInductor-traceable
like Triton (PyTorch dev-discuss). The robust pattern is to make each kernel an
opaque custom op via **`torch.library.custom_op`** (register a FakeTensor/meta
kernel for shapes), then bind fwd+bwd with a **`torch.autograd.Function`**. QuACK
does exactly this (`@cute_op("quack::_rmsnorm_fwd", ...)`). So for each P-kernel:
register `*_fwd` and `*_bwd` custom ops → wrap in an `autograd.Function` → drop in
as a `nn.Module`/functional replacement. The rest of the model stays eager or
`torch.compile`d around these opaque ops.

### Concrete repo-modification recipe (nanochat baseline)
1. `pip install nvidia-cutlass-dsl quack-kernels flash-attn-4` on the H100.
2. **Vendor the model**: copy nanochat's GPT module, swap config to Qwen3-1.7B
   shape (§2), add GQA + per-head QK-norm. Keep Muon+AdamW split.
3. **Drop-in replacements behind a flag** (`--kernels cute`), so eager PyTorch
   remains the correctness reference:
   - `attention` → `flash_attn.cute.flash_attn_func` (GQA, causal).
   - `RMSNorm` (+residual) → QuACK RMSNorm wrapped in `autograd.Function` (P2).
   - `QKNorm+RoPE` → custom CuTeDSL prologue kernel (P3).
   - `lm_head + CE` → fused linear-CE custom op (P1), built on QuACK CE.
   - `Muon.step` → CuTeDSL Gram-Newton-Schulz (P0), adapting Tri Dao's kernels.
   - `optimizer elementwise` → fused AdamW kernel (P4).
4. **Correctness gate**: each kernel ships a `pytest` comparing fwd+bwd against the
   eager module (`torch.allclose`, bf16 tolerances) on random inputs.
5. **Perf gate**: profile each swap with `profiling/run_nsys.sh` (did the bubble /
   launch count drop?) then `run_ncu.sh` (warp util / occupancy / memory on the new
   kernel). Iterate.

---

## 5. The profiling-driven development loop

1. `python profiling/train_qwen3.py --tiny` locally → pipeline smoke test.
2. `cd vast && ./launch.sh` → rent H100, profile baseline, fetch
   `timeline.nsys-rep` + `kernels.ncu-rep`, auto-destroy.
3. Open `nsys-ui timeline.nsys-rep` → find the idle gaps + the kernels eating the
   step (expect: many small norm/RoPE/optimizer launches + the NS iteration chain).
4. Open `ncu-ui kernels.ncu-rep` → confirm they're memory-/launch-bound (low warp
   util, low arithmetic intensity), i.e. fusible.
5. Implement the highest-ROI kernel (start **P0 Newton-Schulz**), gate on
   correctness + perf, re-profile. Repeat down the P-list.

---

## References
- CuTeDSL: github.com/NVIDIA/cutlass (`examples/python/CuTeDSL`), docs.nvidia.com/cutlass, pypi nvidia-cutlass-dsl
- Qwen3: huggingface.co/Qwen/Qwen3-1.7B, arxiv 2505.09388
- Baselines: github.com/KellerJordan/modded-nanoGPT, github.com/karpathy/nanochat, kellerjordan.github.io/posts/muon
- CuTeDSL kernels to reuse: github.com/Dao-AILab/quack, github.com/Dao-AILab/flash-attention (flash-attn-4), tridao.me/blog/2026/gram-newton-schulz, github.com/nil0x9/flash-muon
- Fusion theory: arxiv 2411.09009 (cut-cross-entropy), 2605.19269 (CODA), 2512.22219 (Mirage MPK), github.com/HazyResearch/Megakernels
- FlexAttention: pytorch.org/blog/flexattention
