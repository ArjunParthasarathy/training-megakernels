# torch.compile + CUDAGraph integration of the cute kernels — change report

**Scope.** This documents every `torch.library.custom_op` / `register_fake` /
`register_autograd` site introduced or changed while making the **`modded`** variant
actually run under `torch.compile(mode="reduce-overhead")` (= CUDAGraphs), plus the
three distinct bugs that surfaced on the H100 and how each was fixed. It is meant as
the map for a later deep pass — the places to scrutinize are listed per-site with the
exact invariant each one must satisfy.

All code lives in **`megakernels/cute/wired.py`** (GPU-only, import-guarded). The
gradient oracle that validates these ops is **`profiling/verify_modded_grads.py`**.
Validated on an H100 (NVIDIA H100 80GB HBM3, NGC pytorch:25.01, torch 2.6,
flash-attn-4 + quack + nvidia-cutlass-dsl 4.5.2): all forward/backward gradient
checks PASS and the `modded` nsys timeline profiles cleanly.

Commits (on `dev`):
- `a608090` — wrap the FA4 attention **backward** as an opaque custom op
- `208013c` — make the attention `register_fake`s return **contiguous** tensors
- (`18a034e` added `profiling/verify_modded_grads.py`; the rms_norm wrapping predates
  this work, commit `ffc370f`, and the attention forward wrapping `0caa106`.)

---

## Why any of this exists

Under `torch.compile`, Dynamo/AOTAutograd trace the model with **FakeTensors** to
build forward and backward graphs, and `reduce-overhead` then CUDAGraph-captures
them. A CuTeDSL kernel (flash-attn-4, quack) is **opaque** to the tracer — it
ultimately calls `.data_ptr()` / `from_dlpack` on its inputs, which a FakeTensor has
no storage for. A bare call therefore either graph-breaks (fragmenting CUDAGraph
capture) or hard-errors. The fix is the PyTorch-recommended pattern: declare the
kernel a `custom_op`, give it a `register_fake` (meta kernel) so the tracer can
propagate **shapes and strides** without running the body, and a `register_autograd`
so the analytic backward lands in the AOTAutograd backward graph.

**The non-obvious part (and the source of both runtime bugs): the same opacity rule
applies to the BACKWARD, and the fake's STRIDES must match the real op's output, not
just its shape/dtype.**

---

## The four custom-op sites (current state)

### 1. `megakernels::cute_rms_norm` — RMSNorm forward (quack)
- `wired.py:143` `custom_op`, `:149` `register_fake`, `:175` `register_autograd`.
- **Forward**: `quack.rmsnorm(x, weight, eps=eps)` (note: `eps` is a **keyword** — the
  3rd positional is `bias`).
- **Fake**: `torch.empty_like(x)` — RMSNorm preserves shape/dtype. *Left as plain
  `empty_like`* (no `memory_format`): in this model `rms_norm` is only ever applied to
  contiguous hidden states, so `empty_like(x)` is already contiguous and matches. See
  the deep-pass note below.
- **Backward**: `_backward` (`:160`) is **pure torch** (recomputes `rstd`, analytic
  grad_x / grad_w), so it is itself traceable/CUDAGraph-capturable — no opaque-bwd
  wrapper needed. Mirrors `custom_backward.RMSNormFn` (CPU-gradchecked in
  `tests/test_custom_backward.py`).

### 2. `megakernels::cute_attention` — attention forward (flash-attn-4)
- `wired.py:55` `custom_op`, `:67` `register_fake`, `:126` `register_autograd`.
- **Forward**: `flash_attn_func(..., return_lse=True)` → `(out, lse)`; layout mapped
  model `[B,H,T,D]` ↔ flash `[B,T,H,D]` via `_to_flash`/`_to_model`
  (`transpose(1,2).contiguous()`). Default softmax scale `1/sqrt(D)` (matches eager
  SDPA). Returns `(out, lse)`; the public `attention()` wrapper (`:128`) drops `lse`.
- **Fake** (`:67`): returns `(empty_like(q, memory_format=contiguous_format),
  new_empty([B,H,T], fp32))`. The **`contiguous_format` is load-bearing** — see Bug 2.
  **LSE shape `[B,H,T]` fp32 is an ASSUMPTION** — flagged for the deep pass.
- **Backward**: delegates to the opaque `cute_attention_bwd` (site 3).

### 3. `megakernels::cute_attention_bwd` — attention backward (flash-attn-4)  *(new, `a608090`)*
- `wired.py:91` `custom_op`, `:102` `register_fake`. Called from `cute_attention`'s
  `_backward` (`:119`).
- **Body**: `_flash_attn_bwd(q,k,v,out,grad_out,lse, softmax_scale=None, causal=…)` →
  `(dq,dk,dv)`, layout-mapped back to model. `softmax_scale=None` = flash default,
  consistent with the forward.
- **Fake** (`:102`): contiguous `empty_like` of q/k/v (dk/dv keep the **GQA kv-head
  count** of k/v). Again `contiguous_format` is required (Bug 2).
- **Why a second op at all**: see Bug 1.

### 4. (context) `linear_cross_entropy` and the linear/GEMM layers — **NOT wrapped, on purpose**
- `linear_cross_entropy` stays **eager** (`megakernels/kernels/eager.py`): quack ships
  no fused linear-CE. It is plain `F.linear` + `F.cross_entropy` — native aten,
  fully traceable by Dynamo/Inductor, no `.data_ptr()`. **No custom op needed.**
- `q/k/v/o_proj`, `gate/up/down_proj`, the tied head: `nn.Linear` / `F.linear` — aten
  matmuls, traced and (under reduce-overhead) graphed by Inductor directly.
- **If a cute CE or a cute GEMM is wired later, it must follow the full pattern below
  (opaque forward + opaque backward + stride-correct fakes).**
- **Latent item — `newton_schulz`** (`megakernels/cute/newton_schulz.py:59`,
  `from_dlpack`): it IS an opaque DSL kernel and would hit the same FakeTensor wall
  *if traced*, but it runs inside the **Muon optimizer step, which is never
  `torch.compile`d**, and is currently an eager passthrough. Safe today; would need
  wrapping only if the optimizer is ever compiled AND its real symmetric-GEMM mainloop
  lands.

---

## The three bugs (each a real, distinct integration requirement)

### Bug 0 — opaque forward (pre-existing pattern, `ffc370f` / `0caa106`)
A bare CuTeDSL call graph-breaks / can't be traced. **Fix:** `custom_op` +
`register_fake`. This was already done for rms_norm and the attention forward before
this task; included here for completeness because the same reasoning drives Bugs 1–2.

### Bug 1 — the backward also traced into a raw DSL kernel  *(fix `a608090`)*
- **Symptom (H100, compile step, in `loss.backward()`):**
  ```
  File megakernels/cute/wired.py, in _backward
      dq, dk, dv = _flash_attn_bwd(...)
  RuntimeError: Cannot access data pointer of Tensor (e.g. FakeTensor). If you're using
  torch.compile/export/fx, it is likely that we are erroneously tracing into a custom
  kernel. To fix this, please wrap the custom kernel into an opaque custom op.
  ```
- **Cause:** `register_autograd` injects the backward into AOTAutograd's **backward
  graph**, which is *also* traced. The backward called `_flash_attn_bwd` directly; the
  tracer followed it into flash's dlpack `to_cute_tensor` → `.data_ptr()` on a
  FakeTensor. (Eager autograd ran it fine — which is why the gradient check, run in
  eager, PASSed and masked this; only the *compiled* path breaks.)
- **Fix:** wrap the FA4 backward as its own opaque `cute_attention_bwd` (site 3), and
  have `_backward` call that. AOTAutograd then keeps it as one opaque node.

### Bug 2 — `register_fake` strides didn't match the real op  *(fix `208013c`)*
- **Symptom (H100, compile step, CUDAGraph backward):**
  ```
  File .../cudagraph_trees.py ... in call
      assert_size_stride(getitem, (4, 16, 2048, 128), (4194304, 128, 2048, 1))
  AssertionError: expected size 16==16, stride 262144==128 at dim=1; 2048 stride 128==2048 at dim=2
  ```
- **Cause:** the real ops return `_to_model(...) = transpose(1,2).contiguous()`, i.e. a
  **contiguous** `[B,H,T,D]` (strides `…,262144,128,1`). The fakes used bare
  `empty_like(q/k/v)`, which **inherits the input's stride** — and the inputs at that
  point are transposed views (`[B,T,H,D]`-contiguous seen as `[B,H,T,D]`, strides
  `…,128,2048,1`). Inductor traced the op's output with the *fake's* (transposed)
  strides, then `assert_size_stride` fired when the runtime tensor came back
  contiguous. **A `register_fake` must reproduce the real op's strides, not just its
  shape/dtype.**
- **Fix:** `torch.empty_like(..., memory_format=torch.contiguous_format)` on the
  attention forward `out` and the backward `dq/dk/dv`.

---

## Deep-pass checklist

For each opaque op, verify on a GPU:
1. **Forward fake ≡ real, in shape, dtype, AND stride.** The Bug-2 trap. Anywhere the
   real op does `.contiguous()` / `.transpose()` / reshapes, the fake must match.
   Candidates to re-examine: `cute_rms_norm`'s fake (currently plain `empty_like(x)` —
   correct only while `x` is contiguous; harden to `contiguous_format` if rms_norm ever
   sees a non-contiguous input, e.g. if `qk_norm` is rewired to the cute op — today
   `qk_norm` uses the *eager* rms_norm so it's fine).
2. **Backward contains no un-wrapped DSL call.** The Bug-1 trap. Grep `wired.py`
   `_backward` bodies for direct `_flash_attn_bwd` / `from_dlpack` / quack calls —
   they must go through an opaque op or be pure torch.
3. **LSE assumption** (`cute_attention.register_fake`, `wired.py:75`): shape `[B,H,T]`,
   dtype fp32. Confirm against flash-attn-4's actual return; if wrong, the fake mis-
   declares and grads silently route oddly. (The end-to-end gradcheck PASSing is strong
   evidence it's right, but verify directly.)
4. **Scale consistency**: forward and backward both use flash's default `1/sqrt(D)`
   (`softmax_scale=None`). Keep them in lockstep if either is ever set explicitly.
5. **GQA head counts**: `dk/dv` fakes use k/v's shape (kv-head count), `dq` uses q's
   (query-head count). Correct for GQA 2:1; re-check if head config changes.
6. **`mutates_args=()`** on all four ops — they are pure (no in-place). Keep it accurate;
   a wrong value corrupts the functionalization pass.

## How it's validated

`profiling/verify_modded_grads.py` (run on the H100 before profiling) compares each
cute op's forward output and **every** input/weight grad against the eager oracle via
rel-L2 + cosine. Note it runs in **eager autograd**, so it validates *numerics* (Bug 0
correctness) but NOT the compiled graph — Bugs 1 and 2 only appear under
`torch.compile` and are caught by the nsys profiling run actually completing. A future
hardening would add a tiny `torch.compile(mode="reduce-overhead")` forward+backward
smoke step to the verify script so the compile path is gated too.
