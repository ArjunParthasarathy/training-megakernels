# training-megakernels

Building **CuTeDSL megakernels for training** a Qwen3-1.7B model, and profiling the
training loop on rented H100s to find what to fuse next. CuTeDSL
(`pip install nvidia-cutlass-dsl`) compiles one source for **Hopper (sm90)** and
**Blackwell (sm100/sm120)**.

## Three comparable training runs

One shared eager architecture (`megakernels/model.py`); variants differ only in
optimizer / kernel backend / backward pass, so a comparison isolates exactly that.

| variant | optimizer | kernels | backward | role |
|---|---|---|---|---|
| `baseline` | AdamW | eager (SDPA, explicit CE) | autograd | reference bar |
| `modded` | Muon (Gram-NS) + AdamW | CuTeDSL (eager fallback), fused CE | autograd | **also a baseline** |
| `custom_backward` (`dev`) | same as modded | same | hand-written, more efficient | the experiment |

`dev` is an alias for `custom_backward` (matches the dev branch).

## Layout

| Path | What |
|---|---|
| `megakernels/model.py` | shared eager Qwen3 (GQA, QK-norm, RoPE, RMSNorm, SwiGLU, tied head) |
| `megakernels/kernels/` | op dispatch: `eager.py` reference + cute hooks, backend-selectable |
| `megakernels/cute/` | CuTeDSL kernels (GPU-only, import-guarded). `newton_schulz.py` = Gram-NS symmetric-GEMM; `wired.py` reuses quack/flash-attn-4 |
| `megakernels/optim/muon.py` | Muon + Newton-Schulz (standard **and** Gram); 2D-only, 1D→AdamW |
| `megakernels/custom_backward.py` | autograd.Functions with hand-written backward (the `dev` lever) |
| `megakernels/variants/` | `baseline` / `modded` / `custom_backward` configs + registry |
| `megakernels/train.py` | `python -m megakernels.train --variant <v>` (NVTX + cudaProfilerApi capture) |
| `megakernels/compare.py` | run ≥2 variants on identical data/seed, print loss / step-ms / tok-s / mem |
| `profiling/run_nsys.sh` `run_ncu.sh` | nsys timeline / ncu per-kernel, **take a variant arg** |
| `vast/launch.sh` | rent H100 → profile a variant → download reports → **destroy** |
| `docs/PLAN.md` | the CuTeDSL kernel roadmap + baseline analysis |

## Local dev (CPU, no GPU needed)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pytest -q                                              # 30 tests incl. Gram-NS ≡ standard-NS
python -m megakernels.compare --variants baseline modded dev --tiny --max-steps 20
```
On CPU the cute backend falls back to eager and `modded` is *slower* (Muon's
Newton-Schulz adds matmuls with no GPU to amortize) — the speedup is GPU-only and
is what the Vast profiling run measures.

## Profile a training run on Vast.ai  ← the one command

> **⚠️ Run all local `vastai` / `launch.sh` commands inside the `vast` conda env.**
> `vastai` uses `match`/`case` syntax requiring **Python ≥ 3.10**; base anaconda is
> 3.9 and crashes with `SyntaxError` on import. Activate the dedicated env first,
> every time:
> ```bash
> conda activate vast    # Python 3.11; created once: conda create -n vast python=3.11 -y && pip install vastai
> ```

One-time setup:
```bash
conda activate vast                                    # REQUIRED (Python ≥3.10; see warning above)
pip install vastai
vastai set api-key <YOUR_KEY>                          # ~/.config/vastai/vast_api_key
vastai create ssh-key ~/.ssh/id_ed25519.pub
```

Then, to profile a given run and pull the reports back to this machine:
```bash
cd vast && ./launch.sh <variant>        # variant = baseline | modded | dev
#   e.g.  ./launch.sh modded
```
`launch.sh` does it all: finds the cheapest verified single H100_SXM under
`MAX_DPH` ($3/hr default), creates it, rsyncs the repo up, installs the cute stack,
runs **both** `run_nsys.sh <variant>` (timeline) and `run_ncu.sh <variant>`
(per-kernel), copies `timeline-<variant>.nsys-rep` + `kernels-<variant>.ncu-rep`
into `results/<variant>-<ts>/`, then **`vastai destroy`** (an EXIT trap, fires even
on error/Ctrl-C; `onstart.sh` also arms a `MAX_LIFETIME_SECS` self-destruct
watchdog). `KEEP=1 ./launch.sh <variant>` uses `vastai stop` (keep disk) instead.

Open locally (your Nsight version must be ≥ the remote CLI version):
```bash
nsys-ui results/<variant>-<ts>/timeline-<variant>.nsys-rep
ncu-ui  results/<variant>-<ts>/kernels-<variant>.ncu-rep
```

### Profiler: two tools, two jobs
- kernel launches + bubbles/gaps → Nsight **Systems** (`nsys`, cheap) → `.nsys-rep`
- warp util + occupancy + memory → Nsight **Compute** (`ncu`, slow, scoped to the
  `profile_step` NVTX range) → `.ncu-rep`

`ncu --set full` replays each kernel 30–60×; the harness warms up then captures a
couple of steps via `--profile` (cudaProfilerApi). `torch.profiler` emits
Chrome/TensorBoard traces, **not** Nsight reports.

### Billing & ERR_NVGPUCTRPERM
- `vastai destroy` stops **all** billing (default); `vastai stop` keeps disk and
  still bills storage. After a run, `vastai show instances` should be empty.
- `ncu` needs perf counters: host `NVreg_RestrictProfilingToAdminUsers=0` +
  container `--cap-add=SYS_ADMIN` (Vast Template). Many multi-tenant hosts deny it
  — prefer verified whole-machine hosts, or fall back to nsys-only (timeline
  tracing needs no counters).
- Cold start: NGC image (~10–20 GB, ships ncu+nsys) pulls in ~3–8 min on fast net
  (`inet_down>1000`). Fresh-per-job = cleanest billing; `KEEP=1` (stop) keeps the
  image cached for fast iteration.

## Commit conventions
End commit messages with the Co-Authored-By trailer. Branch before committing on `main`.
