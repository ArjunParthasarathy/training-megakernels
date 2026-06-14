# training-megakernels

Building **CuTeDSL megakernels for training** a Qwen3-1.7B model, and profiling the
training loop on rented H100s to find what to fuse next. CuTeDSL
(`pip install nvidia-cutlass-dsl`) compiles one source for **Hopper (sm90)** and
**Blackwell (sm100/sm120)**.

## Comparable training runs

One shared eager architecture (`megakernels/model.py`); variants differ only in
optimizer / kernel backend / backward pass / graphing, so a comparison isolates
exactly that.

| variant | optimizer | kernels | backward | role |
|---|---|---|---|---|
| `baseline` (`eager`) | AdamW | eager (SDPA, explicit CE) | autograd | reference bar |
| `cudagraph` | AdamW | same eager kernels, `torch.compile(mode="reduce-overhead")` | autograd | eager-but-graphed: isolates the per-launch overhead win |
| `modded` | Muon (Gram-NS) + AdamW | CuTeDSL (eager fallback), fused CE | autograd | **also a baseline** |
| `custom_backward` (`dev`) | same as modded | same | hand-written, more efficient | the experiment |

`eager` is an alias for `baseline` (pure eager, no compile/graphs); `dev` is an
alias for `custom_backward` (matches the dev branch). `cudagraph` keeps the *same*
kernels as `baseline` and only replays them via CUDAGraphs — it does **not** use
`mode="max-autotune"`, which would also swap in autotuned Triton GEMMs and so
confound the launch-overhead measurement with kernel selection. Any variant can be
manually `torch.compile`d in default (non-graph) mode with `--compile`.

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
`MAX_DPH` ($3/hr default), creates it, **explicitly `vastai start`s it** (Vast often
creates instances stopped — see below), rsyncs the repo up, installs the cute stack,
runs `run_nsys.sh <variant>` (timeline), copies `timeline-<variant>.nsys-rep` into
`results/<variant>-<ts>/`, then **`vastai destroy -y`** (an EXIT trap, fires even
on error/Ctrl-C; `onstart.sh` also arms a `MAX_LIFETIME_SECS` self-destruct
watchdog). `KEEP=1 ./launch.sh <variant>` uses `vastai stop` (keep disk) instead.

Open locally (your Nsight version must be ≥ the remote CLI version):
```bash
nsys-ui results/<variant>-<ts>/timeline-<variant>.nsys-rep
```

### Profiler: nsys timeline only (ncu disabled)
We profile with Nsight **Systems** only — kernel launches, bubbles/gaps, and
**host↔device transfers** (HtoD/DtoH memcpy). That's all `-t cuda` tracing, which
needs **no GPU perf counters**, so it runs on any host. `run_nsys.sh` also prints
`cuda_gpu_mem_time_sum` / `cuda_gpu_mem_size_sum` so transfer cost shows next to kernels.

For the graphed variants (`cudagraph`, `modded` — both `reduce-overhead` =
CUDAGraphs), `run_nsys.sh` passes `--cuda-graph-trace=node`. nsys's default
(`graph`) draws each captured graph as one opaque `cudaGraphLaunch` blob; `node`
expands it into its individual kernel nodes, so you can see **exactly which kernels
each capture contains** (and their per-node timing) in the timeline and in
`cuda_gpu_kern_sum`. Eager variants stay on `graph` (no graphs to expand).
Override per run with `CUDA_GRAPH_TRACE=graph|node ./run_nsys.sh <variant>`.

Nsight **Compute** (`ncu`, per-kernel warp util/occupancy) is **disabled on purpose**
— and on Vast it's effectively **impossible**, not just inconvenient. ncu needs GPU
perf counters, which require `NVreg_RestrictProfilingToAdminUsers=0` on the host
(owner-only: `/etc/modprobe.d` + reboot) **and** container `--cap-add=SYS_ADMIN`. But
Vast runs renters in **unprivileged containers** and its Docker-Options field exposes
**only ports/env/hostname — no `--cap-add`, no `--privileged`** (per Vast's Security
FAQ + Docker-Environment docs). So `ERR_NVGPUCTRPERM` is unavoidable on a normal
rental; the only escape is a whole-machine/dedicated host whose owner pre-enabled
counters (~5–10× the cost, not guaranteed, no marketplace filter for it). **For real
per-kernel counter profiling, use a host you control (Lambda/CoreWeave/own box), not
Vast.** Do **not** pass `--gpu-metrics-device` to nsys either — same counter wall,
makes nsys exit with a usage error. `torch.profiler` emits Chrome/TensorBoard traces,
**not** Nsight reports.

### Use Vast's proxy SSH, not direct
Connect/rsync over the **proxy** endpoint (`ssh_host`/`ssh_port` from
`vastai show instance --raw`, e.g. `root@ssh3.vast.ai:25804`) — **not** the direct
`vastai ssh-url` (`root@<public-ip>:<port>`). Direct ports are frequently
unavailable (`direct_port_start: -1`) or take minutes to open even after the
instance shows `running`, giving `Connection refused` / `Permission denied
(publickey)` that aborts the run. The proxy (`ssh{N}.vast.ai`) is Vast's managed
jump host and comes up reliably. `launch.sh` resolves the proxy and still probes it
(`ssh … true` in a loop) before uploading, since even the proxy can lag a few
seconds after `running` (Vast's own docs: "if authentication fails, try again after
a few seconds"). It's a readiness race, **not** a key problem — verify the key once
with `vastai show ssh-keys` vs `ssh-keygen -lf ~/.ssh/id_ed25519.pub`.

### Iterating on a bug: restart, don't recreate
When a run fails on a **code/script bug** (not a dead host), don't destroy +
re-rent — eat the cold start for nothing. Instead **reuse the same instance**:
launch with `KEEP=1` so the EXIT trap `vastai stop`s (keeps disk + the pulled
image on that host) instead of destroying, then iterate:
```bash
KEEP=1 ./launch.sh <variant>                 # first run; stops (not destroys) at end
vastai start instance <id>                   # bring it back (seconds — image cached)
rsync -az ... <repo>/ root@host:/workspace/repo/   # push the code fix
vastai ssh <id> -- bash profiling/run_nsys.sh <variant> /workspace/out/timeline-<variant>
vastai destroy instance <id> -y              # only when truly done
```
Image layers live in the **host's** Docker cache (per machine), so a restart on the
same host skips the 3–8 min pull. Caveat: `stop` frees the GPU — restart is
best-effort (another renter can take it). `destroy -y` when done to stop all billing.

### Billing, the `-y` trap, and the created-stopped gotcha
- **`vastai destroy` MUST pass `-y`.** Newer vastai prompts `[y/N]`; without `-y`
  (or `yes |`) the EXIT trap and the onstart watchdog both **abort the destroy and
  leak billing**. Both call sites now pass `-y`. After a run, `vastai show
  instances` should be empty — verify it.
- **Vast often creates instances `stopped`** (intended_status=stopped): the host
  pulls the image then parks instead of running. `launch.sh` now `vastai start`s
  explicitly after create and re-nudges if it sees `stopped`, else the wait loop
  spins forever and SSH fails.
- `vastai stop` keeps disk and **still bills storage**; `vastai destroy -y` stops
  **all** billing.
- `ncu` counter profiling is **not possible on a standard Vast rental** (unprivileged
  containers; Docker Options expose no caps; host modprobe is owner-only) → we stay
  nsys-only (timeline + transfers need no counters). See the Profiler section.
- Cold start: NGC image (~10–20 GB) pulls in ~3–8 min on fast net
  (`inet_down>1000`). Fresh-per-job = cleanest billing; `KEEP=1` (stop) keeps the
  image cached on that host for fast restart (see "Iterating on a bug" above).

## Commit conventions
End commit messages with the Co-Authored-By trailer. Branch before committing on `main`.
