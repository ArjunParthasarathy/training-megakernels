# training-megakernels

Building **CuTeDSL megakernels for training** a Qwen3-1.7B-style model, and
profiling that training loop on rented H100s to find what to fuse next.

CuTeDSL (`pip install nvidia-cutlass-dsl`) compiles one Python source for both
**Hopper (sm90)** and **Blackwell (sm100/sm120)** — write the kernel once, run on
either arch.

## Layout

| Path | What |
|------|------|
| `profiling/profile_utils.py` | warmup + `cudaProfilerApi` capture + NVTX `profile_step`/forward/backward/optimizer ranges. Wrap any loop with `profiled_loop(...)`. |
| `profiling/train_qwen3.py` | self-contained Qwen3-1.7B profiling target (random init, synthetic data). Swap for the real nanochat/torchtitan loop later. |
| `profiling/run_nsys.sh` | Nsight **Systems** timeline (launches, bubbles, overlap) → `.nsys-rep`. Run this FIRST. |
| `profiling/run_ncu.sh` | Nsight **Compute** per-kernel (warp util, occupancy, memory) → `.ncu-rep`. Run on the few kernels nsys flagged. |
| `profiling/fix_profiling_perms.sh` | fixes `ERR_NVGPUCTRPERM` so ncu can read perf counters. |
| `vast/launch.sh` | one command: rent H100 → run nsys+ncu → fetch reports → **destroy**. |
| `vast/onstart.sh` | boot script: enables counters + arms a self-destruct watchdog. |
| `docs/PLAN.md` | the CuTeDSL kernel plan + baseline (modded-nanoGPT/nanochat) analysis. |

## Profiler: two tools, two jobs

Your ask ("kernel launches, bubbles, warp utilization") spans **both** Nsight tools:

- **kernel launches + bubbles/gaps** → Nsight **Systems** (`nsys`, timeline, cheap) → `.nsys-rep`
- **warp utilization + occupancy + memory** → Nsight **Compute** (`ncu`, per-kernel, slow) → `.ncu-rep`

Workflow: `nsys` first to find the idle gaps and the expensive kernels, then `ncu`
scoped to just those. Both open in their respective local UIs (`nsys-ui`, `ncu-ui`).
**Your local Nsight version must be ≥ the remote CLI version** or the report won't open.
PyTorch's built-in `torch.profiler` emits Chrome/TensorBoard traces, **not** these
formats — it won't open in Nsight Compute.

`ncu --set full` replays every kernel 30–60× → can run for GPU-hours. The harness
warms up `PROFILE_WARMUP` (default 10) steps then captures `PROFILE_STEPS` (default 3
for nsys, forced to 1 for ncu) via the `profile_step` NVTX range, so a run is minutes.

## Run a profiling job on Vast.ai (the short version)

The Vast CLI handles search/create/destroy; one wrapper script chains it.

```bash
pip install vastai
vastai set api-key <YOUR_KEY>                      # stored in ~/.config/vastai/vast_api_key
vastai create ssh-key ~/.ssh/id_ed25519.pub        # so the launcher can SSH in

cd vast && ./launch.sh                              # rent H100 → profile → fetch → destroy
#   ./launch.sh -- --tiny      # smoke-test the pipeline on a small/cheap GPU first
```

`launch.sh`: searches the cheapest **verified single H100_SXM** under `MAX_DPH`
($3.0/hr default, fast net), creates it with `onstart.sh`, waits until running,
rsyncs `profiling/` up, runs `run_nsys.sh` + `run_ncu.sh`, copies
`timeline.nsys-rep` / `kernels.ncu-rep` into `results/<timestamp>/`, then
**`vastai destroy instance`** (an `EXIT` trap, so it fires even on error/Ctrl-C).
`onstart.sh` also arms a `MAX_LIFETIME_SECS` (default 3600s) self-destruct
watchdog as a billing backstop. Set `KEEP=1` to `stop` (keep disk) instead of
destroy when iterating.

Open results locally:
```bash
nsys-ui results/<ts>/timeline.nsys-rep
ncu-ui  results/<ts>/kernels.ncu-rep
```

### Billing & ERR_NVGPUCTRPERM
- `vastai stop instance <id>` halts compute but **still bills storage**;
  `vastai destroy instance <id>` stops **all** billing. We destroy by default.
- After any run: `vastai show instances` should be empty.
- `ncu` needs GPU perf counters: host `NVreg_RestrictProfilingToAdminUsers=0`
  **and** container `--cap-add=SYS_ADMIN` (set via a Vast **Template**'s Docker
  options). Many multi-tenant hosts won't grant it — prefer **verified
  whole-machine** hosts, or fall back to **nsys-only** (timeline tracing does
  not need counters).

### Cold-start tradeoff (fresh instance per job vs. reuse)
- NGC image (`nvcr.io/nvidia/pytorch`, ~10–20 GB) ships ncu+nsys but pulls in
  ~3–8 min on a fast host (filter `inet_down>1000`). Lean `pytorch/pytorch:*-runtime`
  (~4–7 GB) pulls faster but needs extra installs.
- **Fresh per job:** cleanest billing, pay only while running, but eat cold start
  every run. **Reuse one instance:** no repeated cold start, but pay while idle.
  **Middle ground:** `KEEP=1` → `vastai stop` keeps the image/disk cached (fast
  restart, storage-only billing) while iterating; `destroy` when truly done.
  A persistent **volume** keeps datasets/checkpoints across destroyed instances.

## Local dev

```bash
pip install -r profiling/requirements.txt
python profiling/train_qwen3.py --tiny --max-steps 20   # CPU/small-GPU smoke test
```

## Commit conventions
End commit messages with the Co-Authored-By trailer. Branch before committing on `main`.
