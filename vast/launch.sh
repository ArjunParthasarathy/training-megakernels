#!/usr/bin/env bash
# End-to-end Vast.ai H100 profiling launcher:
#   search cheapest reliable H100  ->  create  ->  start  ->  wait ready  ->  rsync
#   ->  run nsys (timeline) over SSH  ->  copy .nsys-rep back  ->  DESTROY.
#
# ncu (per-kernel) is intentionally DISABLED: the kernels are already optimized;
# we only care about the timeline (kernel launches, bubbles) and host<->device
# transfers, both of which nsys captures without GPU perf counters (no SYS_ADMIN
# needed). To re-enable ncu, restore the run_ncu.sh block below.
#
# Destroy at the end stops ALL billing (compute + storage). The onstart watchdog
# is a second safety net in case this script dies mid-run. After it finishes,
# `vastai show instances` should be empty.
#
# Prereqs:  pip install vastai  &&  vastai set api-key <KEY>
#           an SSH key registered:  vastai create ssh-key ~/.ssh/id_ed25519.pub
#
# Usage:  ./launch.sh <variant> [-- extra args forwarded to megakernels.train]
#   variant: baseline | modded | dev   (which training run to profile)
#   Env knobs:
#     IMAGE        docker image (default NGC pytorch, ships ncu+nsys)
#     DISK_GB      instance disk (default 64)
#     MAX_DPH      max $/hr to accept (default 3.0)
#     KEEP         set =1 to `stop` (keep disk) instead of `destroy` at the end
#     RESULTS_DIR  local dir for fetched reports (default ./results/<variant>-<ts>)
set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT="$(cd .. && pwd)"

VARIANT="${1:-baseline}"; shift || true
[[ "${1:-}" == "--" ]] && shift || true
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:25.01-py3}"
DISK_GB="${DISK_GB:-64}"
MAX_DPH="${MAX_DPH:-3.0}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results/$VARIANT-$(date +%Y%m%d-%H%M%S)}"
EXTRA_ARGS=("$@")
echo ">> profiling variant: $VARIANT"

jqpy() { python3 -c "import sys,json; print(json.load(sys.stdin)$1)"; }

cleanup() {
  if [[ -n "${ID:-}" ]]; then
    if [[ "${KEEP:-0}" == "1" ]]; then
      echo ">> stopping instance $ID (keeping disk; you still pay storage)"
      vastai stop instance "$ID" || true
    else
      echo ">> destroying instance $ID (stops all billing)"
      # -y is REQUIRED: newer vastai prompts [y/N] and would otherwise abort here,
      # leaking billing. Pipe `yes` too in case an older CLI lacks the flag.
      yes | vastai destroy instance "$ID" -y || true
    fi
    vastai show instances || true
  fi
}
trap cleanup EXIT

echo ">> searching for a single H100_SXM offer under \$$MAX_DPH/hr ..."
OFFER=$(vastai search offers \
  "gpu_name=H100_SXM num_gpus=1 rentable=true verified=true reliability>0.99 inet_down>1000 disk_space>$DISK_GB direct_port_count>=1 dph_total<$MAX_DPH" \
  -o 'dph' --raw | jqpy "[0]['id']")
echo ">> selected offer $OFFER"

echo ">> creating instance ..."
ID=$(MAX_LIFETIME_SECS="${MAX_LIFETIME_SECS:-3600}" \
  vastai create instance "$OFFER" \
    --image "$IMAGE" --disk "$DISK_GB" --ssh --direct \
    --onstart onstart.sh \
    --raw | jqpy "['new_contract']")
echo ">> instance id = $ID"

# Vast often creates the instance in a STOPPED state (intended_status=stopped) when
# the host can't schedule it instantly — it pulls the image then parks. Explicitly
# start it so it boots into `running` (idempotent if already running).
echo ">> starting instance $ID ..."
vastai start instance "$ID" || true

echo ">> waiting for instance to be running ..."
for _ in $(seq 1 80); do
  ST=$(vastai show instance "$ID" --raw | jqpy ".get('actual_status')" || echo "?")
  echo "   status=$ST"
  [[ "$ST" == "running" ]] && break
  [[ "$ST" == "exited" || "$ST" == "offline" ]] && { echo "instance died"; exit 1; }
  # if it parked in stopped/created, nudge it again (start is idempotent)
  [[ "$ST" == "stopped" ]] && vastai start instance "$ID" || true
  sleep 15
done

echo ">> waiting for onstart to finish (ONSTART_READY marker) ..."
for _ in $(seq 1 60); do
  vastai logs "$ID" >/tmp/vast_$ID.log 2>/dev/null || true
  grep -q "ONSTART_READY" /tmp/vast_$ID.log && break
  sleep 10
done

# Resolve direct SSH endpoint.
SSH_URL=$(vastai ssh-url "$ID")          # ssh://root@HOST:PORT
SSH_HOST=$(echo "$SSH_URL" | sed -E 's#ssh://[^@]+@([^:]+):.*#\1#')
SSH_PORT=$(echo "$SSH_URL" | sed -E 's#.*:([0-9]+)$#\1#')
SSH=(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p "$SSH_PORT" "root@$SSH_HOST")
echo ">> ssh endpoint: root@$SSH_HOST:$SSH_PORT"

echo ">> uploading repo (megakernels/ + profiling/) ..."
rsync -az --exclude .venv --exclude .git --exclude results --exclude '__pycache__' \
  -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $SSH_PORT" \
  "$REPO_ROOT/" "root@$SSH_HOST:/workspace/repo/"

echo ">> running nsys (timeline) for variant=$VARIANT on the H100 ..."
"${SSH[@]}" bash -s <<EOF
set -e
cd /workspace/repo
# GPU-only fused-kernel stack (best-effort: cute backend if it installs, else eager)
pip install -q nvidia-cutlass-dsl quack-kernels flash-attn-4 >/dev/null 2>&1 || true
mkdir -p /workspace/out
echo "=== nsys (timeline) ==="
bash profiling/run_nsys.sh "$VARIANT" /workspace/out/timeline-$VARIANT ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} || true
# ncu (per-kernel) intentionally disabled — see header. It needs GPU perf counters
# (SYS_ADMIN) which most multi-tenant hosts deny, and we only want the timeline +
# host<->device transfers. To re-enable, uncomment:
#   echo "=== ncu (per-kernel) ==="
#   bash profiling/fix_profiling_perms.sh || true
#   bash profiling/run_ncu.sh "$VARIANT" /workspace/out/kernels-$VARIANT ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} || true
ls -lh /workspace/out
EOF

echo ">> fetching reports to $RESULTS_DIR ..."
mkdir -p "$RESULTS_DIR"
# Pull over the SAME direct SSH that the upload used. NOT `vastai copy`: that goes
# through Vast's SSH proxy (vastai_kaalia@host:65535), fails publickey, yet exits 0
# — so it silently fetches nothing. Direct rsync is what works here; vastai copy is
# only a last-ditch fallback.
rsync -az -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $SSH_PORT" \
  "root@$SSH_HOST:/workspace/out/" "$RESULTS_DIR/" \
  || vastai copy "$ID":/workspace/out/ "local:$RESULTS_DIR/" || true

# Safety net: never destroy a good run before its report is safely local. If no
# .nsys-rep landed, keep the instance (stop, not destroy) so it can be re-fetched
# without re-renting — see "Iterating on a bug: restart, don't recreate" in CLAUDE.md.
if ! ls "$RESULTS_DIR"/*.nsys-rep >/dev/null 2>&1; then
  echo "!! WARNING: no .nsys-rep fetched to $RESULTS_DIR — forcing KEEP=1 so the"
  echo "!! instance is STOPPED (disk kept), not destroyed. Re-fetch manually:"
  echo "!!   vastai start instance $ID && rsync -az -e 'ssh -p <port>' root@<host>:/workspace/out/ $RESULTS_DIR/"
  KEEP=1
fi

echo ">> done. Reports in $RESULTS_DIR :"
ls -lh "$RESULTS_DIR" || true
echo ">> open locally:  nsys-ui $RESULTS_DIR/timeline-$VARIANT.nsys-rep"
# trap cleanup() destroys the instance now.
