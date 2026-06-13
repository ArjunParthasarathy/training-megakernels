#!/usr/bin/env bash
# End-to-end Vast.ai H100 profiling launcher:
#   search cheapest reliable H100  ->  create  ->  wait ready  ->  rsync code up
#   ->  run nsys + ncu over SSH  ->  copy .nsys-rep/.ncu-rep back  ->  DESTROY.
#
# Destroy at the end stops ALL billing (compute + storage). The onstart watchdog
# is a second safety net in case this script dies mid-run. After it finishes,
# `vastai show instances` should be empty.
#
# Prereqs:  pip install vastai  &&  vastai set api-key <KEY>
#           an SSH key registered:  vastai create ssh-key ~/.ssh/id_ed25519.pub
#
# Usage:  ./launch.sh [-- extra args forwarded to train_qwen3.py]
#   Env knobs:
#     IMAGE        docker image (default NGC pytorch, ships ncu+nsys)
#     DISK_GB      instance disk (default 64)
#     MAX_DPH      max $/hr to accept (default 3.0)
#     KEEP         set =1 to `stop` (keep disk) instead of `destroy` at the end
#     RESULTS_DIR  local dir for fetched reports (default ./results/<timestamp>)
#     PROFILE_WARMUP / PROFILE_STEPS  forwarded to the profiler
set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT="$(cd .. && pwd)"

IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:25.01-py3}"
DISK_GB="${DISK_GB:-64}"
MAX_DPH="${MAX_DPH:-3.0}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results/$(date +%Y%m%d-%H%M%S)}"
EXTRA_ARGS=("$@")

jqpy() { python3 -c "import sys,json; print(json.load(sys.stdin)$1)"; }

cleanup() {
  if [[ -n "${ID:-}" ]]; then
    if [[ "${KEEP:-0}" == "1" ]]; then
      echo ">> stopping instance $ID (keeping disk; you still pay storage)"
      vastai stop instance "$ID" || true
    else
      echo ">> destroying instance $ID (stops all billing)"
      vastai destroy instance "$ID" || true
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

echo ">> waiting for instance to be running ..."
for _ in $(seq 1 80); do
  ST=$(vastai show instance "$ID" --raw | jqpy ".get('actual_status')" || echo "?")
  echo "   status=$ST"
  [[ "$ST" == "running" ]] && break
  [[ "$ST" == "exited" || "$ST" == "offline" ]] && { echo "instance died"; exit 1; }
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

echo ">> uploading profiling/ code ..."
rsync -az -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $SSH_PORT" \
  "$REPO_ROOT/profiling/" "root@$SSH_HOST:/workspace/profiling/"

echo ">> running nsys + ncu on the H100 ..."
"${SSH[@]}" bash -s <<EOF
set -e
cd /workspace/profiling
pip install -q transformers >/dev/null 2>&1 || true
bash fix_profiling_perms.sh || true
export PROFILE_WARMUP="${PROFILE_WARMUP:-10}" PROFILE_STEPS="${PROFILE_STEPS:-3}"
echo "=== nsys (timeline) ==="
bash run_nsys.sh /workspace/out/timeline ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} || true
echo "=== ncu (per-kernel) ==="
PROFILE_STEPS=1 bash run_ncu.sh /workspace/out/kernels ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} || true
ls -lh /workspace/out
EOF

echo ">> fetching reports to $RESULTS_DIR ..."
mkdir -p "$RESULTS_DIR"
vastai copy "$ID":/workspace/out/ "local:$RESULTS_DIR/" || \
  rsync -az -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $SSH_PORT" \
    "root@$SSH_HOST:/workspace/out/" "$RESULTS_DIR/"

echo ">> done. Reports in $RESULTS_DIR :"
ls -lh "$RESULTS_DIR" || true
echo ">> open locally:  nsys-ui $RESULTS_DIR/timeline.nsys-rep ; ncu-ui $RESULTS_DIR/kernels.ncu-rep"
# trap cleanup() destroys the instance now.
