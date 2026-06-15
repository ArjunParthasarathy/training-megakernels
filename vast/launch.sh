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
# Minimum host-driver CUDA (nvidia-smi "CUDA Version"), via Vast's cuda_max_good.
# The NGC pytorch:25.01 image ships the CUDA 12.8 toolkit and the cute stack
# (nvidia-cutlass-dsl / flash-attn-4 / quack) JIT-compiles at runtime, so a host
# whose driver caps below the toolkit risks the cute install/JIT failing (we drew a
# 12.5 host and silently fell back to eager). Floor it to 12.6 with margin; raise
# toward 12.8 to match the toolkit exactly, lower if no offers come back.
MIN_CUDA="${MIN_CUDA:-12.6}"
# results/ is gitignored (large binary .nsys-rep/.sqlite), so a worktree's results
# never reach the main checkout via git. But that main checkout is where the user
# views them (their local dev branch). So fetch reports into the MAIN worktree's
# results/ regardless of which worktree we launch from (porcelain lists it first).
MAIN_WT="$(git -C "$REPO_ROOT" worktree list --porcelain 2>/dev/null | awk '/^worktree /{print $2; exit}')"
[[ -z "$MAIN_WT" ]] && MAIN_WT="$REPO_ROOT"   # not a git checkout / no worktrees -> here
RESULTS_DIR="${RESULTS_DIR:-$MAIN_WT/results/$VARIANT-$(date +%Y%m%d-%H%M%S)}"
EXTRA_ARGS=("$@")
echo ">> profiling variant: $VARIANT"

# strict=False: Vast's --raw JSON sometimes contains literal control chars (e.g. in
# an instance label/description), which json.load() rejects with
# "Invalid control character at ..." — strict=False tolerates them.
jqpy() { python3 -c "import sys,json; print(json.loads(sys.stdin.read(), strict=False)$1)"; }

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

echo ">> searching for a single H100_SXM offer under \$$MAX_DPH/hr (cuda_max_good>=$MIN_CUDA) ..."
OFFER=$(vastai search offers \
  "gpu_name=H100_SXM num_gpus=1 rentable=true verified=true reliability>0.99 inet_down>1000 disk_space>$DISK_GB direct_port_count>=1 cuda_max_good>=$MIN_CUDA dph_total<$MAX_DPH" \
  -o 'dph' --raw | jqpy "[0]['id']")
[[ -z "$OFFER" || "$OFFER" == "None" ]] && { echo "!! no H100_SXM offer matched (try lowering MIN_CUDA=$MIN_CUDA or raising MAX_DPH=$MAX_DPH)"; exit 1; }
echo ">> selected offer $OFFER"

# Keyed onstart: inject our pubkey so DIRECT SSH authenticates even when Vast's proxy
# key is broken account-wide and/or Vast never populates authorized_keys (see
# onstart.sh step 0 + CLAUDE.md "proxy SSH" + memory). We rewrite the LAUNCHER_PUBKEY=
# placeholder line in a temp copy and pass THAT to --onstart.
KEY="${KEY_FILE:-$HOME/.ssh/id_ed25519}"
PUBKEY_FILE="$KEY.pub"
[[ -f "$PUBKEY_FILE" ]] || { echo "!! no pubkey at $PUBKEY_FILE (set KEY_FILE)"; exit 1; }
PUBKEY="$(tr -d '\n' < "$PUBKEY_FILE")"
KEYED_ONSTART="${CLAUDE_JOB_DIR:-/tmp}/onstart_keyed.sh"
mkdir -p "$(dirname "$KEYED_ONSTART")"
# '|' sed delim (the key contains /,+,= but no '|'); single-quote the key in-file.
sed "s|^LAUNCHER_PUBKEY=.*|LAUNCHER_PUBKEY='$PUBKEY'|" onstart.sh > "$KEYED_ONSTART"
grep -q "LAUNCHER_PUBKEY='ssh-" "$KEYED_ONSTART" || { echo "!! pubkey injection into onstart failed"; exit 1; }
echo ">> keyed onstart written to $KEYED_ONSTART"

echo ">> creating instance ..."
ID=$(MAX_LIFETIME_SECS="${MAX_LIFETIME_SECS:-3600}" \
  vastai create instance "$OFFER" \
    --image "$IMAGE" --disk "$DISK_GB" --ssh --direct \
    --onstart "$KEYED_ONSTART" \
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

# Resolve SSH endpoint. CLAUDE.md historically preferred Vast's PROXY (ssh_host/
# ssh_port), but the proxy has been rejecting our account key account-wide (memory:
# vast-proxy-ssh-broken), so we now try BOTH transports and use whichever accepts our
# key: proxy first (ssh{N}.vast.ai) for back-compat, then DIRECT (public_ipaddr + the
# host port mapped to container 22/tcp) which works thanks to the keyed onstart above.
INST_JSON=$(vastai show instance "$ID" --raw)
PROXY_HOST=$(echo "$INST_JSON" | jqpy "['ssh_host']")
PROXY_PORT=$(echo "$INST_JSON" | jqpy "['ssh_port']")
DIRECT_HOST=$(echo "$INST_JSON" | jqpy ".get('public_ipaddr')")
# ports map: {'22/tcp': [{'HostPort': '12345', ...}]}; tolerate missing/empty.
DIRECT_PORT=$(echo "$INST_JSON" | jqpy ".get('ports',{}).get('22/tcp',[{}])[0].get('HostPort')")
echo ">> candidate endpoints: proxy root@$PROXY_HOST:$PROXY_PORT  direct root@$DIRECT_HOST:$DIRECT_PORT"

# common ssh opts; -i + IdentitiesOnly so we only offer (and the host only checks) our
# launcher key — avoids agent keys muddying the direct-endpoint auth.
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10
          -i "$KEY" -o IdentitiesOnly=yes)

# From here on the instance is billing, so any failure should STOP (keep disk) — not
# destroy — so we can restart + retry without re-renting (see CLAUDE.md). Remember
# the user's choice, then flip the trap to keep-mode; we restore it only after a
# clean fetch (so a successful run still destroys unless the user asked to keep).
KEEP_DEFAULT="${KEEP:-0}"
KEEP=1

# Probe both transports until one accepts the key (sshd + key propagation lag a few
# seconds after `running`; direct ports can take minutes to open). First to answer wins.
echo ">> probing ssh transports for key acceptance ..."
SSH_HOST=""; SSH_PORT=""
for _ in $(seq 1 36); do
  if [[ -n "$PROXY_HOST" && "$PROXY_HOST" != "None" ]] \
     && ssh "${SSH_OPTS[@]}" -o BatchMode=yes -p "$PROXY_PORT" "root@$PROXY_HOST" true 2>/dev/null; then
    SSH_HOST="$PROXY_HOST"; SSH_PORT="$PROXY_PORT"; echo ">> using PROXY transport"; break
  fi
  if [[ -n "$DIRECT_HOST" && "$DIRECT_HOST" != "None" && -n "$DIRECT_PORT" && "$DIRECT_PORT" != "None" ]] \
     && ssh "${SSH_OPTS[@]}" -o BatchMode=yes -p "$DIRECT_PORT" "root@$DIRECT_HOST" true 2>/dev/null; then
    SSH_HOST="$DIRECT_HOST"; SSH_PORT="$DIRECT_PORT"; echo ">> using DIRECT transport"; break
  fi
  sleep 5
done
[[ -n "$SSH_HOST" ]] || { echo "!! no ssh transport accepted the key; instance kept ($ID) for inspection"; exit 1; }
SSH=(ssh "${SSH_OPTS[@]}" -p "$SSH_PORT" "root@$SSH_HOST")
SSH_E="ssh ${SSH_OPTS[*]} -p $SSH_PORT"   # for rsync -e (same opts/key/port)
echo ">> ssh endpoint: root@$SSH_HOST:$SSH_PORT"
SSH_OK=1

echo ">> uploading repo (megakernels/ + profiling/) ..."
for attempt in 1 2 3; do
  rsync -az --exclude .venv --exclude .git --exclude results --exclude '__pycache__' \
    -e "$SSH_E" \
    "$REPO_ROOT/" "root@$SSH_HOST:/workspace/repo/" && break
  echo "   rsync upload attempt $attempt failed; retrying in 5s ..."; sleep 5
done

echo ">> running nsys (timeline) for variant=$VARIANT on the H100 ..."
"${SSH[@]}" bash -s <<EOF
set -e
cd /workspace/repo
mkdir -p /workspace/out
# Install + VERIFY the cute stack, fully logged (profiling/install_cute.sh) — this
# replaces the old silent \`pip install ... >/dev/null 2>&1 || true\` that hid every
# failure behind an eager fallback. \`if\` is exempt from \`set -e\`, so a non-zero
# (cute unavailable) does NOT abort: we still profile (eager) and the install log +
# train header say exactly why.
echo "=== install + verify cute stack ==="
if bash profiling/install_cute.sh /workspace/out/install-$VARIANT.log; then
  echo ">> cute stack: AVAILABLE — profiling the cute backend"
else
  echo ">> cute stack: UNAVAILABLE — profiling EAGER fallback (see install-$VARIANT.log)"
fi
# For the modded (cute) variant, VALIDATE the hand-registered custom_op backwards
# (quack rms_norm, flash-attn-4 attention) against the eager oracle before profiling —
# these shipped GPU-UNVALIDATED (commits ffc370f, 0caa106). Log goes to /workspace/out
# so it is fetched with the report; non-zero rc is recorded but does NOT abort the
# profiling (\`|| true\`), so we still get the timeline either way.
if [[ "$VARIANT" == "modded" || "$VARIANT" == "dev" ]]; then
  echo "=== gradient match: cute custom_ops vs eager ==="
  python -m profiling.verify_modded_grads 2>&1 | tee /workspace/out/gradcheck-$VARIANT.log || true
fi
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
# --timeout=120 --partial: the Vast SSH proxy intermittently stalls mid-transfer on
# the larger artifacts (e.g. the multi-MB .sqlite), which used to hang the fetch
# forever; bound it and keep partial progress so a retry/vastai-copy can finish.
rsync -az --timeout=120 --partial -e "$SSH_E" \
  "root@$SSH_HOST:/workspace/out/" "$RESULTS_DIR/" \
  || vastai copy "$ID":/workspace/out/ "local:$RESULTS_DIR/" || true

# Safety net: never destroy a good run before its report is safely local. If no
# .nsys-rep landed, keep the instance (stop, not destroy) so it can be re-fetched
# without re-renting — see "Iterating on a bug: restart, don't recreate" in CLAUDE.md.
if ls "$RESULTS_DIR"/*.nsys-rep >/dev/null 2>&1; then
  echo ">> report fetched OK; restoring KEEP=$KEEP_DEFAULT for teardown"
  KEEP="$KEEP_DEFAULT"   # clean run: destroy (unless the user asked to keep)
else
  echo "!! WARNING: no .nsys-rep fetched to $RESULTS_DIR — keeping instance (KEEP=1) so"
  echo "!! it is STOPPED (disk kept), not destroyed. Re-fetch without re-renting:"
  echo "!!   vastai start instance $ID && rsync -az -e 'ssh -p <port>' root@<host>:/workspace/out/ $RESULTS_DIR/"
  KEEP=1
fi

echo ">> done. Reports in $RESULTS_DIR :"
ls -lh "$RESULTS_DIR" || true
echo ">> open locally:  nsys-ui $RESULTS_DIR/timeline-$VARIANT.nsys-rep"
# trap cleanup() destroys the instance now.
