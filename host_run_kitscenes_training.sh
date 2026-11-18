#!/usr/bin/env bash
set -uo pipefail

CONT="auto_e2e_ubuntu24"
REPO="/home/adone/docker_workspaces/auto_e2e"
EXT="/mnt/datadisk/ae2e"
LOG="$REPO/host_logs/kitscenes_host_supervisor.log"

TZ_NAME="Europe/Rome"

# Training is allowed every day:
# 19:00 -> 07:59
BLOCKED_FROM=8
ALLOWED_FROM=19

CHECK_SECONDS=60
GPU_CHECK_SECONDS=300
STALL_SECONDS=1800
SLOW_SECONDS=900
MIN_DOWNLOAD_MBPS=2
DOWNLOAD_MBPS=0 #0 means unlimited
RESTART_SECONDS=60

mkdir -p "$REPO/host_logs"

log() {
  echo "[HOST-SUPERVISOR] $(TZ="$TZ_NAME" date '+%Y-%m-%d %H:%M:%S %Z') $*" | tee -a "$LOG"
}

allowed_now() {
  local h
  h="$(TZ="$TZ_NAME" date +%H)"
  h=$((10#$h))

  # Blocked 08:00 <= hour < 19:00
  if (( h >= BLOCKED_FROM && h < ALLOWED_FROM )); then
    return 1
  fi

  return 0
}

seconds_until_allowed() {
  local now target

  now="$(date +%s)"
  target="$(TZ="$TZ_NAME" date -d "today ${ALLOWED_FROM}:00:00" +%s)"

  if (( target <= now )); then
    target="$(TZ="$TZ_NAME" date -d "tomorrow ${ALLOWED_FROM}:00:00" +%s)"
  fi

  echo $(( target - now ))
}

cleanup_partial_downloads() {
  rm -rf \
    "$REPO/datasets/kitscenes_bounded_cache/.kitscenes_staging" \
    "$REPO/datasets/kitscenes_bounded_cache/.cache" \
    2>/dev/null || true
}

stop_training() {
  if docker ps --format '{{.Names}}' | grep -qx "$CONT"; then
    log "Stopping KITScenes training process."

    docker exec "$CONT" bash -lc '
      pkill -TERM -f train_kitscenes_bounded_shards.py 2>/dev/null || true
    ' >/dev/null 2>&1 || true

    sleep 20

    docker exec "$CONT" bash -lc '
      pkill -KILL -f train_kitscenes_bounded_shards.py 2>/dev/null || true
    ' >/dev/null 2>&1 || true
  fi
}

stop_container_for_schedule() {
  stop_training
  cleanup_partial_downloads

  if docker ps --format '{{.Names}}' | grep -qx "$CONT"; then
    log "Stopping Docker container to release GPU during blocked hours."
    docker stop "$CONT" >/dev/null 2>&1 || true
  fi
}

preflight_container_gpu() {
  docker exec "$CONT" bash -lc '
    cd /auto_e2e
    source /root/e2e-env/bin/activate

    nvidia-smi >/dev/null

    python - <<PY
import torch, sys

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    sys.exit(42)

print("gpu:", torch.cuda.get_device_name(0))
PY
  '
}

ensure_container_gpu() {
  if ! nvidia-smi >/dev/null 2>&1; then
    log "ERROR: host NVIDIA GPU is not working."
    return 10
  fi

  log "Starting Docker container."
  docker start "$CONT" >/dev/null

  log "Checking GPU inside Docker."

  if preflight_container_gpu; then
    log "GPU is visible inside Docker."
    return 0
  fi

  log "GPU not visible inside Docker. Restarting Docker service and container."

  docker stop "$CONT" >/dev/null 2>&1 || true
  systemctl restart docker
  sleep 10
  docker start "$CONT" >/dev/null

  log "Rechecking GPU inside Docker."

  if ! preflight_container_gpu; then
    log "ERROR: GPU still not visible after Docker restart."
    return 13
  fi

  log "GPU recovered after Docker restart."
  return 0
}

preflight_storage() {
  if [ ! -d "$EXT" ]; then
    log "ERROR: external disk path not found: $EXT"
    return 11
  fi

  mkdir -p "$REPO/external_storage"

  if ! mountpoint -q "$REPO/external_storage"; then
    log "Binding external disk into repo external_storage."
    mount --bind "$EXT" "$REPO/external_storage"
  fi

  if [ ! -f "$REPO/checkpoints/kitscenes_bounded/last.pt" ]; then
    log "ERROR: checkpoint not found: $REPO/checkpoints/kitscenes_bounded/last.pt"

    find "$REPO" "$EXT" \
      \( -name "last.pt" -o -name "best.pt" \) \
      2>/dev/null | tee -a "$LOG" || true

    return 12
  fi

  return 0
}

launch_training() {
  log "Cleaning stale KITScenes training/supervisor processes."

  docker exec "$CONT" bash -lc '
    pkill -TERM -f run_kitscenes_supervised.py 2>/dev/null || true
    pkill -TERM -f train_kitscenes_bounded_shards.py 2>/dev/null || true
    sleep 5
    pkill -KILL -f run_kitscenes_supervised.py 2>/dev/null || true
    pkill -KILL -f train_kitscenes_bounded_shards.py 2>/dev/null || true
  ' >/dev/null 2>&1 || true

  cleanup_partial_downloads

  log "Launching KITScenes trainer directly."
  log "checkpoint: /auto_e2e/checkpoints/kitscenes_bounded/last.pt"
  log "download limit: ${DOWNLOAD_MBPS} Mbps"
  log "slow threshold: ${MIN_DOWNLOAD_MBPS} Mbps for ${SLOW_SECONDS}s"
  log "stall watchdog: ${STALL_SECONDS}s"
  log "GPU check interval: ${GPU_CHECK_SECONDS}s"

  coproc TRAIN_PROC {
    docker exec -i \
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$CONT" bash -lc "
        cd /auto_e2e
        source /root/e2e-env/bin/activate

        exec /root/e2e-env/bin/python \
          /auto_e2e/train_kitscenes_bounded_shards.py \
          --epochs 5 \
          --accum-steps 2 \
          --save-every-scenes 1 \
          --validate-every-scenes 50 \
          --val-scenes-per-eval 5 \
          --shuffle-scenes \
          --download-rate-mbps ${DOWNLOAD_MBPS} \
          --download-pause-seconds 5 \
          --resume /auto_e2e/checkpoints/kitscenes_bounded/last.pt
      " 2>&1
  }

  local train_pid="$TRAIN_PROC_PID"
  local train_fd="${TRAIN_PROC[0]}"

  local now
  local last_output
  local last_gpu_check
  local slow_since=0
  local restart_reason=""

  now="$(date +%s)"
  last_output="$now"
  last_gpu_check="$now"

  while kill -0 "$train_pid" 2>/dev/null; do

    if IFS= read -r -t "$CHECK_SECONDS" line <&"$train_fd"; then
      echo "$line" | tee -a "$LOG"

      now="$(date +%s)"
      last_output="$now"

      # CUDA failure reported by trainer.
      if [[ "$line" == *"device: cpu"* ]] ||
         [[ "$line" == *"No CUDA GPUs are available"* ]] ||
         [[ "$line" == *"CUDA is unavailable inside Docker"* ]] ||
         [[ "$line" == *"cuda available: False"* ]]; then

        restart_reason="cuda"
        log "CUDA failure detected in training output."
        break
      fi

      if [[ "$line" == *"Bounded shard training complete."* ]]; then
        log "KITScenes training completed successfully."

        set +e
        wait "$train_pid"
        set -e

        return 0
      fi

      if [[ "$line" == *"download complete"* ]] ||
         [[ "$line" =~ epoch=[0-9]+.*loss= ]]; then
        slow_since=0
      fi

      if [[ "$line" =~ download\ progress.*avg=([0-9.]+)\ Mbps ]]; then
        local avg="${BASH_REMATCH[1]}"

        if awk -v a="$avg" -v b="$MIN_DOWNLOAD_MBPS" \
             'BEGIN { exit !(a < b) }'; then

          now="$(date +%s)"

          if (( slow_since == 0 )); then
            slow_since="$now"
            log "Slow download detected: avg=${avg} Mbps."
          elif (( now - slow_since >= SLOW_SECONDS )); then
            restart_reason="slow"
            log "Slow download persisted for ${SLOW_SECONDS}s."
            break
          fi
        else
          slow_since=0
        fi
      fi
    fi

    now="$(date +%s)"

    # Schedule is controlled ONLY here, on the host.
    if ! allowed_now; then
      restart_reason="schedule"
      log "Training window closed at 08:00. Releasing GPU."
      break
    fi

    # External GPU watchdog.
    if (( now - last_gpu_check >= GPU_CHECK_SECONDS )); then
      last_gpu_check="$now"

      if ! docker exec "$CONT" nvidia-smi >/dev/null 2>&1; then
        restart_reason="cuda"
        log "GPU disappeared from inside Docker."
        break
      fi
    fi

    # No-output watchdog.
    if (( now - last_output >= STALL_SECONDS )); then
      restart_reason="stall"
      log "No training output for ${STALL_SECONDS}s."
      break
    fi
  done

  if [[ "$restart_reason" == "schedule" ]]; then
    stop_container_for_schedule

    set +e
    wait "$train_pid" >/dev/null 2>&1
    set -e

    return 20
  fi

  if [[ -n "$restart_reason" ]]; then
    stop_training
    cleanup_partial_downloads

    set +e
    wait "$train_pid" >/dev/null 2>&1
    set -e

    if [[ "$restart_reason" == "cuda" ]]; then
      log "CUDA/GPU recovery requested. Restarting Docker."

      docker stop "$CONT" >/dev/null 2>&1 || true
      systemctl restart docker
      sleep 10
    fi

    return 1
  fi

  set +e
  wait "$train_pid"
  local rc=$?
  set -e

  log "Training process exited with code $rc."
  cleanup_partial_downloads

  return "$rc"
}

main() {
  log "Starting single host-level KITScenes watchdog."

  while true; do

    # ------------------------------------------------------------
    # BLOCKED WINDOW: 08:00 -> 18:59
    # ------------------------------------------------------------
    if ! allowed_now; then
      stop_container_for_schedule

      local sleep_seconds
      sleep_seconds="$(seconds_until_allowed)"

      log "Training blocked by schedule. GPU released."
      log "Sleeping until 19:00 (${sleep_seconds}s)."

      sleep "$sleep_seconds"
      continue
    fi

    # ------------------------------------------------------------
    # ALLOWED WINDOW: 19:00 -> 07:59
    # ------------------------------------------------------------
    if ! preflight_storage; then
      log "Storage/checkpoint preflight failed. Retrying in ${RESTART_SECONDS}s."
      sleep "$RESTART_SECONDS"
      continue
    fi

    if ! ensure_container_gpu; then
      log "GPU preflight failed. Retrying in ${RESTART_SECONDS}s."
      sleep "$RESTART_SECONDS"
      continue
    fi

    launch_training
    rc=$?

    if (( rc == 0 )); then
      log "Training is complete. Keeping watchdog alive; no further training will be launched."

      stop_container_for_schedule

      while true; do
        sleep 3600
      done
    fi

    if (( rc == 20 )); then
      # Schedule closed. Main loop will calculate exact sleep-to-19:00.
      continue
    fi

    log "Training needs recovery. Retrying in ${RESTART_SECONDS}s."
    sleep "$RESTART_SECONDS"
  done
}

main
