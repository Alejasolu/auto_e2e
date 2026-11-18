#!/usr/bin/env bash
set -u

ROOT="/auto_e2e"
LOG="$ROOT/checkpoints/kitscenes_bounded/train.log"
CKPT="$ROOT/checkpoints/kitscenes_bounded/last.pt"
DATA="$ROOT/datasets/kitscenes_bounded_cache"

CHECK_SECONDS="${CHECK_SECONDS:-60}"
STALL_SECONDS="${STALL_SECONDS:-2700}"   # 45 min without log growth = stuck
DOWNLOAD_MBPS="${DOWNLOAD_MBPS:-60}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KITSCENES_TRAINING_TIME_BLOCK=1
export KITSCENES_TRAINING_TZ=Europe/Rome
export KITSCENES_BLOCKED_FROM_HOUR=8
export KITSCENES_ALLOWED_FROM_HOUR=19
export KITSCENES_ALLOW_WEEKEND_ALL_DAY=1

allowed_now() {
python - <<'PY'
from datetime import datetime
from zoneinfo import ZoneInfo
import os

tz = os.environ.get("KITSCENES_TRAINING_TZ", "Europe/Rome")
now = datetime.now(ZoneInfo(tz))

if os.environ.get("KITSCENES_ALLOW_WEEKEND_ALL_DAY", "1").lower() in {"1","true","yes","y"} and now.weekday() >= 5:
    print("yes")
    raise SystemExit

blocked_from = int(os.environ.get("KITSCENES_BLOCKED_FROM_HOUR", "8"))
allowed_from = int(os.environ.get("KITSCENES_ALLOWED_FROM_HOUR", "19"))

# Block 08:00 <= hour < 19:00
if blocked_from <= now.hour < allowed_from:
    print("no")
else:
    print("yes")
PY
}

cleanup_partial_downloads() {
  rm -rf "$DATA/.kitscenes_staging"
  rm -rf "$DATA/.cache"
}

while true; do
  cd "$ROOT" || exit 1
  source /root/e2e-env/bin/activate

  if [ ! -f "$CKPT" ]; then
    echo "[SUPERVISOR] ERROR: checkpoint not found: $CKPT"
    exit 2
  fi

  cleanup_partial_downloads

  echo
  echo "[SUPERVISOR] $(date) starting/resuming KITScenes training"
  echo "[SUPERVISOR] checkpoint: $CKPT"
  echo "[SUPERVISOR] download limit: ${DOWNLOAD_MBPS} Mbps"
  echo "[SUPERVISOR] stall watchdog: ${STALL_SECONDS}s"
  echo

  export LOG
  setsid bash -c "
    cd '$ROOT'
    source /root/e2e-env/bin/activate
    python '$ROOT/train_kitscenes_bounded_shards.py' \
      --epochs 5 \
      --accum-steps 2 \
      --save-every-scenes 1 \
      --validate-every-scenes 50 \
      --val-scenes-per-eval 5 \
      --shuffle-scenes \
      --download-rate-mbps '$DOWNLOAD_MBPS' \
      --download-pause-seconds 5 \
      --resume '$CKPT' \
      2>&1 | tee -a '$LOG'
  " &
  runner_pid=$!

  while kill -0 "$runner_pid" 2>/dev/null; do
    sleep "$CHECK_SECONDS"

    [ -f "$LOG" ] || continue

    now_ts=$(date +%s)
    log_ts=$(stat -c %Y "$LOG")
    age=$((now_ts - log_ts))
    allowed=$(allowed_now)

    if [ "$allowed" = "yes" ] && [ "$age" -gt "$STALL_SECONDS" ]; then
      echo "[SUPERVISOR] $(date) STALL DETECTED: log unchanged for ${age}s during allowed time."
      echo "[SUPERVISOR] Killing training process group and restarting from checkpoint."
      kill -TERM -- "-$runner_pid" 2>/dev/null || true
      sleep 20
      kill -KILL -- "-$runner_pid" 2>/dev/null || true
      wait "$runner_pid" 2>/dev/null || true
      cleanup_partial_downloads
      break
    fi
  done

  if grep -q "Bounded shard training complete" "$LOG" 2>/dev/null; then
    echo "[SUPERVISOR] Training complete."
    exit 0
  fi

  echo "[SUPERVISOR] $(date) training exited/stopped; restarting in 60 seconds..."
  sleep 60
done
