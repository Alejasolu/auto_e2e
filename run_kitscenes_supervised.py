#!/usr/bin/env python3
import os
import re
import sys
import time
import signal
import select
import subprocess
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path("/auto_e2e")
TRAIN_SCRIPT = ROOT / "train_kitscenes_bounded_shards.py"
LOG = ROOT / "checkpoints/kitscenes_bounded/train.log"
CKPT = ROOT / "checkpoints/kitscenes_bounded/last.pt"
DATA = ROOT / "datasets/kitscenes_bounded_cache"

CHECK_SECONDS = int(os.environ.get("CHECK_SECONDS", "60"))
STALL_SECONDS = int(os.environ.get("STALL_SECONDS", "1800"))          # 30 min no output
SLOW_SECONDS = int(os.environ.get("SLOW_SECONDS", "900"))             # 15 min slow download
MIN_DOWNLOAD_MBPS = float(os.environ.get("MIN_DOWNLOAD_MBPS", "2"))   # kill if avg < 2 Mbps
DOWNLOAD_MBPS = float(os.environ.get("DOWNLOAD_MBPS", "60"))

progress_re = re.compile(r"download progress scene=.*?avg=([0-9.]+) Mbps")


def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_supervisor(msg):
    text = f"[SUPERVISOR] {timestamp()} {msg}"
    print(text, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8", errors="replace") as f:
        f.write(text + "\n")


def cleanup_partial_downloads():
    for d in [DATA / ".kitscenes_staging", DATA / ".cache"]:
        if d.exists():
            subprocess.run(["rm", "-rf", str(d)], check=False)


def allowed_now():
    tz = os.environ.get("KITSCENES_TRAINING_TZ", "Europe/Rome")
    now = datetime.now(ZoneInfo(tz))

    allow_weekend = os.environ.get(
        "KITSCENES_ALLOW_WEEKEND_ALL_DAY", "1"
    ).strip().lower() in {"1", "true", "yes", "y"}

    if allow_weekend and now.weekday() >= 5:
        return True

    blocked_from = int(os.environ.get("KITSCENES_BLOCKED_FROM_HOUR", "8"))
    allowed_from = int(os.environ.get("KITSCENES_ALLOWED_FROM_HOUR", "19"))

    return not (blocked_from <= now.hour < allowed_from)


def kill_process_group(proc, reason):
    write_supervisor(reason)
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    time.sleep(20)

    if proc.poll() is None:
        write_supervisor("Process did not stop after SIGTERM. Sending SIGKILL.")
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def launch_once():
    if not CKPT.exists():
        write_supervisor(f"ERROR: checkpoint not found: {CKPT}")
        sys.exit(2)

    cleanup_partial_downloads()

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    env["KITSCENES_TRAINING_TIME_BLOCK"] = "1"
    env["KITSCENES_TRAINING_TZ"] = "Europe/Rome"
    env["KITSCENES_BLOCKED_FROM_HOUR"] = "8"
    env["KITSCENES_ALLOWED_FROM_HOUR"] = "19"
    env["KITSCENES_ALLOW_WEEKEND_ALL_DAY"] = "1"

    env["KITSCENES_URL_TIMEOUT_SECONDS"] = os.environ.get("KITSCENES_URL_TIMEOUT_SECONDS", "120")
    env["KITSCENES_MIN_DOWNLOAD_MBPS"] = str(MIN_DOWNLOAD_MBPS)
    env["KITSCENES_MIN_DOWNLOAD_GRACE_SECONDS"] = os.environ.get(
        "KITSCENES_MIN_DOWNLOAD_GRACE_SECONDS", "900"
    )

    cmd = [
        "/root/e2e-env/bin/python",
        str(TRAIN_SCRIPT),
        "--epochs", "5",
        "--accum-steps", "2",
        "--save-every-scenes", "1",
        "--validate-every-scenes", "50",
        "--val-scenes-per-eval", "5",
        "--shuffle-scenes",
        "--download-rate-mbps", str(DOWNLOAD_MBPS),
        "--download-pause-seconds", "5",
        "--resume", str(CKPT),
    ]

    write_supervisor("Starting/resuming KITScenes training")
    write_supervisor(f"checkpoint: {CKPT}")
    write_supervisor(f"download limit: {DOWNLOAD_MBPS:g} Mbps")
    write_supervisor(f"min download speed: {MIN_DOWNLOAD_MBPS:g} Mbps after grace period")
    write_supervisor(f"stall watchdog: {STALL_SECONDS}s without output during allowed time")
    write_supervisor("weekend training: enabled all day")

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )

    last_output_time = time.time()
    slow_since = None

    LOG.parent.mkdir(parents=True, exist_ok=True)

    with LOG.open("a", encoding="utf-8", errors="replace") as log_file:
        while proc.poll() is None:
            ready, _, _ = select.select([proc.stdout], [], [], CHECK_SECONDS)

            if ready:
                line = proc.stdout.readline()
                if line:
                    print(line, end="", flush=True)
                    log_file.write(line)
                    log_file.flush()

                    last_output_time = time.time()

                    if (
                        "device: cpu" in line
                        or "No CUDA GPUs are available" in line
                        or "CUDA is unavailable inside Docker" in line
                        or "cuda available: False" in line
                    ):
                        write_supervisor(
                            "GPU/CUDA failure detected inside Docker. "
                            "Exiting so the host systemd service can restart Docker and relaunch safely."
                        )
                        kill_process_group(proc, "Stopping failed CPU/CUDA run.")
                        sys.exit(77)

                    if (
                        "device: cpu" in line
                        or "No CUDA GPUs are available" in line
                        or "CUDA is unavailable inside Docker" in line
                        or "cuda available: False" in line
                    ):
                        write_supervisor(
                            "GPU/CUDA failure detected inside Docker. "
                            "Exiting so the host systemd service can restart Docker and relaunch safely."
                        )
                        kill_process_group(proc, "Stopping failed CPU/CUDA run.")
                        sys.exit(77)

                    if "Bounded shard training complete" in line:
                        write_supervisor("Training complete.")
                        return 0

                    if "download complete" in line or re.search(r"epoch=\d+ .*loss=", line):
                        slow_since = None

                    m = progress_re.search(line)
                    if m and allowed_now():
                        avg = float(m.group(1))
                        if avg < MIN_DOWNLOAD_MBPS:
                            if slow_since is None:
                                slow_since = time.time()
                                write_supervisor(
                                    f"Slow download detected: avg={avg:.2f} Mbps. "
                                    f"Will restart if still slow for {SLOW_SECONDS}s."
                                )
                            elif time.time() - slow_since > SLOW_SECONDS:
                                kill_process_group(
                                    proc,
                                    f"Slow download persisted for >{SLOW_SECONDS}s. Restarting from checkpoint.",
                                )
                                cleanup_partial_downloads()
                                return 1
                        else:
                            slow_since = None

            age = time.time() - last_output_time

            if allowed_now() and age > STALL_SECONDS:
                kill_process_group(
                    proc,
                    f"No output for {int(age)}s during allowed time. Restarting from checkpoint.",
                )
                cleanup_partial_downloads()
                return 1

        rc = proc.returncode
        write_supervisor(f"Training process exited with code {rc}.")
        cleanup_partial_downloads()
        return rc if rc is not None else 1


def main():
    while True:
        rc = launch_once()

        if rc == 0:
            return

        write_supervisor("Restarting in 60 seconds...")
        time.sleep(60)


if __name__ == "__main__":
    main()
