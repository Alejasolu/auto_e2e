import os
import re
import time
from pathlib import Path

LOG = Path("/auto_e2e/checkpoints/kitscenes_bounded/train.log")
TOTAL_TRAIN_SCENES = 585
REFRESH_SECONDS = 10

train_line_re = re.compile(r"epoch=(\d+)\s+(?:scene|train_scene)=(\d+)/(\d+)")
epoch_re = re.compile(r"Epoch\s+(\d+)/(\d+)")
validating_re = re.compile(r"Validating on\s+(\d+)\s+bounded val shard")
val_scene_re = re.compile(r"val_scene=(\d+)/(\d+)")

start_time = time.time()
start_attempted = None


def bar(done, total, width=36):
    if not total:
        return "[" + "-" * width + "]"
    done = max(0, min(done, total))
    filled = int(width * done / total)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def read_lines():
    if not LOG.exists():
        return []
    try:
        return LOG.read_text(errors="replace").splitlines()
    except Exception:
        return []


def parse_progress(lines):
    current_epoch = None
    total_epochs = None

    last_epoch_line_idx = 0
    for i, line in enumerate(lines):
        m = epoch_re.search(line)
        if m:
            current_epoch = int(m.group(1))
            total_epochs = int(m.group(2))
            last_epoch_line_idx = i

    current_lines = lines[last_epoch_line_idx:]

    latest_scene_number = 0
    latest_scene_total = TOTAL_TRAIN_SCENES
    success_count = 0

    for line in current_lines:
        m = train_line_re.search(line)
        if m:
            success_count += 1
            latest_scene_number = int(m.group(2))
            latest_scene_total = int(m.group(3))

    skipped_count = sum(
        1 for line in current_lines
        if "WARNING: train scene skipped" in line
    )

    attempted = max(latest_scene_number, success_count + skipped_count)
    total = latest_scene_total or TOTAL_TRAIN_SCENES

    # Validation state: look only after the last "Validating on..."
    val_total = 0
    val_done = 0
    validating = False

    last_val_start = None
    for i, line in enumerate(lines):
        if validating_re.search(line):
            last_val_start = i

    if last_val_start is not None:
        val_lines = lines[last_val_start:]
        m = validating_re.search(lines[last_val_start])
        if m:
            val_total = int(m.group(1))

        val_success = 0
        for line in val_lines:
            vm = val_scene_re.search(line)
            if vm:
                val_success = max(val_success, int(vm.group(1)))
                val_total = max(val_total, int(vm.group(2)))

        val_skipped = sum(
            1 for line in val_lines
            if "WARNING: validation scene skipped" in line
        )

        val_done = max(val_success, val_skipped)

        validation_finished = any(
            "bounded_val_loss=" in line or "end_val_loss=" in line
            for line in val_lines
        )

        validating = val_total > 0 and not validation_finished

    return {
        "current_epoch": current_epoch,
        "total_epochs": total_epochs,
        "attempted": attempted,
        "total": total,
        "success_count": success_count,
        "skipped_count": skipped_count,
        "validating": validating,
        "val_done": val_done,
        "val_total": val_total,
        "last_lines": lines[-8:],
    }


def format_eta(attempted):
    global start_attempted

    now = time.time()

    if start_attempted is None:
        start_attempted = attempted
        return "collecting speed..."

    delta_scenes = attempted - start_attempted
    elapsed = now - start_time

    if delta_scenes <= 0 or elapsed < 30:
        return "collecting speed..."

    scenes_per_hour = delta_scenes / elapsed * 3600
    remaining = max(0, TOTAL_TRAIN_SCENES - attempted)

    if scenes_per_hour <= 0:
        return "unknown"

    hours_left = remaining / scenes_per_hour
    return f"{hours_left:.1f} h left at {scenes_per_hour:.1f} scenes/hour"


while True:
    lines = read_lines()
    p = parse_progress(lines)

    attempted = p["attempted"]
    total = p["total"]
    percent = attempted / total * 100 if total else 0

    os.system("clear")

    print("KITScenes bounded training monitor")
    print("=" * 48)

    if p["current_epoch"] is not None:
        print(f"Epoch: {p['current_epoch']}/{p['total_epochs']}")
    else:
        print("Epoch: not detected yet")

    print()
    print("Training shards")
    print(f"{bar(attempted, total)} {attempted}/{total} ({percent:.1f}%)")
    print(f"successful scene logs this epoch: {p['success_count']}")
    print(f"skipped train scenes this epoch:  {p['skipped_count']}")
    print(f"ETA: {format_eta(attempted)}")

    print()
    print("Validation")
    if p["validating"]:
        vpct = p["val_done"] / p["val_total"] * 100 if p["val_total"] else 0
        print(f"{bar(p['val_done'], p['val_total'])} {p['val_done']}/{p['val_total']} ({vpct:.1f}%)")
    else:
        print("not currently validating")

    print()
    print("Disk usage")
    os.system("du -sh /auto_e2e/datasets/kitscenes_bounded_cache /auto_e2e/checkpoints/kitscenes_bounded 2>/dev/null")

    print()
    print("Latest log lines")
    print("-" * 48)
    for line in p["last_lines"]:
        print(line)

    time.sleep(REFRESH_SECONDS)
