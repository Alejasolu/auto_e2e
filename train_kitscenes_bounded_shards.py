import argparse
import json
import math
import os
import random
import shutil
import sys
import tempfile
import urllib.request
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append("/auto_e2e/Model")

from data_parsing.kit_scenes.source import PinnedKITScenesDownloader, sdk_split_scene_ids
from data_parsing.kit_scenes.dataset import KitScenesDataset
from model_components.auto_e2e import AutoE2E
from model_components.losses import TrajectoryImitationLoss


def gpu_mem(label):
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"{label}: free={free/1024**3:.2f} GB / total={total/1024**3:.2f} GB", flush=True)


def normalize_image_tensor(x):
    x = x.float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device)

    if x.ndim == 5:
        mean = mean.view(1, 1, 3, 1, 1)
        std = std.view(1, 1, 3, 1, 1)
    elif x.ndim == 4:
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    else:
        raise ValueError(f"Unexpected image tensor rank: {x.ndim}")

    return (x - mean) / std


def collate_auto_e2e(batch):
    visual_tiles = torch.stack([b["visual_tiles"] for b in batch])
    map_context = torch.stack([b["map_tile"] for b in batch])

    map_context = F.interpolate(
        map_context,
        size=(256, 256),
        mode="bilinear",
        align_corners=False,
    )

    egomotion_history = torch.stack([b["egomotion_history"] for b in batch])
    visual_history = torch.stack([b["visual_history"] for b in batch])
    trajectory_target = torch.stack([b["trajectory_target"] for b in batch])

    batch_size = visual_tiles.shape[0]
    height = map_context.shape[-2]
    width = map_context.shape[-1]

    return {
        "visual_tiles": visual_tiles,
        "map_context": map_context,
        "route_mask": torch.zeros(batch_size, 2, height, width, dtype=torch.float32),
        "map_valid": torch.ones(batch_size, dtype=torch.bool),
        "route_valid": torch.zeros(batch_size, dtype=torch.bool),
        "egomotion_history": egomotion_history,
        "visual_history": visual_history,
        "trajectory_target": trajectory_target,
    }


def move_batch_to_device(batch, device):
    return {
        "visual_tiles": normalize_image_tensor(batch["visual_tiles"]).to(device, non_blocking=True),
        "map_context": normalize_image_tensor(batch["map_context"]).to(device, non_blocking=True),
        "visual_history": batch["visual_history"].to(device, non_blocking=True),
        "egomotion_history": batch["egomotion_history"].to(device, non_blocking=True),
        "trajectory_target": batch["trajectory_target"].to(device, non_blocking=True),
        "route_mask": batch["route_mask"].to(device, non_blocking=True),
        "map_valid": batch["map_valid"].to(device, non_blocking=True),
        "route_valid": batch["route_valid"].to(device, non_blocking=True),
    }


def scene_dir(data_root, split, scene_id):
    return data_root / "data" / split / scene_id


def cleanup_scene(data_root, split, scene_id):
    d = scene_dir(data_root, split, scene_id)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)

    # Remove HF/extraction temp dirs but keep manifest if present.
    shutil.rmtree(data_root / ".kitscenes_staging", ignore_errors=True)
    shutil.rmtree(data_root / ".cache", ignore_errors=True)




def _get_hf_token(downloader):
    """Return a Hugging Face token from downloader/env/local login if available."""
    if isinstance(getattr(downloader, "token", None), str) and downloader.token:
        return downloader.token

    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value:
            return value

    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            return token
    except Exception:
        pass

    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        token = token_path.read_text().strip()
        if token:
            return token

    return None



def _current_scheduled_rate_mbps(default_rate_mbps):
    """
    Return current download limit in Mbps.

    Schedule:
      limited_from_hour <= current hour < unlimited_from_hour: day/limited rate
      otherwise: night/unlimited rate

    A rate <= 0 means unlimited.
    """
    enabled = os.environ.get("KITSCENES_BANDWIDTH_SCHEDULE", "0") == "1"
    if not enabled:
        return default_rate_mbps

    tz_name = os.environ.get("KITSCENES_BANDWIDTH_TZ", "Europe/Rome")
    limited_from = int(os.environ.get("KITSCENES_LIMITED_FROM_HOUR", "8"))
    unlimited_from = int(os.environ.get("KITSCENES_UNLIMITED_FROM_HOUR", "18"))

    day_rate = float(os.environ.get("KITSCENES_DAY_DOWNLOAD_RATE_MBPS", str(default_rate_mbps)))
    night_rate = float(os.environ.get("KITSCENES_NIGHT_DOWNLOAD_RATE_MBPS", "0"))

    now = datetime.now(ZoneInfo(tz_name))
    hour = now.hour

    if limited_from <= unlimited_from:
        is_limited_time = limited_from <= hour < unlimited_from
    else:
        is_limited_time = hour >= limited_from or hour < unlimited_from

    return day_rate if is_limited_time else night_rate


def _rate_label(rate_mbps):
    if rate_mbps is None or rate_mbps <= 0:
        return "unlimited"
    return f"{rate_mbps:g} Mbps"



def _training_block_enabled():
    return os.environ.get("KITSCENES_TRAINING_TIME_BLOCK", "0") == "1"


def _now_local():
    tz_name = os.environ.get("KITSCENES_TRAINING_TZ", "Europe/Rome")
    return datetime.now(ZoneInfo(tz_name))


def _is_training_blocked_now():
    """
    True means: do not download, train, or validate now.

    Default weekday behavior:
      08:00–18:59 blocked
      19:00–07:59 allowed

    Weekend override:
      if KITSCENES_ALLOW_WEEKEND_ALL_DAY=1, Saturday and Sunday are always allowed.
    """
    if not _training_block_enabled():
        return False

    now = _now_local()

    allow_weekend = os.environ.get(
        "KITSCENES_ALLOW_WEEKEND_ALL_DAY", "0"
    ).strip().lower() in {"1", "true", "yes", "y"}

    # Python weekday(): Monday=0 ... Saturday=5, Sunday=6
    if allow_weekend and now.weekday() >= 5:
        return False

    blocked_from = int(os.environ.get("KITSCENES_BLOCKED_FROM_HOUR", "8"))
    allowed_from = int(os.environ.get("KITSCENES_ALLOWED_FROM_HOUR", "19"))

    hour = now.hour

    if blocked_from < allowed_from:
        return blocked_from <= hour < allowed_from
    else:
        return hour >= blocked_from or hour < allowed_from


def _seconds_until_training_allowed():
    now = _now_local()
    allowed_from = int(os.environ.get("KITSCENES_ALLOWED_FROM_HOUR", "19"))

    target = now.replace(hour=allowed_from, minute=0, second=0, microsecond=0)
    if target <= now:
        target = target + timedelta(days=1)

    return max(1, int((target - now).total_seconds()))


def _wait_if_training_blocked(context):
    if not _training_block_enabled():
        return

    while _is_training_blocked_now():
        seconds = _seconds_until_training_allowed()
        minutes = seconds / 60
        now = _now_local().strftime("%Y-%m-%d %H:%M:%S %Z")

        print(
            f"TRAINING PAUSED by schedule during {context}. "
            f"Now={now}. Waiting until allowed hour. "
            f"Sleep={minutes:.1f} min.",
            flush=True,
        )

        time.sleep(min(seconds, 300))


def _download_archive_rate_limited(downloader, data_root, split, scene_id, rate_mbps):
    """Download and extract one KITScenes archive with an optional Mbps limit."""
    from huggingface_hub import hf_hub_url
    from kitscenes.download.downloader import _extract_tar_file

    archive = downloader.archives[scene_id]

    if archive.split != split:
        raise ValueError(
            f"KITScenes scene {scene_id} belongs to {archive.split!r}, not {split!r}"
        )

    staging_root = data_root / ".kitscenes_staging"
    staging_root.mkdir(parents=True, exist_ok=True)

    active_rate_mbps = _current_scheduled_rate_mbps(rate_mbps)
    active_rate_bytes_per_second = None
    if active_rate_mbps is not None and active_rate_mbps > 0:
        active_rate_bytes_per_second = active_rate_mbps * 1_000_000 / 8.0
    last_rate_mbps = active_rate_mbps

    with tempfile.TemporaryDirectory(prefix="kitscenes_dl_", dir=staging_root) as temporary:
        temporary = Path(temporary)
        tar_path = temporary / Path(archive.filename).name

        url = hf_hub_url(
            repo_id=downloader.repo_id,
            filename=archive.filename,
            repo_type="dataset",
            revision=downloader.revision,
        )

        headers = {"User-Agent": "auto_e2e-kitscenes-bounded-trainer"}
        hf_token = _get_hf_token(downloader)
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
        else:
            print(
                "WARNING: no Hugging Face token found for direct throttled download. "
                "If this dataset requires auth, the download may fail.",
                flush=True,
            )

        request = urllib.request.Request(url, headers=headers)

        print(
            f"Downloading one shard with current limit: "
            f"{_rate_label(active_rate_mbps)}",
            flush=True,
        )

        started = time.time()
        last_print = started
        downloaded = 0
        chunk_size = 1024 * 1024

        with urllib.request.urlopen(request) as response, tar_path.open("wb") as out:
            while True:
                _wait_if_training_blocked("download")
                chunk = response.read(chunk_size)
                if not chunk:
                    break

                out.write(chunk)
                downloaded += len(chunk)

                now = time.time()
                elapsed = max(1e-6, now - started)

                active_rate_mbps = _current_scheduled_rate_mbps(rate_mbps)

                if active_rate_mbps != last_rate_mbps:
                    print(
                        f"Bandwidth schedule changed: "
                        f"{_rate_label(last_rate_mbps)} -> {_rate_label(active_rate_mbps)}",
                        flush=True,
                    )
                    last_rate_mbps = active_rate_mbps

                active_rate_bytes_per_second = None
                if active_rate_mbps is not None and active_rate_mbps > 0:
                    active_rate_bytes_per_second = active_rate_mbps * 1_000_000 / 8.0

                if active_rate_bytes_per_second is not None:
                    target_chunk_seconds = len(chunk) / active_rate_bytes_per_second
                    time.sleep(target_chunk_seconds)

                now = time.time()
                if now - last_print >= 10:
                    avg_mbps = downloaded * 8 / max(1e-6, now - started) / 1_000_000
                    pct = downloaded / archive.size_bytes * 100
                    print(
                        f"download progress scene={scene_id} "
                        f"{pct:.1f}% avg={avg_mbps:.1f} Mbps "
                        f"limit={_rate_label(active_rate_mbps)} "
                        f"downloaded={downloaded/1024**3:.2f}GB",
                        flush=True,
                    )
                    last_print = now

        elapsed = max(1e-6, time.time() - started)
        avg_mbps = downloaded * 8 / elapsed / 1_000_000

        print(
            f"download complete scene={scene_id} "
            f"size={downloaded/1024**3:.2f}GB "
            f"time_min={elapsed/60:.1f} "
            f"avg={avg_mbps:.1f} Mbps",
            flush=True,
        )

        actual_size = tar_path.stat().st_size
        if actual_size != archive.size_bytes:
            raise ValueError(
                f"KITScenes archive size mismatch for {scene_id}: "
                f"expected {archive.size_bytes}, got {actual_size}"
            )

        _extract_tar_file(
            tar_path,
            data_root / "data" / archive.split,
            expected_sha256=archive.sha256,
            size_bytes=archive.size_bytes,
            scene_id=scene_id,
        )

    shutil.rmtree(staging_root, ignore_errors=True)
    shutil.rmtree(data_root / ".cache", ignore_errors=True)




def _get_hf_token(downloader):
    """Return a Hugging Face token from downloader/env/local login if available."""
    if isinstance(getattr(downloader, "token", None), str) and downloader.token:
        return downloader.token

    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value:
            return value

    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            return token
    except Exception:
        pass

    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        token = token_path.read_text().strip()
        if token:
            return token

    return None



def _current_scheduled_rate_mbps(default_rate_mbps):
    """
    Return current download limit in Mbps.

    Schedule:
      limited_from_hour <= current hour < unlimited_from_hour: day/limited rate
      otherwise: night/unlimited rate

    A rate <= 0 means unlimited.
    """
    enabled = os.environ.get("KITSCENES_BANDWIDTH_SCHEDULE", "0") == "1"
    if not enabled:
        return default_rate_mbps

    tz_name = os.environ.get("KITSCENES_BANDWIDTH_TZ", "Europe/Rome")
    limited_from = int(os.environ.get("KITSCENES_LIMITED_FROM_HOUR", "8"))
    unlimited_from = int(os.environ.get("KITSCENES_UNLIMITED_FROM_HOUR", "18"))

    day_rate = float(os.environ.get("KITSCENES_DAY_DOWNLOAD_RATE_MBPS", str(default_rate_mbps)))
    night_rate = float(os.environ.get("KITSCENES_NIGHT_DOWNLOAD_RATE_MBPS", "0"))

    now = datetime.now(ZoneInfo(tz_name))
    hour = now.hour

    if limited_from <= unlimited_from:
        is_limited_time = limited_from <= hour < unlimited_from
    else:
        is_limited_time = hour >= limited_from or hour < unlimited_from

    return day_rate if is_limited_time else night_rate


def _rate_label(rate_mbps):
    if rate_mbps is None or rate_mbps <= 0:
        return "unlimited"
    return f"{rate_mbps:g} Mbps"



def _training_block_enabled():
    return os.environ.get("KITSCENES_TRAINING_TIME_BLOCK", "0") == "1"


def _now_local():
    tz_name = os.environ.get("KITSCENES_TRAINING_TZ", "Europe/Rome")
    return datetime.now(ZoneInfo(tz_name))


def _is_training_blocked_now():
    """
    True means: do not download, train, or validate now.

    Default weekday behavior:
      08:00–18:59 blocked
      19:00–07:59 allowed

    Weekend override:
      if KITSCENES_ALLOW_WEEKEND_ALL_DAY=1, Saturday and Sunday are always allowed.
    """
    if not _training_block_enabled():
        return False

    now = _now_local()

    allow_weekend = os.environ.get(
        "KITSCENES_ALLOW_WEEKEND_ALL_DAY", "0"
    ).strip().lower() in {"1", "true", "yes", "y"}

    # Python weekday(): Monday=0 ... Saturday=5, Sunday=6
    if allow_weekend and now.weekday() >= 5:
        return False

    blocked_from = int(os.environ.get("KITSCENES_BLOCKED_FROM_HOUR", "8"))
    allowed_from = int(os.environ.get("KITSCENES_ALLOWED_FROM_HOUR", "19"))

    hour = now.hour

    if blocked_from < allowed_from:
        return blocked_from <= hour < allowed_from
    else:
        return hour >= blocked_from or hour < allowed_from


def _seconds_until_training_allowed():
    now = _now_local()
    allowed_from = int(os.environ.get("KITSCENES_ALLOWED_FROM_HOUR", "19"))

    target = now.replace(hour=allowed_from, minute=0, second=0, microsecond=0)
    if target <= now:
        target = target + timedelta(days=1)

    return max(1, int((target - now).total_seconds()))


def _wait_if_training_blocked(context):
    if not _training_block_enabled():
        return

    while _is_training_blocked_now():
        seconds = _seconds_until_training_allowed()
        minutes = seconds / 60
        now = _now_local().strftime("%Y-%m-%d %H:%M:%S %Z")

        print(
            f"TRAINING PAUSED by schedule during {context}. "
            f"Now={now}. Waiting until allowed hour. "
            f"Sleep={minutes:.1f} min.",
            flush=True,
        )

        time.sleep(min(seconds, 300))


def _download_archive_rate_limited(downloader, data_root, split, scene_id, rate_mbps):
    """Download and extract one KITScenes archive with an optional Mbps limit."""
    from huggingface_hub import hf_hub_url
    from kitscenes.download.downloader import _extract_tar_file

    archive = downloader.archives[scene_id]

    if archive.split != split:
        raise ValueError(
            f"KITScenes scene {scene_id} belongs to {archive.split!r}, not {split!r}"
        )

    staging_root = data_root / ".kitscenes_staging"
    staging_root.mkdir(parents=True, exist_ok=True)

    active_rate_mbps = _current_scheduled_rate_mbps(rate_mbps)
    active_rate_bytes_per_second = None
    if active_rate_mbps is not None and active_rate_mbps > 0:
        active_rate_bytes_per_second = active_rate_mbps * 1_000_000 / 8.0
    last_rate_mbps = active_rate_mbps

    with tempfile.TemporaryDirectory(prefix="kitscenes_dl_", dir=staging_root) as temporary:
        temporary = Path(temporary)
        tar_path = temporary / Path(archive.filename).name

        url = hf_hub_url(
            repo_id=downloader.repo_id,
            filename=archive.filename,
            repo_type="dataset",
            revision=downloader.revision,
        )

        headers = {"User-Agent": "auto_e2e-kitscenes-bounded-trainer"}
        hf_token = _get_hf_token(downloader)
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
        else:
            print(
                "WARNING: no Hugging Face token found for direct throttled download. "
                "If this dataset requires auth, the download may fail.",
                flush=True,
            )

        request = urllib.request.Request(url, headers=headers)

        print(
            f"Downloading one shard with current limit: "
            f"{_rate_label(active_rate_mbps)}",
            flush=True,
        )

        started = time.time()
        last_print = started
        downloaded = 0
        chunk_size = 1024 * 1024

        with urllib.request.urlopen(request) as response, tar_path.open("wb") as out:
            while True:
                _wait_if_training_blocked("download")
                chunk = response.read(chunk_size)
                if not chunk:
                    break

                out.write(chunk)
                downloaded += len(chunk)

                now = time.time()
                elapsed = max(1e-6, now - started)

                active_rate_mbps = _current_scheduled_rate_mbps(rate_mbps)

                if active_rate_mbps != last_rate_mbps:
                    print(
                        f"Bandwidth schedule changed: "
                        f"{_rate_label(last_rate_mbps)} -> {_rate_label(active_rate_mbps)}",
                        flush=True,
                    )
                    last_rate_mbps = active_rate_mbps

                active_rate_bytes_per_second = None
                if active_rate_mbps is not None and active_rate_mbps > 0:
                    active_rate_bytes_per_second = active_rate_mbps * 1_000_000 / 8.0

                if active_rate_bytes_per_second is not None:
                    target_chunk_seconds = len(chunk) / active_rate_bytes_per_second
                    time.sleep(target_chunk_seconds)

                now = time.time()
                if now - last_print >= 10:
                    avg_mbps = downloaded * 8 / max(1e-6, now - started) / 1_000_000
                    pct = downloaded / archive.size_bytes * 100
                    print(
                        f"download progress scene={scene_id} "
                        f"{pct:.1f}% avg={avg_mbps:.1f} Mbps "
                        f"limit={_rate_label(active_rate_mbps)} "
                        f"downloaded={downloaded/1024**3:.2f}GB",
                        flush=True,
                    )
                    last_print = now

        elapsed = max(1e-6, time.time() - started)
        avg_mbps = downloaded * 8 / elapsed / 1_000_000

        print(
            f"download complete scene={scene_id} "
            f"size={downloaded/1024**3:.2f}GB "
            f"time_min={elapsed/60:.1f} "
            f"avg={avg_mbps:.1f} Mbps",
            flush=True,
        )

        actual_size = tar_path.stat().st_size
        if actual_size != archive.size_bytes:
            raise ValueError(
                f"KITScenes archive size mismatch for {scene_id}: "
                f"expected {archive.size_bytes}, got {actual_size}"
            )

        _extract_tar_file(
            tar_path,
            data_root / "data" / archive.split,
            expected_sha256=archive.sha256,
            size_bytes=archive.size_bytes,
            scene_id=scene_id,
        )

    shutil.rmtree(staging_root, ignore_errors=True)
    shutil.rmtree(data_root / ".cache", ignore_errors=True)


def ensure_one_scene(downloader, data_root, split, scene_id):
    _wait_if_training_blocked("download")
    d = scene_dir(data_root, split, scene_id)
    if d.is_dir():
        return

    rate_mbps = float(os.environ.get("KITSCENES_DOWNLOAD_RATE_MBPS", "0"))
    pause_seconds = float(os.environ.get("KITSCENES_DOWNLOAD_PAUSE_SECONDS", "0"))

    print(f"Downloading one shard: split={split} scene={scene_id}", flush=True)

    # Always use the logging downloader so we can measure Mbps.
    # rate_mbps <= 0 means unlimited, but still prints avg download speed.
    _download_archive_rate_limited(
        downloader=downloader,
        data_root=data_root,
        split=split,
        scene_id=scene_id,
        rate_mbps=rate_mbps,
    )

    if pause_seconds > 0:
        print(f"Pausing {pause_seconds:.1f}s after download", flush=True)
        time.sleep(pause_seconds)


def build_one_scene_dataset(data_root, split, scene_id, image_size):
    return KitScenesDataset(
        data_root=data_root,
        split=split,
        scene_ids=[scene_id],
        image_size=image_size,
        include_navigation=False,
    )


def make_model(args, device):
    model = AutoE2E(
        backbone=args.backbone,
        num_views=7,
        is_pretrained=False,
        embed_dim=args.embed_dim,
        image_feature_size=args.image_feature_size,
        view_fusion_kwargs={
            "bev_h": args.bev_h,
            "bev_w": args.bev_w,
            "num_points_in_pillar": args.pillar_points,
            "image_size": args.image_size,
        },
        map_fusion_mode="residual",
        planner_mode="bezier",
        enable_world_model=False,
        enable_reasoning=False,
    ).to(device)

    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)
    return model


def save_checkpoint(path, model, optimizer, epoch, scene_pos, global_step, micro_step, best_val_loss, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "scene_pos": scene_pos,
            "global_step": global_step,
            "micro_step": micro_step,
            "best_val_loss": best_val_loss,
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return {
        "epoch": int(ckpt.get("epoch", 1)),
        "scene_pos": int(ckpt.get("scene_pos", 0)),
        "global_step": int(ckpt.get("global_step", 0)),
        "micro_step": int(ckpt.get("micro_step", 0)),
        "best_val_loss": float(ckpt.get("best_val_loss", math.inf)),
    }


@torch.no_grad()
def validate_bounded(args, model, loss_fn, downloader, val_plan, device, amp_dtype, max_val_scenes):
    model.eval()
    losses = []
    data_root = Path(args.data_root)

    selected = val_plan if max_val_scenes is None else val_plan[:max_val_scenes]

    print(f"Validating on {len(selected)} bounded val shard(s)", flush=True)

    for i, (scene_id, split) in enumerate(selected, start=1):
        try:
            ensure_one_scene(downloader, data_root, split, scene_id)
            ds = build_one_scene_dataset(data_root, split, scene_id, args.image_size)
            loader = DataLoader(
                ds,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                collate_fn=collate_auto_e2e,
            )

            scene_losses = []
            for batch in loader:
                b = move_batch_to_device(batch, device)
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    pred = model(
                        b["visual_tiles"],
                        b["map_context"],
                        b["visual_history"],
                        b["egomotion_history"],
                        route_mask=b["route_mask"],
                        map_valid=b["map_valid"],
                        route_valid=b["route_valid"],
                        mode="eval",
                    )
                    loss = loss_fn(pred.float(), b["trajectory_target"])
                scene_losses.append(float(loss.detach().cpu()))

            if scene_losses:
                scene_loss = sum(scene_losses) / len(scene_losses)
                losses.append(scene_loss)
                print(f"val_scene={i}/{len(selected)} loss={scene_loss:.6f}", flush=True)

        except Exception as exc:
            print(f"WARNING: validation scene skipped split={split} scene={scene_id}: {exc}", flush=True)

        finally:
            cleanup_scene(data_root, split, scene_id)

    model.train()
    return sum(losses) / max(1, len(losses))


def make_plans(args):
    official_train = sdk_split_scene_ids("train")
    official_val = sdk_split_scene_ids("val")

    train_base = official_train[:args.train_shards]
    val_to_train = official_val[:args.val_shards_to_train]
    final_val = official_val[args.val_shards_to_train:]

    if args.final_val_shards is not None:
        final_val = final_val[:args.final_val_shards]

    train_plan = [(scene_id, "train") for scene_id in train_base]
    train_plan += [(scene_id, "val") for scene_id in val_to_train]
    val_plan = [(scene_id, "val") for scene_id in final_val]

    print("Bounded custom split:", flush=True)
    print(f"  official train available: {len(official_train)}", flush=True)
    print(f"  official val available: {len(official_val)}", flush=True)
    print(f"  train shards used from train: {len(train_base)}", flush=True)
    print(f"  val shards moved to train: {len(val_to_train)}", flush=True)
    print(f"  total train shards: {len(train_plan)}", flush=True)
    print(f"  final val shards: {len(val_plan)}", flush=True)

    return train_plan, val_plan


def train(args):
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_root = Path(args.data_root)
    ckpt_dir = Path(args.ckpt_dir)
    data_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    os.environ["KITSCENES_DOWNLOAD_RATE_MBPS"] = str(args.download_rate_mbps)
    os.environ["KITSCENES_DOWNLOAD_PAUSE_SECONDS"] = str(args.download_pause_seconds)
    print(f"Download rate limit: {args.download_rate_mbps if args.download_rate_mbps > 0 else 'unlimited'} Mbps", flush=True)
    print(f"Pause after each download: {args.download_pause_seconds}s", flush=True)

    os.environ["KITSCENES_DOWNLOAD_RATE_MBPS"] = str(args.download_rate_mbps)
    os.environ["KITSCENES_DOWNLOAD_PAUSE_SECONDS"] = str(args.download_pause_seconds)
    print(f"Download rate limit: {args.download_rate_mbps if args.download_rate_mbps > 0 else 'unlimited'} Mbps", flush=True)
    print(f"Pause after each download: {args.download_pause_seconds}s", flush=True)

    train_plan, val_plan = make_plans(args)

    if args.shuffle_scenes:
        random.shuffle(train_plan)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    if device.type != "cuda":
        raise RuntimeError(
            "CUDA is unavailable inside Docker. Stop here: do not train on CPU. "
            "Check nvidia-smi and torch.cuda.is_available() inside the container."
        )

    gpu_mem("start")

    downloader = PinnedKITScenesDownloader(data_root)
    model = make_model(args, device)
    loss_fn = TrajectoryImitationLoss().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print("amp dtype:", amp_dtype, flush=True)

    start_epoch = 1
    start_scene_pos = 0
    global_step = 0
    micro_step = 0
    best_val_loss = math.inf

    if args.resume:
        state = load_checkpoint(Path(args.resume), model, optimizer, device)
        start_epoch = state["epoch"]
        start_scene_pos = state["scene_pos"]
        global_step = state["global_step"]
        micro_step = state["micro_step"]
        best_val_loss = state["best_val_loss"]
        print(f"Resumed from {args.resume}: epoch={start_epoch} scene_pos={start_scene_pos} step={global_step}", flush=True)

    (ckpt_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2))

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        if epoch != start_epoch:
            start_scene_pos = 0
            if args.shuffle_scenes:
                random.shuffle(train_plan)

        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)

        for scene_pos in range(start_scene_pos, len(train_plan)):
            scene_id, split = train_plan[scene_pos]
            scene_number = scene_pos + 1

            try:
                ensure_one_scene(downloader, data_root, split, scene_id)
                ds = build_one_scene_dataset(data_root, split, scene_id, args.image_size)

                loader = DataLoader(
                    ds,
                    batch_size=1,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True,
                    collate_fn=collate_auto_e2e,
                )

                scene_losses = []

                for batch in loader:
                    b = move_batch_to_device(batch, device)

                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        pred = model(
                            b["visual_tiles"],
                            b["map_context"],
                            b["visual_history"],
                            b["egomotion_history"],
                            route_mask=b["route_mask"],
                            map_valid=b["map_valid"],
                            route_valid=b["route_valid"],
                            mode="train",
                        )
                        loss = loss_fn(pred.float(), b["trajectory_target"])
                        scaled_loss = loss / args.accum_steps

                    scaled_loss.backward()

                    micro_step += 1
                    scene_losses.append(float(loss.detach().cpu()))

                    if micro_step % args.accum_steps == 0:
                        if args.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1

                avg_scene_loss = sum(scene_losses) / max(1, len(scene_losses))
                print(
                    f"epoch={epoch} scene={scene_number}/{len(train_plan)} "
                    f"split={split} samples={len(ds)} step={global_step} "
                    f"loss={avg_scene_loss:.6f}",
                    flush=True,
                )

                if scene_number % args.log_mem_every_scenes == 0:
                    gpu_mem("train")

                if args.save_every_scenes > 0 and scene_number % args.save_every_scenes == 0:
                    save_checkpoint(
                        ckpt_dir / "last.pt",
                        model,
                        optimizer,
                        epoch,
                        scene_pos + 1,
                        global_step,
                        micro_step,
                        best_val_loss,
                        args,
                    )

                if (
                    args.validate_every_scenes > 0
                    and scene_number % args.validate_every_scenes == 0
                ):
                    val_loss = validate_bounded(
                        args,
                        model,
                        loss_fn,
                        downloader,
                        val_plan,
                        device,
                        amp_dtype,
                        args.val_scenes_per_eval,
                    )
                    print(f"bounded_val_loss={val_loss:.6f}", flush=True)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(
                            ckpt_dir / "best.pt",
                            model,
                            optimizer,
                            epoch,
                            scene_pos + 1,
                            global_step,
                            micro_step,
                            best_val_loss,
                            args,
                        )
                        print(f"new best val_loss={best_val_loss:.6f}", flush=True)

            except Exception as exc:
                print(f"WARNING: train scene skipped split={split} scene={scene_id}: {exc}", flush=True)

            finally:
                cleanup_scene(data_root, split, scene_id)

            if args.max_scenes is not None and scene_number >= args.max_scenes:
                print("Stopping early because --max-scenes was reached.", flush=True)
                break

        # Flush leftover accumulated gradients at epoch end.
        if micro_step % args.accum_steps != 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        val_loss = validate_bounded(
            args,
            model,
            loss_fn,
            downloader,
            val_plan,
            device,
            amp_dtype,
            args.val_scenes_per_eval,
        )

        print(
            f"epoch={epoch} end_val_loss={val_loss:.6f} "
            f"time_min={(time.time() - epoch_start) / 60:.1f}",
            flush=True,
        )

        save_checkpoint(
            ckpt_dir / "last.pt",
            model,
            optimizer,
            epoch + 1,
            0,
            global_step,
            micro_step,
            best_val_loss,
            args,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                ckpt_dir / "best.pt",
                model,
                optimizer,
                epoch + 1,
                0,
                global_step,
                micro_step,
                best_val_loss,
                args,
            )
            print(f"new best val_loss={best_val_loss:.6f}", flush=True)

    print("Bounded shard training complete.", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", default="/auto_e2e/datasets/kitscenes_bounded_cache")
    parser.add_argument("--ckpt-dir", default="/auto_e2e/checkpoints/kitscenes_bounded")

    # Your requested split: 533 train + first 52 val = 585 train, remaining 65 val.
    parser.add_argument("--train-shards", type=int, default=533)
    parser.add_argument("--val-shards-to-train", type=int, default=52)
    parser.add_argument("--final-val-shards", type=int, default=65)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--accum-steps", type=int, default=2)
    parser.add_argument("--max-scenes", type=int, default=None)

    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--image-feature-size", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--bev-h", type=int, default=450)
    parser.add_argument("--bev-w", type=int, default=300)
    parser.add_argument("--pillar-points", type=int, default=4)
    parser.add_argument("--backbone", default="conv_next_v2_tiny")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--save-every-scenes", type=int, default=10)
    parser.add_argument("--validate-every-scenes", type=int, default=50)
    parser.add_argument("--val-scenes-per-eval", type=int, default=5)
    parser.add_argument("--log-mem-every-scenes", type=int, default=10)

    parser.add_argument("--shuffle-scenes", action="store_true")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--download-rate-mbps", type=float, default=0.0)
    parser.add_argument("--download-pause-seconds", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
