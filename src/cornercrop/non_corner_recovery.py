"""Second-pass recovery for images archived with non-corner watermark signals."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from PIL import Image, ImageFile, UnidentifiedImageError

from .batch import AdaptiveParallelismConfig, process_batch
from .cli import SUPPORTED_IMAGE_EXTENSIONS
from .cropper import (
    BOTTOM_BRANDING_ZONE_FRACTION,
    EDGE_BRANDING_ZONE_FRACTION,
    TOP_BRANDING_ZONE_FRACTION,
    BrandingCandidate,
    CropProfile,
    CropResult,
    compute_crop,
    find_branding_candidates,
    should_use_cover_profile,
)
from .library_runner import DEFAULT_ARCHIVE_DIR_NAME, removed_area_ratio
from .pipeline import (
    VerificationStatus,
    _build_processed_image,
    _collect_text_regions,
    _save_image_atomic,
    _verify_processed_image,
)

DEFAULT_ROOT = "."
DEFAULT_REASON_DIR_NAME = "non_corner_watermark"
DEFAULT_STATE_DIR = os.path.join("runs", "non-corner-recovery", "live")
STATE_VERSION = 1
AREA_RATIO_EPSILON = 1e-6

# NAS-hosted photo sets can contain mildly truncated JPEGs that Pillow can still
# decode well enough for OCR and crop recovery.
ImageFile.LOAD_TRUNCATED_IMAGES = True


class RecoveryAction(str, Enum):
    """Final disposition for one archived image."""

    RECOVER = "recover"
    KEEP = "keep"


@dataclass(frozen=True)
class RecoveryConfig:
    """Runtime config for the non-corner archive recovery pass."""

    root: str = DEFAULT_ROOT
    state_dir: str = DEFAULT_STATE_DIR
    archive_dir_name: str = DEFAULT_ARCHIVE_DIR_NAME
    reason_dir_name: str = DEFAULT_REASON_DIR_NAME
    dry_run: bool = False
    force: bool = False
    max_workers: int = 3
    resource_profile: str = "balanced"
    resource_poll_interval: float = 5.0
    heartbeat_interval: float = 30.0
    progress_every: int = 25
    min_confidence: float = 0.25
    margin: int = 8
    max_crop_frac: float = 0.30
    max_removed_area_ratio: float = 0.30
    verify: bool = True
    stop_file: str | None = None
    limit: int | None = None


@dataclass(frozen=True)
class RecoveryDecision:
    """Policy decision for a second-pass inspected image."""

    action: RecoveryAction
    reason: str
    original_size: tuple[int, int] | None = None
    output_size: tuple[int, int] | None = None
    removed_area_ratio: float = 0.0
    selected_profile: CropProfile | None = None
    crop_result: CropResult | None = None
    verification_status: str = VerificationStatus.NOT_RUN
    residual_count: int = 0
    branding_candidates: tuple[BrandingCandidate, ...] = ()


@dataclass(frozen=True)
class RecoveryResult:
    """Recorded result for one source archive path."""

    source_path: str
    album_path: str
    action: str
    reason: str
    output_path: str | None
    original_size: tuple[int, int] | None
    output_size: tuple[int, int] | None
    removed_area_ratio: float
    selected_profile: str | None
    verification_status: str
    residual_count: int
    error: str | None = None


class RecoveryDatabase:
    """Small SQLite state store used for resume and monitoring."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._connection = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._init_schema()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _init_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS images (
                source_path TEXT PRIMARY KEY,
                album_path TEXT NOT NULL,
                action TEXT NOT NULL,
                reason TEXT NOT NULL,
                output_path TEXT,
                original_width INTEGER,
                original_height INTEGER,
                output_width INTEGER,
                output_height INTEGER,
                removed_area_ratio REAL NOT NULL DEFAULT 0.0,
                selected_profile TEXT,
                verification_status TEXT NOT NULL,
                residual_count INTEGER NOT NULL DEFAULT 0,
                dry_run INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_recovery_album ON images(album_path);
            CREATE INDEX IF NOT EXISTS idx_recovery_action ON images(action);
            """
        )
        self._connection.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            ("state_version", str(STATE_VERSION)),
        )
        self._connection.commit()

    def has_completed_image(self, source_path: str, dry_run: bool) -> bool:
        with self._lock:
            row = self._connection.execute(
                "SELECT action, dry_run, error FROM images WHERE source_path = ?",
                (source_path,),
            ).fetchone()
        if row is None:
            return False
        if bool(row["dry_run"]) != dry_run:
            return False
        return row["error"] is None and row["action"] in {
            "recovered",
            "kept",
            "would_recover",
            "would_keep",
        }

    def record(self, result: RecoveryResult, dry_run: bool) -> None:
        now = time.time()
        original_width, original_height = result.original_size or (None, None)
        output_width, output_height = result.output_size or (None, None)
        with self._lock:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO images(
                    source_path, album_path, action, reason, output_path,
                    original_width, original_height, output_width, output_height,
                    removed_area_ratio, selected_profile, verification_status,
                    residual_count, dry_run, error, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.source_path,
                    result.album_path,
                    result.action,
                    result.reason,
                    result.output_path,
                    original_width,
                    original_height,
                    output_width,
                    output_height,
                    result.removed_area_ratio,
                    result.selected_profile,
                    result.verification_status,
                    result.residual_count,
                    1 if dry_run else 0,
                    result.error,
                    now,
                ),
            )
            self._connection.commit()

    def summary(self) -> dict:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT
                    COUNT(*) AS processed,
                    SUM(CASE WHEN action IN ('recovered', 'would_recover') THEN 1 ELSE 0 END) AS recovered,
                    SUM(CASE WHEN action IN ('kept', 'would_keep') THEN 1 ELSE 0 END) AS kept,
                    SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS failed
                FROM images
                """
            ).fetchone()
        return {
            "images_processed": int(row["processed"] or 0),
            "images_recovered": int(row["recovered"] or 0),
            "images_kept": int(row["kept"] or 0),
            "images_failed": int(row["failed"] or 0),
        }


def iter_non_corner_archive_images(
    root: str,
    archive_dir_name: str = DEFAULT_ARCHIVE_DIR_NAME,
    reason_dir_name: str = DEFAULT_REASON_DIR_NAME,
) -> Iterable[str]:
    """Yield image files inside ``_cornercrop_archive/non_corner_watermark`` directories."""
    root = os.path.abspath(root)
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            with os.scandir(directory) as entries:
                dirs = []
                files = []
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            dirs.append(entry)
                        elif entry.is_file(follow_symlinks=False) and _is_supported_image(entry.path):
                            files.append(entry.path)
                    except OSError:
                        continue
        except OSError:
            continue

        if _is_target_reason_dir(directory, archive_dir_name, reason_dir_name):
            for path in sorted(files):
                yield os.path.abspath(path)
            continue

        for entry in sorted(dirs, key=lambda item: item.name, reverse=True):
            if entry.name in {".cornercrop", "__MACOSX"} or entry.name.startswith(".cornercrop-"):
                continue
            stack.append(entry.path)


def run_recovery(config: RecoveryConfig) -> dict:
    """Run the resumable second-pass recovery."""
    os.makedirs(config.state_dir, exist_ok=True)
    db = RecoveryDatabase(os.path.join(config.state_dir, "non-corner-recovery.sqlite3"))
    _write_run_config(config)
    started_at = time.time()

    try:
        source_paths = list(
            iter_non_corner_archive_images(
                config.root,
                archive_dir_name=config.archive_dir_name,
                reason_dir_name=config.reason_dir_name,
            )
        )
        if config.limit is not None:
            source_paths = source_paths[: config.limit]

        _write_progress(
            config.state_dir,
            "discovered",
            db.summary(),
            {
                "total_sources": len(source_paths),
                "dry_run": config.dry_run,
                **_progress_metrics(0, len(source_paths), started_at),
            },
        )

        if source_paths:
            adaptive = _adaptive_config(config)
            process_batch(
                source_paths,
                lambda source_path: _process_archive_image(source_path, config, db),
                adaptive,
                progress_callback=lambda completed, total, target, snapshot: _write_progress(
                    config.state_dir,
                    "running",
                    db.summary(),
                    {
                        "completed_sources": completed,
                        "total_sources": total,
                        "target_workers": target,
                        "cpu_percent": snapshot.cpu_percent,
                        "memory_percent": snapshot.memory_percent,
                        "read_mbps": snapshot.read_mbps,
                        "write_mbps": snapshot.write_mbps,
                        "dry_run": config.dry_run,
                        **_progress_metrics(completed, total, started_at),
                    },
                ),
            )

        final_summary = db.summary()
        _write_progress(
            config.state_dir,
            "finished" if not _should_stop(config) else "stopped",
            final_summary,
            {
                "total_sources": len(source_paths),
                "dry_run": config.dry_run,
                **_progress_metrics(final_summary["images_processed"], len(source_paths), started_at),
            },
        )
        return final_summary
    finally:
        db.close()


def inspect_recovery_candidate(source_path: str, config: RecoveryConfig) -> RecoveryDecision:
    """Inspect one archive image and decide whether a clean crop is possible."""
    with Image.open(source_path) as image:
        width, height = image.size
        text_regions = _collect_text_regions(
            source_path,
            image,
            min_confidence=config.min_confidence,
        )
        candidates = find_branding_candidates(
            text_regions,
            width,
            height,
            top_frac=TOP_BRANDING_ZONE_FRACTION,
            bottom_frac=BOTTOM_BRANDING_ZONE_FRACTION,
            edge_frac=EDGE_BRANDING_ZONE_FRACTION,
        )

        if not candidates:
            return RecoveryDecision(
                action=RecoveryAction.KEEP,
                reason="no_watermark_detected_second_pass",
                original_size=(width, height),
                output_size=(width, height),
                branding_candidates=(),
            )

        profile_order = _profile_order(candidates, width)
        attempts = _candidate_crop_results(
            candidates,
            width,
            height,
            profile_order,
            config,
        )
        no_crop_seen = False
        viable_attempts: list[tuple[float, CropProfile, CropResult]] = []
        excessive_attempts: list[tuple[float, CropProfile, CropResult]] = []
        best_residual: tuple[float, CropProfile, CropResult, int] | None = None
        profile_rank = {profile: index for index, profile in enumerate(profile_order)}

        for profile, crop_result, ratio in attempts:
            if not crop_result.needs_crop:
                no_crop_seen = True
                continue
            if _exceeds_removed_area_limit(ratio, config.max_removed_area_ratio):
                excessive_attempts.append((ratio, profile, crop_result))
                continue
            viable_attempts.append((ratio, profile, crop_result))

        viable_attempts.sort(key=lambda attempt: (attempt[0], profile_rank[attempt[1]]))
        for ratio, profile, crop_result in viable_attempts:
            verification_status = VerificationStatus.NOT_RUN
            residual_count = 0
            if config.verify:
                final_image = _build_processed_image(image, crop_result)
                try:
                    residual_matches = []
                    verification_status, residual_matches = _verify_processed_image(
                        final_image,
                        min_confidence=min(config.min_confidence, 0.1),
                    )
                    residual_count = len(residual_matches)
                finally:
                    final_image.close()

                if verification_status != VerificationStatus.CLEAN:
                    if best_residual is None or ratio < best_residual[0]:
                        best_residual = (ratio, profile, crop_result, residual_count)
                    continue

            return RecoveryDecision(
                action=RecoveryAction.RECOVER,
                reason="second_pass_watermark_crop",
                original_size=crop_result.original_size,
                output_size=crop_result.output_size,
                removed_area_ratio=ratio,
                selected_profile=profile,
                crop_result=crop_result,
                verification_status=verification_status,
                residual_count=residual_count,
                branding_candidates=tuple(candidates),
            )

        if best_residual is not None:
            ratio, profile, crop_result, residual_count = best_residual
            return RecoveryDecision(
                action=RecoveryAction.KEEP,
                reason="residual_watermark_after_second_pass_crop",
                original_size=crop_result.original_size,
                output_size=crop_result.output_size,
                removed_area_ratio=ratio,
                selected_profile=profile,
                crop_result=crop_result,
                verification_status=VerificationStatus.RESIDUAL,
                residual_count=residual_count,
                branding_candidates=tuple(candidates),
            )

        if excessive_attempts:
            ratio, profile, crop_result = min(
                excessive_attempts,
                key=lambda attempt: (attempt[0], profile_rank[attempt[1]]),
            )
            return RecoveryDecision(
                action=RecoveryAction.KEEP,
                reason="crop_would_remove_too_much_image_area",
                original_size=crop_result.original_size,
                output_size=crop_result.output_size,
                removed_area_ratio=ratio,
                selected_profile=profile,
                crop_result=crop_result,
                verification_status=VerificationStatus.NOT_RUN,
                residual_count=0,
                branding_candidates=tuple(candidates),
            )

        return RecoveryDecision(
            action=RecoveryAction.KEEP,
            reason="no_second_pass_crop_available" if no_crop_seen else "no_acceptable_second_pass_crop",
            original_size=(width, height),
            output_size=(width, height),
            branding_candidates=tuple(candidates),
        )


def recover_image(source_path: str, config: RecoveryConfig) -> RecoveryResult:
    """Recover one archived image if second-pass crop verification succeeds."""
    album_path = album_dir_for_archive_image(
        source_path,
        archive_dir_name=config.archive_dir_name,
        reason_dir_name=config.reason_dir_name,
    )
    decision = inspect_recovery_candidate(source_path, config)
    if decision.action == RecoveryAction.KEEP:
        return RecoveryResult(
            source_path=source_path,
            album_path=album_path,
            action="would_keep" if config.dry_run else "kept",
            reason=decision.reason,
            output_path=None,
            original_size=decision.original_size,
            output_size=decision.output_size,
            removed_area_ratio=decision.removed_area_ratio,
            selected_profile=decision.selected_profile.value if decision.selected_profile else None,
            verification_status=decision.verification_status,
            residual_count=decision.residual_count,
        )

    if decision.crop_result is None:
        raise RuntimeError("recovery decision is missing crop result")

    target_path = safe_recovered_path(source_path, album_path)
    if not config.dry_run:
        _save_recovered_image(source_path, target_path, decision.crop_result)
        os.unlink(source_path)

    return RecoveryResult(
        source_path=source_path,
        album_path=album_path,
        action="would_recover" if config.dry_run else "recovered",
        reason=decision.reason,
        output_path=target_path,
        original_size=decision.original_size,
        output_size=decision.output_size,
        removed_area_ratio=decision.removed_area_ratio,
        selected_profile=decision.selected_profile.value if decision.selected_profile else None,
        verification_status=decision.verification_status,
        residual_count=decision.residual_count,
    )


def album_dir_for_archive_image(
    source_path: str,
    archive_dir_name: str = DEFAULT_ARCHIVE_DIR_NAME,
    reason_dir_name: str = DEFAULT_REASON_DIR_NAME,
) -> str:
    """Return the parent album directory for an archived non-corner image."""
    reason_dir = os.path.dirname(os.path.abspath(source_path))
    archive_dir = os.path.dirname(reason_dir)
    album_dir = os.path.dirname(archive_dir)
    if os.path.basename(reason_dir) != reason_dir_name or os.path.basename(archive_dir) != archive_dir_name:
        raise ValueError(f"not a {archive_dir_name}/{reason_dir_name} image: {source_path}")
    return album_dir


def safe_recovered_path(source_path: str, album_path: str) -> str:
    """Return a collision-safe target path in the album root."""
    stem, ext = os.path.splitext(os.path.basename(source_path))
    candidate = os.path.join(album_path, stem + ext)
    if not os.path.exists(candidate):
        return candidate

    index = 1
    while True:
        candidate = os.path.join(album_path, f"{stem}__cornercrop_recovered_{index}{ext}")
        if not os.path.exists(candidate):
            return candidate
        index += 1


def _process_archive_image(source_path: str, config: RecoveryConfig, db: RecoveryDatabase) -> str:
    if _should_stop(config):
        return "stopped"
    if not config.force and db.has_completed_image(source_path, dry_run=config.dry_run):
        return "skipped_existing_record"

    try:
        result = recover_image(source_path, config)
    except UnidentifiedImageError as exc:
        result = RecoveryResult(
            source_path=source_path,
            album_path=_safe_album_dir_for_record(source_path, config),
            action="kept",
            reason="unreadable_image_kept_in_archive",
            output_path=None,
            original_size=None,
            output_size=None,
            removed_area_ratio=0.0,
            selected_profile=None,
            verification_status=VerificationStatus.NOT_RUN,
            residual_count=0,
            error=None,
        )
    except Exception as exc:
        result = RecoveryResult(
            source_path=source_path,
            album_path=_safe_album_dir_for_record(source_path, config),
            action="failed",
            reason="processing_error",
            output_path=None,
            original_size=None,
            output_size=None,
            removed_area_ratio=0.0,
            selected_profile=None,
            verification_status=VerificationStatus.NOT_RUN,
            residual_count=0,
            error=str(exc),
        )

    db.record(result, dry_run=config.dry_run)
    return result.action


def _safe_album_dir_for_record(source_path: str, config: RecoveryConfig) -> str:
    try:
        return album_dir_for_archive_image(source_path, config.archive_dir_name, config.reason_dir_name)
    except ValueError:
        return os.path.dirname(os.path.abspath(source_path))


def _save_recovered_image(source_path: str, target_path: str, crop_result: CropResult) -> None:
    with Image.open(source_path) as image:
        final_image = _build_processed_image(image, crop_result)
        try:
            ext = os.path.splitext(target_path)[1].lower()
            save_kwargs = {}
            if ext in (".jpg", ".jpeg"):
                save_kwargs["quality"] = 95
            _save_image_atomic(final_image, target_path, **save_kwargs)
        finally:
            final_image.close()


def _profile_order(candidates: list[BrandingCandidate], width: int) -> list[CropProfile]:
    if should_use_cover_profile(candidates, width):
        return [CropProfile.COVER, CropProfile.STRIP]
    return [CropProfile.STRIP, CropProfile.COVER]


def _candidate_crop_results(
    candidates: list[BrandingCandidate],
    width: int,
    height: int,
    profiles: list[CropProfile],
    config: RecoveryConfig,
) -> list[tuple[CropProfile, CropResult, float]]:
    attempts = []
    seen_boxes = set()
    for profile in profiles:
        crop_result = compute_crop(
            candidates,
            width,
            height,
            strategy=profile,
            margin=config.margin,
            max_crop_frac=config.max_crop_frac,
        )
        key = crop_result.crop_box
        if key in seen_boxes:
            continue
        seen_boxes.add(key)
        attempts.append(
            (
                profile,
                crop_result,
                removed_area_ratio(crop_result.original_size, crop_result.output_size),
            )
        )
    return attempts


def _exceeds_removed_area_limit(ratio: float, limit: float) -> bool:
    return ratio - limit > AREA_RATIO_EPSILON


def _is_target_reason_dir(directory: str, archive_dir_name: str, reason_dir_name: str) -> bool:
    return (
        os.path.basename(directory) == reason_dir_name
        and os.path.basename(os.path.dirname(directory)) == archive_dir_name
    )


def _is_supported_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS


def _adaptive_config(config: RecoveryConfig) -> AdaptiveParallelismConfig:
    profile_defaults = {
        "conservative": {
            "cpu_low_water": 45.0,
            "cpu_high_water": 78.0,
            "memory_low_water": 70.0,
            "memory_high_water": 84.0,
            "read_low_mbps": 60.0,
            "read_high_mbps": 180.0,
            "write_low_mbps": 25.0,
            "write_high_mbps": 80.0,
        },
        "balanced": {
            "cpu_low_water": 55.0,
            "cpu_high_water": 86.0,
            "memory_low_water": 84.0,
            "memory_high_water": 92.0,
            "read_low_mbps": 120.0,
            "read_high_mbps": 320.0,
            "write_low_mbps": 45.0,
            "write_high_mbps": 140.0,
        },
        "aggressive": {
            "cpu_low_water": 65.0,
            "cpu_high_water": 92.0,
            "memory_low_water": 78.0,
            "memory_high_water": 90.0,
            "read_low_mbps": 200.0,
            "read_high_mbps": 520.0,
            "write_low_mbps": 70.0,
            "write_high_mbps": 220.0,
        },
    }[config.resource_profile]
    return AdaptiveParallelismConfig(
        enabled=True,
        min_workers=1,
        max_workers=config.max_workers,
        poll_interval=config.resource_poll_interval,
        progress_interval=max(1, config.progress_every),
        heartbeat_interval=config.heartbeat_interval,
        **profile_defaults,
    )


def _should_stop(config: RecoveryConfig) -> bool:
    return bool(config.stop_file and os.path.exists(config.stop_file))


def _write_run_config(config: RecoveryConfig) -> None:
    _atomic_json_dump(os.path.join(config.state_dir, "run-config.json"), dict(config.__dict__))


def _write_progress(state_dir: str, phase: str, summary: dict, extra: dict | None) -> None:
    payload = {
        "phase": phase,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "summary": summary,
    }
    if extra:
        payload.update(extra)
    _atomic_json_dump(os.path.join(state_dir, "progress.json"), payload)


def _progress_metrics(completed: int, total: int, started_at: float) -> dict:
    now = time.time()
    elapsed = max(0.0, now - started_at)
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining = max(0, total - completed)
    return {
        "elapsed_seconds": round(elapsed, 1),
        "completion_ratio": round(completed / total, 6) if total else 1.0,
        "items_per_minute": round(rate * 60.0, 2) if rate > 0 else 0.0,
        "estimated_remaining_seconds": round(remaining / rate, 1) if rate > 0 else None,
    }


def _atomic_json_dump(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=os.path.dirname(path) or ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        finally:
            raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cornercrop-recover-non-corner",
        description="Second-pass recover images from non-corner watermark archives.",
    )
    parser.add_argument("root", nargs="?", default=DEFAULT_ROOT, help="Root image library (default: current directory)")
    parser.add_argument("--state-dir", default=DEFAULT_STATE_DIR, help=f"Local state directory (default: {DEFAULT_STATE_DIR})")
    parser.add_argument("--archive-dir-name", default=DEFAULT_ARCHIVE_DIR_NAME)
    parser.add_argument("--reason-dir-name", default=DEFAULT_REASON_DIR_NAME)
    parser.add_argument("--dry-run", action="store_true", help="Record decisions without modifying images")
    parser.add_argument("--force", action="store_true", help="Reprocess images already marked complete in this state dir")
    parser.add_argument("--max-workers", type=_positive_int, default=3)
    parser.add_argument("--resource-profile", choices=["conservative", "balanced", "aggressive"], default="balanced")
    parser.add_argument("--resource-poll-interval", type=_positive_float, default=5.0)
    parser.add_argument("--heartbeat-interval", type=_positive_float, default=30.0)
    parser.add_argument("--progress-every", type=_positive_int, default=25)
    parser.add_argument("--min-confidence", type=_bounded_float(0.0, 1.0, "minimum confidence"), default=0.25)
    parser.add_argument("--margin", type=_non_negative_int, default=8)
    parser.add_argument("--max-crop", type=_bounded_float(0.0, 1.0, "max crop fraction"), default=0.30)
    parser.add_argument("--max-removed-area", type=_bounded_float(0.0, 1.0, "max removed area"), default=0.30)
    parser.add_argument("--no-verify", action="store_true", help="Accept crop geometry without OCR verification")
    parser.add_argument("--stop-file", help="If this file appears, stop after currently running images")
    parser.add_argument("--limit", type=_positive_int, help="Process only the first N discovered archive images")
    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = RecoveryConfig(
        root=os.path.abspath(args.root),
        state_dir=os.path.abspath(args.state_dir),
        archive_dir_name=args.archive_dir_name,
        reason_dir_name=args.reason_dir_name,
        dry_run=args.dry_run,
        force=args.force,
        max_workers=args.max_workers,
        resource_profile=args.resource_profile,
        resource_poll_interval=args.resource_poll_interval,
        heartbeat_interval=args.heartbeat_interval,
        progress_every=args.progress_every,
        min_confidence=args.min_confidence,
        margin=args.margin,
        max_crop_frac=args.max_crop,
        max_removed_area_ratio=args.max_removed_area,
        verify=not args.no_verify,
        stop_file=os.path.abspath(args.stop_file) if args.stop_file else None,
        limit=args.limit,
    )
    if not os.path.isdir(config.root):
        parser.error(f"Root directory does not exist: {config.root}")

    summary = run_recovery(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _positive_int(raw_value: str) -> int:
    value = int(raw_value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def _non_negative_int(raw_value: str) -> int:
    value = int(raw_value)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return value


def _positive_float(raw_value: str) -> float:
    value = float(raw_value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive number")
    return value


def _bounded_float(lower: float, upper: float, label: str):
    def _validator(raw_value: str) -> float:
        value = float(raw_value)
        if not lower <= value <= upper:
            raise argparse.ArgumentTypeError(f"{label} must be between {lower} and {upper}")
        return value

    return _validator


if __name__ == "__main__":
    raise SystemExit(main())
