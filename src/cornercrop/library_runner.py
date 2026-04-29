"""Resumable album-level runner for large NAS-backed image libraries."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from PIL import Image, ImageFile, UnidentifiedImageError

from .batch import AdaptiveParallelismConfig, ResourceMonitor, process_batch
from .cli import SUPPORTED_IMAGE_EXTENSIONS
from .cropper import (
    BOTTOM_BRANDING_ZONE_FRACTION,
    EDGE_BRANDING_ZONE_FRACTION,
    TOP_BRANDING_ZONE_FRACTION,
    CropProfile,
    compute_crop,
    find_branding_candidates,
    find_corner_watermarks,
)
from .pipeline import (
    ProcessResult,
    VerificationStatus,
    _build_processed_image,
    _collect_text_regions,
    _save_image_atomic,
)

DEFAULT_ARCHIVE_DIR_NAME = "_cornercrop_archive"
DEFAULT_STATE_DIR = os.path.join("runs", "library-watermark", "live")
STATE_VERSION = 1

# NAS-hosted photo sets often contain a few JPEGs with tiny trailing truncations.
# Pillow can still decode these safely enough for watermark inspection/cropping.
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageAction(str, Enum):
    """Final disposition for one image."""

    CROP = "crop"
    SKIP = "skip"
    ARCHIVE = "archive"


class ArchiveReason(str, Enum):
    """Reasons that require moving an image out of the active album."""

    NON_CORNER_WATERMARK = "non_corner_watermark"
    EXCESSIVE_CROP_AREA = "excessive_crop_area"
    UNREADABLE_IMAGE = "unreadable_image"


@dataclass(frozen=True)
class ImageDecision:
    """Policy decision for a processed image inspection."""

    action: ImageAction
    removed_area_ratio: float = 0.0
    archive_reason: ArchiveReason | None = None
    reason: str = ""


@dataclass(frozen=True)
class LibraryRunConfig:
    """Runtime config for a large image-library pass."""

    root: str
    state_dir: str
    archive_dir_name: str = DEFAULT_ARCHIVE_DIR_NAME
    dry_run: bool = False
    force: bool = False
    max_album_workers: int = 2
    resource_profile: str = "conservative"
    resource_poll_interval: float = 5.0
    progress_every: int = 10
    min_confidence: float = 0.25
    corner_frac: float = 0.22
    margin: int = 8
    max_crop_frac: float = 0.30
    max_removed_area_ratio: float = 0.30
    stop_file: str | None = None


class JobDatabase:
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
            CREATE TABLE IF NOT EXISTS albums (
                path TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                total INTEGER NOT NULL DEFAULT 0,
                processed INTEGER NOT NULL DEFAULT 0,
                cropped INTEGER NOT NULL DEFAULT 0,
                skipped INTEGER NOT NULL DEFAULT 0,
                archived INTEGER NOT NULL DEFAULT 0,
                failed INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                started_at REAL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS images (
                path TEXT PRIMARY KEY,
                album_path TEXT NOT NULL,
                action TEXT NOT NULL,
                reason TEXT,
                output_path TEXT,
                original_width INTEGER,
                original_height INTEGER,
                output_width INTEGER,
                output_height INTEGER,
                removed_area_ratio REAL NOT NULL DEFAULT 0.0,
                dry_run INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_images_album ON images(album_path);
            """
        )
        self._connection.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            ("state_version", str(STATE_VERSION)),
        )
        self._connection.commit()

    def mark_album_started(self, album_path: str, total: int) -> None:
        now = time.time()
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO albums(path, status, total, started_at, updated_at)
                VALUES (?, 'running', ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    status='running',
                    total=excluded.total,
                    error=NULL,
                    started_at=COALESCE(albums.started_at, excluded.started_at),
                    updated_at=excluded.updated_at
                """,
                (album_path, total, now, now),
            )
            self._connection.commit()

    def mark_album_done(self, album_path: str, counts: dict[str, int], error: str | None = None) -> None:
        status = "failed" if error else "done"
        now = time.time()
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO albums(
                    path, status, total, processed, cropped, skipped, archived, failed, error, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    status=excluded.status,
                    total=excluded.total,
                    processed=excluded.processed,
                    cropped=excluded.cropped,
                    skipped=excluded.skipped,
                    archived=excluded.archived,
                    failed=excluded.failed,
                    error=excluded.error,
                    updated_at=excluded.updated_at
                """,
                (
                    album_path,
                    status,
                    counts.get("total", 0),
                    counts.get("processed", 0),
                    counts.get("cropped", 0),
                    counts.get("skipped", 0),
                    counts.get("archived", 0),
                    counts.get("failed", 0),
                    error,
                    now,
                ),
            )
            self._connection.commit()

    def mark_album_stopped(self, album_path: str, counts: dict[str, int]) -> None:
        now = time.time()
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO albums(
                    path, status, total, processed, cropped, skipped, archived, failed, error, updated_at
                )
                VALUES (?, 'stopped', ?, ?, ?, ?, ?, ?, NULL, ?)
                ON CONFLICT(path) DO UPDATE SET
                    status='stopped',
                    total=excluded.total,
                    processed=excluded.processed,
                    cropped=excluded.cropped,
                    skipped=excluded.skipped,
                    archived=excluded.archived,
                    failed=excluded.failed,
                    error=NULL,
                    updated_at=excluded.updated_at
                """,
                (
                    album_path,
                    counts.get("total", 0),
                    counts.get("processed", 0),
                    counts.get("cropped", 0),
                    counts.get("skipped", 0),
                    counts.get("archived", 0),
                    counts.get("failed", 0),
                    now,
                ),
            )
            self._connection.commit()

    def update_album_counts(self, album_path: str, counts: dict[str, int]) -> None:
        now = time.time()
        with self._lock:
            self._connection.execute(
                """
                UPDATE albums
                SET
                    total=?,
                    processed=?,
                    cropped=?,
                    skipped=?,
                    archived=?,
                    failed=?,
                    updated_at=?
                WHERE path=?
                """,
                (
                    counts.get("total", 0),
                    counts.get("processed", 0),
                    counts.get("cropped", 0),
                    counts.get("skipped", 0),
                    counts.get("archived", 0),
                    counts.get("failed", 0),
                    now,
                    album_path,
                ),
            )
            self._connection.commit()

    def record_image(
        self,
        *,
        path: str,
        album_path: str,
        action: str,
        reason: str | None,
        output_path: str | None,
        original_size: tuple[int, int] | None,
        output_size: tuple[int, int] | None,
        removed_area_ratio: float,
        dry_run: bool,
        error: str | None = None,
    ) -> None:
        now = time.time()
        original_width, original_height = original_size or (None, None)
        output_width, output_height = output_size or (None, None)
        with self._lock:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO images(
                    path, album_path, action, reason, output_path,
                    original_width, original_height, output_width, output_height,
                    removed_area_ratio, dry_run, error, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    path,
                    album_path,
                    action,
                    reason,
                    output_path,
                    original_width,
                    original_height,
                    output_width,
                    output_height,
                    removed_area_ratio,
                    1 if dry_run else 0,
                    error,
                    now,
                ),
            )
            self._connection.commit()

    def has_completed_image(self, path: str, dry_run: bool) -> bool:
        with self._lock:
            row = self._connection.execute(
                "SELECT action, dry_run, error FROM images WHERE path = ?",
                (path,),
            ).fetchone()
        if row is None:
            return False
        if bool(row["dry_run"]) != dry_run:
            return False
        return row["error"] is None and row["action"] in {
            "cropped",
            "skipped",
            "archived",
            "would_crop",
            "would_skip",
            "would_archive",
        }

    def summary(self) -> dict:
        with self._lock:
            image_rows = self._connection.execute(
                """
                SELECT
                    COUNT(*) AS processed,
                    SUM(CASE WHEN action IN ('cropped', 'would_crop') THEN 1 ELSE 0 END) AS cropped,
                    SUM(CASE WHEN action IN ('skipped', 'would_skip') THEN 1 ELSE 0 END) AS skipped,
                    SUM(CASE WHEN action IN ('archived', 'would_archive') THEN 1 ELSE 0 END) AS archived,
                    SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS failed
                FROM images
                """
            ).fetchone()
            album_rows = self._connection.execute(
                """
                SELECT
                    COUNT(*) AS albums_seen,
                    SUM(CASE WHEN status = 'done' THEN 1 ELSE 0 END) AS albums_done,
                    SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) AS albums_running,
                    SUM(CASE WHEN status = 'stopped' THEN 1 ELSE 0 END) AS albums_stopped,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS albums_failed
                FROM albums
                """
            ).fetchone()
        return {
            "images_processed": int(image_rows["processed"] or 0),
            "images_cropped": int(image_rows["cropped"] or 0),
            "images_skipped": int(image_rows["skipped"] or 0),
            "images_archived": int(image_rows["archived"] or 0),
            "images_failed": int(image_rows["failed"] or 0),
            "albums_seen": int(album_rows["albums_seen"] or 0),
            "albums_done": int(album_rows["albums_done"] or 0),
            "albums_running": int(album_rows["albums_running"] or 0),
            "albums_stopped": int(album_rows["albums_stopped"] or 0),
            "albums_failed": int(album_rows["albums_failed"] or 0),
        }


def iter_album_dirs(
    root: str,
    archive_dir_name: str = DEFAULT_ARCHIVE_DIR_NAME,
    include_root: bool = True,
) -> Iterable[str]:
    """Yield directories that directly contain supported image files."""
    root = os.path.abspath(root)
    ignored_names = {archive_dir_name, ".cornercrop", "__MACOSX"}

    def _walk(directory: str):
        try:
            with os.scandir(directory) as entries:
                dirs = []
                has_image = False
                for entry in entries:
                    if entry.name in ignored_names or entry.name.startswith(".cornercrop-"):
                        continue
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            dirs.append(entry.path)
                        elif entry.is_file(follow_symlinks=False) and _is_supported_image(entry.path):
                            has_image = True
                    except OSError:
                        continue
        except OSError:
            return

        if has_image and (include_root or directory != root):
            yield directory
        for child in sorted(dirs):
            yield from _walk(child)

    yield from _walk(root)


def build_image_decision(result: ProcessResult, max_removed_area_ratio: float) -> ImageDecision:
    """Decide whether a processed result should be cropped, skipped, or archived."""
    if not result.branding_candidates:
        return ImageDecision(ImageAction.SKIP, reason="no_watermark_detected")

    non_corner = [candidate for candidate in result.branding_candidates if not candidate.corners]
    if non_corner:
        return ImageDecision(
            ImageAction.ARCHIVE,
            archive_reason=ArchiveReason.NON_CORNER_WATERMARK,
            reason="watermark_not_confined_to_corner",
        )

    removed_ratio = removed_area_ratio(result.original_size, result.output_size)
    if removed_ratio > max_removed_area_ratio:
        return ImageDecision(
            ImageAction.ARCHIVE,
            removed_area_ratio=removed_ratio,
            archive_reason=ArchiveReason.EXCESSIVE_CROP_AREA,
            reason="crop_would_remove_too_much_image_area",
        )

    if not result.crop_result.needs_crop:
        return ImageDecision(ImageAction.SKIP, removed_area_ratio=removed_ratio, reason="no_crop_needed")

    return ImageDecision(ImageAction.CROP, removed_area_ratio=removed_ratio, reason="corner_watermark_crop")


def removed_area_ratio(original_size: tuple[int, int], output_size: tuple[int, int]) -> float:
    """Return the fraction of original area that would be removed."""
    original_area = max(0, original_size[0]) * max(0, original_size[1])
    if original_area == 0:
        return 1.0
    output_area = max(0, output_size[0]) * max(0, output_size[1])
    return max(0.0, min(1.0, 1.0 - (output_area / original_area)))


def safe_archive_path(source_path: str, archive_dir: str) -> str:
    """Return a collision-safe archive path preserving the source basename."""
    os.makedirs(archive_dir, exist_ok=True)
    stem, ext = os.path.splitext(os.path.basename(source_path))
    candidate = os.path.join(archive_dir, stem + ext)
    if not os.path.exists(candidate):
        return candidate

    index = 1
    while True:
        candidate = os.path.join(archive_dir, f"{stem}__cornercrop_{index}{ext}")
        if not os.path.exists(candidate):
            return candidate
        index += 1


def inspect_image(image_path: str, config: LibraryRunConfig) -> ProcessResult:
    """Run one OCR pass and build a corner-only crop result plus archive signals."""
    with Image.open(image_path) as image:
        width, height = image.size
        text_regions = _collect_text_regions(
            image_path,
            image,
            min_confidence=config.min_confidence,
        )

        branding_candidates = find_branding_candidates(
            text_regions,
            width,
            height,
            top_frac=TOP_BRANDING_ZONE_FRACTION,
            bottom_frac=BOTTOM_BRANDING_ZONE_FRACTION,
            edge_frac=EDGE_BRANDING_ZONE_FRACTION,
        )
        corner_candidates = find_corner_watermarks(
            text_regions,
            width,
            height,
            corner_frac=config.corner_frac,
        )
        candidates = _merge_candidates(corner_candidates, branding_candidates)
        crop_candidates = [candidate for candidate in candidates if candidate.corners]
        crop_result = compute_crop(
            crop_candidates,
            width,
            height,
            strategy=CropProfile.STRIP,
            margin=config.margin,
            max_crop_frac=config.max_crop_frac,
        )

    return ProcessResult(
        input_path=image_path,
        output_path=None,
        original_size=(width, height),
        output_size=crop_result.output_size,
        text_regions=text_regions,
        branding_candidates=candidates,
        crop_result=crop_result,
        selected_profile=CropProfile.STRIP,
        verification_status=VerificationStatus.NOT_RUN,
        residual_text_matches=[],
        saved=False,
    )


def run_library(config: LibraryRunConfig) -> dict:
    """Run a segmented, resumable library pass."""
    os.makedirs(config.state_dir, exist_ok=True)
    db = JobDatabase(os.path.join(config.state_dir, "cornercrop-job.sqlite3"))
    run_started_at = time.time()
    _write_run_config(config)
    _write_progress(config.state_dir, "starting", db.summary(), None, None)

    try:
        top_dirs = _immediate_directories(config.root)
        if not top_dirs:
            top_dirs = [os.path.abspath(config.root)]

        for index, top_dir in enumerate(top_dirs, start=1):
            if _should_stop(config):
                _write_progress(config.state_dir, "stopped", db.summary(), None, {"stopped_at": top_dir})
                break
            albums = list(iter_album_dirs(top_dir, archive_dir_name=config.archive_dir_name))
            _write_progress(
                config.state_dir,
                "segment_discovered",
                db.summary(),
                None,
                {"segment": top_dir, "segment_index": index, "segment_total": len(top_dirs), "albums": len(albums)},
            )
            if not albums:
                continue

            adaptive = _adaptive_config(config)
            segment_started_at = time.time()
            process_batch(
                albums,
                lambda album: _process_album(album, config, db),
                adaptive,
                progress_callback=lambda completed, total, target, snapshot: _write_progress(
                    config.state_dir,
                    "running",
                    db.summary(),
                    None,
                    {
                        "segment": top_dir,
                        "segment_index": index,
                        "segment_total": len(top_dirs),
                        "segment_completed_albums": completed,
                        "segment_total_albums": total,
                        "target_album_workers": target,
                        "cpu_percent": snapshot.cpu_percent,
                        "memory_percent": snapshot.memory_percent,
                        "read_mbps": snapshot.read_mbps,
                        "write_mbps": snapshot.write_mbps,
                        **_progress_metrics(completed, total, segment_started_at),
                    },
                ),
            )

        final_summary = db.summary()
        _write_progress(
            config.state_dir,
            "finished",
            final_summary,
            None,
            {"elapsed_seconds": round(time.time() - run_started_at, 1)},
        )
        return final_summary
    finally:
        db.close()


def _process_album(album_path: str, config: LibraryRunConfig, db: JobDatabase) -> dict[str, int]:
    image_paths = _direct_image_files(album_path)
    counts = {"total": len(image_paths), "processed": 0, "cropped": 0, "skipped": 0, "archived": 0, "failed": 0}
    db.mark_album_started(album_path, len(image_paths))
    monitor = ResourceMonitor(config.resource_poll_interval)
    monitor.start()
    stopped = False
    try:
        for image_path in image_paths:
            if _should_stop(config):
                stopped = True
                break
            if not config.force and db.has_completed_image(image_path, dry_run=config.dry_run):
                counts["processed"] += 1
                continue
            _wait_for_headroom(monitor, config)
            if _should_stop(config):
                stopped = True
                break
            action = _process_image_path(album_path, image_path, config, db)
            counts["processed"] += 1
            if action == "cropped":
                counts["cropped"] += 1
            elif action == "skipped":
                counts["skipped"] += 1
            elif action == "archived":
                counts["archived"] += 1
            elif action == "failed":
                counts["failed"] += 1

            if counts["processed"] % max(1, config.progress_every) == 0:
                db.update_album_counts(album_path, counts)
        if stopped and counts["processed"] < counts["total"]:
            db.mark_album_stopped(album_path, counts)
        else:
            db.mark_album_done(album_path, counts)
    except Exception as exc:
        db.mark_album_done(album_path, counts, error=str(exc))
    finally:
        monitor.stop()
        monitor.join(timeout=1.0)
    return counts


def _process_image_path(
    album_path: str,
    image_path: str,
    config: LibraryRunConfig,
    db: JobDatabase,
) -> str:
    try:
        result = inspect_image(image_path, config)
        decision = build_image_decision(result, max_removed_area_ratio=config.max_removed_area_ratio)

        if decision.action == ImageAction.SKIP:
            action = "would_skip" if config.dry_run else "skipped"
            db.record_image(
                path=image_path,
                album_path=album_path,
                action=action,
                reason=decision.reason,
                output_path=None,
                original_size=result.original_size,
                output_size=result.output_size,
                removed_area_ratio=decision.removed_area_ratio,
                dry_run=config.dry_run,
            )
            return "skipped"

        if decision.action == ImageAction.ARCHIVE:
            archive_reason = decision.archive_reason or ArchiveReason.NON_CORNER_WATERMARK
            archive_dir = os.path.join(album_path, config.archive_dir_name, archive_reason.value)
            archive_path = safe_archive_path(image_path, archive_dir)
            if not config.dry_run:
                shutil.move(image_path, archive_path)
            action = "would_archive" if config.dry_run else "archived"
            db.record_image(
                path=image_path,
                album_path=album_path,
                action=action,
                reason=archive_reason.value,
                output_path=archive_path,
                original_size=result.original_size,
                output_size=result.output_size,
                removed_area_ratio=decision.removed_area_ratio,
                dry_run=config.dry_run,
            )
            return "archived"

        if not config.dry_run:
            _save_crop_in_place(image_path, result)
        action = "would_crop" if config.dry_run else "cropped"
        db.record_image(
            path=image_path,
            album_path=album_path,
            action=action,
            reason=decision.reason,
            output_path=image_path,
            original_size=result.original_size,
            output_size=result.output_size,
            removed_area_ratio=decision.removed_area_ratio,
            dry_run=config.dry_run,
        )
        return "cropped"
    except UnidentifiedImageError as exc:
        return _archive_unreadable_image(album_path, image_path, config, db, str(exc))
    except Exception as exc:
        db.record_image(
            path=image_path,
            album_path=album_path,
            action="failed",
            reason="processing_error",
            output_path=None,
            original_size=None,
            output_size=None,
            removed_area_ratio=0.0,
            dry_run=config.dry_run,
            error=str(exc),
        )
        return "failed"


def _archive_unreadable_image(
    album_path: str,
    image_path: str,
    config: LibraryRunConfig,
    db: JobDatabase,
    error: str,
) -> str:
    archive_dir = os.path.join(album_path, config.archive_dir_name, ArchiveReason.UNREADABLE_IMAGE.value)
    archive_path = safe_archive_path(image_path, archive_dir)
    if not config.dry_run:
        try:
            shutil.move(image_path, archive_path)
        except Exception as exc:
            db.record_image(
                path=image_path,
                album_path=album_path,
                action="failed",
                reason="unreadable_image_archive_error",
                output_path=None,
                original_size=None,
                output_size=None,
                removed_area_ratio=0.0,
                dry_run=config.dry_run,
                error=f"{error}; archive failed: {exc}",
            )
            return "failed"

    action = "would_archive" if config.dry_run else "archived"
    db.record_image(
        path=image_path,
        album_path=album_path,
        action=action,
        reason=ArchiveReason.UNREADABLE_IMAGE.value,
        output_path=archive_path,
        original_size=None,
        output_size=None,
        removed_area_ratio=0.0,
        dry_run=config.dry_run,
    )
    return "archived"


def _save_crop_in_place(image_path: str, result: ProcessResult) -> None:
    with Image.open(image_path) as image:
        final_image = _build_processed_image(image, result.crop_result)
        try:
            ext = os.path.splitext(image_path)[1].lower()
            save_kwargs = {}
            if ext in (".jpg", ".jpeg"):
                save_kwargs["quality"] = 95
            _save_image_atomic(final_image, image_path, **save_kwargs)
        finally:
            final_image.close()


def _merge_candidates(corner_candidates, branding_candidates):
    merged = []
    seen = set()
    for candidate in list(corner_candidates) + list(branding_candidates):
        bbox = candidate.px_bbox
        key = (
            candidate.text.strip().lower(),
            bbox.get("x"),
            bbox.get("y"),
            bbox.get("w"),
            bbox.get("h"),
            tuple(sorted(candidate.matched_rules)),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(candidate)
    return merged


def _direct_image_files(directory: str) -> list[str]:
    try:
        with os.scandir(directory) as entries:
            paths = [
                entry.path
                for entry in entries
                if entry.is_file(follow_symlinks=False) and _is_supported_image(entry.path)
            ]
    except OSError:
        return []
    return sorted(paths)


def _immediate_directories(root: str) -> list[str]:
    root = os.path.abspath(root)
    try:
        with os.scandir(root) as entries:
            return sorted(
                entry.path
                for entry in entries
                if entry.is_dir(follow_symlinks=False)
                and entry.name != DEFAULT_ARCHIVE_DIR_NAME
                and not entry.name.startswith(".")
            )
    except OSError:
        return []


def _is_supported_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS


def _adaptive_config(config: LibraryRunConfig) -> AdaptiveParallelismConfig:
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
        max_workers=config.max_album_workers,
        poll_interval=config.resource_poll_interval,
        progress_interval=1,
        heartbeat_interval=max(config.resource_poll_interval, 30.0),
        **profile_defaults,
    )


def _wait_for_headroom(monitor: ResourceMonitor, config: LibraryRunConfig) -> None:
    profile = _adaptive_config(config)
    while True:
        snapshot = monitor.latest()
        if (
            snapshot.cpu_percent < profile.cpu_high_water
            and snapshot.memory_percent < profile.memory_high_water
            and snapshot.read_mbps < profile.read_high_mbps
            and snapshot.write_mbps < profile.write_high_mbps
        ):
            return
        if _should_stop(config):
            return
        time.sleep(config.resource_poll_interval)


def _should_stop(config: LibraryRunConfig) -> bool:
    return bool(config.stop_file and os.path.exists(config.stop_file))


def _write_run_config(config: LibraryRunConfig) -> None:
    payload = dict(config.__dict__)
    path = os.path.join(config.state_dir, "run-config.json")
    _atomic_json_dump(path, payload)


def _write_progress(
    state_dir: str,
    phase: str,
    summary: dict,
    active_album: str | None,
    extra: dict | None,
) -> None:
    payload = {
        "phase": phase,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "active_album": active_album,
        "summary": summary,
    }
    if extra:
        payload.update(extra)
    _atomic_json_dump(os.path.join(state_dir, "progress.json"), payload)


def _progress_metrics(completed: int, total: int, started_at: float) -> dict:
    elapsed = max(0.0, time.time() - started_at)
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
        prog="cornercrop-library",
        description="Run a resumable album-level CornerCrop pass over a large image library.",
    )
    parser.add_argument("root", help="Root image-library directory")
    parser.add_argument("--state-dir", default=DEFAULT_STATE_DIR, help=f"Local state directory (default: {DEFAULT_STATE_DIR})")
    parser.add_argument("--archive-dir-name", default=DEFAULT_ARCHIVE_DIR_NAME, help="Archive subfolder name inside each album")
    parser.add_argument("--dry-run", action="store_true", help="Record decisions without modifying images")
    parser.add_argument("--force", action="store_true", help="Reprocess images already marked complete in this state dir")
    parser.add_argument("--max-album-workers", type=_positive_int, default=2, help="Maximum albums processed concurrently")
    parser.add_argument("--resource-profile", choices=["conservative", "balanced", "aggressive"], default="conservative")
    parser.add_argument("--resource-poll-interval", type=_positive_float, default=5.0)
    parser.add_argument("--progress-every", type=_positive_int, default=10, help="Images per album between album checkpoint writes")
    parser.add_argument("--min-confidence", type=_bounded_float(0.0, 1.0, "minimum confidence"), default=0.25)
    parser.add_argument("--corner", type=_bounded_float(0.0, 0.5, "corner fraction"), default=0.22)
    parser.add_argument("--margin", type=_non_negative_int, default=8)
    parser.add_argument("--max-crop", type=_bounded_float(0.0, 1.0, "max crop fraction"), default=0.30)
    parser.add_argument("--max-removed-area", type=_bounded_float(0.0, 1.0, "max removed area"), default=0.30)
    parser.add_argument("--stop-file", help="If this file appears, stop after the current image")
    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = LibraryRunConfig(
        root=os.path.abspath(args.root),
        state_dir=os.path.abspath(args.state_dir),
        archive_dir_name=args.archive_dir_name,
        dry_run=args.dry_run,
        force=args.force,
        max_album_workers=args.max_album_workers,
        resource_profile=args.resource_profile,
        resource_poll_interval=args.resource_poll_interval,
        progress_every=args.progress_every,
        min_confidence=args.min_confidence,
        corner_frac=args.corner,
        margin=args.margin,
        max_crop_frac=args.max_crop,
        max_removed_area_ratio=args.max_removed_area,
        stop_file=os.path.abspath(args.stop_file) if args.stop_file else None,
    )
    if not os.path.isdir(config.root):
        parser.error(f"Root directory does not exist: {config.root}")

    summary = run_library(config)
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
    sys.exit(main())
