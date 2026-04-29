#!/usr/bin/env python3
"""Audit a CornerCrop library run against the current image tree."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import asdict, dataclass, field

DEFAULT_STATE_DIR = os.path.join("runs", "library-watermark", "live")
DEFAULT_ARCHIVE_DIR_NAME = "_cornercrop_archive"
SUPPORTED_IMAGE_EXTENSIONS = {".bmp", ".heic", ".heif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
COMPLETED_DIRECT_ACTIONS = {"cropped", "skipped", "would_crop", "would_skip"}
ARCHIVE_ACTIONS = {"archived", "would_archive"}


@dataclass
class AuditReport:
    root: str
    db_path: str
    direct_images_scanned: int = 0
    processed_direct_images: int = 0
    missing_direct_images: list[str] = field(default_factory=list)
    failed_db_images: list[str] = field(default_factory=list)
    missing_archive_outputs: list[str] = field(default_factory=list)
    archive_outputs_checked: int = 0

    @property
    def ok(self) -> bool:
        return not self.missing_direct_images and not self.failed_db_images and not self.missing_archive_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit CornerCrop large-library completion.")
    parser.add_argument("root", help="Root image-library directory")
    parser.add_argument("--state-dir", default=DEFAULT_STATE_DIR)
    parser.add_argument("--archive-dir-name", default=DEFAULT_ARCHIVE_DIR_NAME)
    parser.add_argument("--sample-limit", type=int, default=20)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    report = audit_run(
        root=os.path.abspath(args.root),
        db_path=os.path.abspath(os.path.join(args.state_dir, "cornercrop-job.sqlite3")),
        archive_dir_name=args.archive_dir_name,
        sample_limit=max(0, args.sample_limit),
    )
    if args.json:
        payload = asdict(report)
        payload["ok"] = report.ok
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_human(report)
    return 0 if report.ok else 1


def audit_run(root: str, db_path: str, archive_dir_name: str, sample_limit: int) -> AuditReport:
    if not os.path.isdir(root):
        raise SystemExit(f"Root directory does not exist: {root}")
    if not os.path.exists(db_path):
        raise SystemExit(f"CornerCrop database does not exist: {db_path}")

    direct_records, failed_paths, archive_outputs = _read_db(db_path)
    report = AuditReport(root=root, db_path=db_path, failed_db_images=_limited(failed_paths, sample_limit))

    for image_path in _iter_direct_images(root, archive_dir_name):
        report.direct_images_scanned += 1
        if image_path in direct_records:
            report.processed_direct_images += 1
            continue
        if len(report.missing_direct_images) < sample_limit:
            report.missing_direct_images.append(image_path)

    for output_path in archive_outputs:
        report.archive_outputs_checked += 1
        if output_path and not os.path.exists(output_path) and len(report.missing_archive_outputs) < sample_limit:
            report.missing_archive_outputs.append(output_path)

    return report


def _read_db(db_path: str) -> tuple[set[str], list[str], list[str]]:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        direct_records = {
            row[0]
            for row in connection.execute(
                f"SELECT path FROM images WHERE error IS NULL AND action IN ({_placeholders(COMPLETED_DIRECT_ACTIONS)})",
                tuple(COMPLETED_DIRECT_ACTIONS),
            )
        }
        failed_paths = [
            row[0]
            for row in connection.execute(
                "SELECT path FROM images WHERE error IS NOT NULL ORDER BY updated_at DESC"
            )
        ]
        archive_outputs = [
            row[0]
            for row in connection.execute(
                f"""
                SELECT output_path FROM images
                WHERE error IS NULL
                  AND action IN ({_placeholders(ARCHIVE_ACTIONS)})
                  AND output_path IS NOT NULL
                """,
                tuple(ARCHIVE_ACTIONS),
            )
        ]
    finally:
        connection.close()
    return direct_records, failed_paths, archive_outputs


def _iter_direct_images(root: str, archive_dir_name: str):
    ignored_names = {archive_dir_name, ".cornercrop", "__MACOSX"}
    stack = [root]
    while stack:
        directory = stack.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.name in ignored_names or entry.name.startswith(".cornercrop-"):
                        continue
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False) and _is_supported_image(entry.path):
                            yield os.path.abspath(entry.path)
                    except OSError:
                        continue
        except OSError:
            continue


def _is_supported_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS


def _placeholders(values: set[str]) -> str:
    return ",".join("?" for _ in values)


def _limited(values: list[str], limit: int) -> list[str]:
    if limit == 0:
        return []
    return values[:limit]


def print_human(report: AuditReport) -> None:
    status = "ok" if report.ok else "needs_attention"
    print(
        f"status={status} direct_scanned={report.direct_images_scanned} "
        f"direct_recorded={report.processed_direct_images} "
        f"missing_direct_samples={len(report.missing_direct_images)} "
        f"failed_db_samples={len(report.failed_db_images)} "
        f"archive_outputs_checked={report.archive_outputs_checked} "
        f"missing_archive_samples={len(report.missing_archive_outputs)}"
    )
    for label, paths in (
        ("missing_direct", report.missing_direct_images),
        ("failed_db", report.failed_db_images),
        ("missing_archive_output", report.missing_archive_outputs),
    ):
        for path in paths:
            print(f"{label}: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
