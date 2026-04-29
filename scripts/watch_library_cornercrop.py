#!/usr/bin/env python3
"""Print or poll a CornerCrop large-library progress file."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time

DEFAULT_STATE_DIR = os.path.join("runs", "library-watermark", "live")


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor a CornerCrop large-library run.")
    parser.add_argument("--state-dir", default=DEFAULT_STATE_DIR)
    parser.add_argument("--watch", action="store_true", help="Poll until interrupted")
    parser.add_argument("--interval", type=float, default=30.0, help="Polling interval in seconds")
    args = parser.parse_args()

    while True:
        print_status(args.state_dir)
        if not args.watch:
            return 0
        time.sleep(args.interval)


def print_status(state_dir: str) -> None:
    progress_path = os.path.join(state_dir, "progress.json")
    db_path = os.path.join(state_dir, "cornercrop-job.sqlite3")
    if not os.path.exists(progress_path):
        print(f"No progress file yet: {progress_path}")
        return

    with open(progress_path, "r", encoding="utf-8") as handle:
        progress = json.load(handle)

    summary = dict(progress.get("summary") or {})
    if os.path.exists(db_path):
        summary.update(read_db_summary(db_path))

    print(
        "{updated_at} | phase={phase} | albums {albums_done}/{albums_seen} done, "
        "{albums_running} running, {albums_stopped} stopped, {albums_failed} failed | images processed={images_processed}, "
        "cropped={images_cropped}, archived={images_archived}, skipped={images_skipped}, failed={images_failed}".format(
            updated_at=progress.get("updated_at", "unknown"),
            phase=progress.get("phase", "unknown"),
            albums_done=summary.get("albums_done", 0),
            albums_seen=summary.get("albums_seen", 0),
            albums_running=summary.get("albums_running", 0),
            albums_stopped=summary.get("albums_stopped", 0),
            albums_failed=summary.get("albums_failed", 0),
            images_processed=summary.get("images_processed", 0),
            images_cropped=summary.get("images_cropped", 0),
            images_archived=summary.get("images_archived", 0),
            images_skipped=summary.get("images_skipped", 0),
            images_failed=summary.get("images_failed", 0),
        )
    )
    for key in (
        "segment",
        "segment_completed_albums",
        "segment_total_albums",
        "completion_ratio",
        "items_per_minute",
        "estimated_remaining_seconds",
        "target_album_workers",
        "read_mbps",
        "write_mbps",
    ):
        if key in progress:
            print(f"  {key}: {progress[key]}")


def read_db_summary(db_path: str) -> dict:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        image_rows = connection.execute(
            """
            SELECT
                COUNT(*) AS images_processed,
                SUM(CASE WHEN action IN ('cropped', 'would_crop') THEN 1 ELSE 0 END) AS images_cropped,
                SUM(CASE WHEN action IN ('skipped', 'would_skip') THEN 1 ELSE 0 END) AS images_skipped,
                SUM(CASE WHEN action IN ('archived', 'would_archive') THEN 1 ELSE 0 END) AS images_archived,
                SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS images_failed
            FROM images
            """
        ).fetchone()
        album_rows = connection.execute(
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
    finally:
        connection.close()

    return {
        "images_processed": int(image_rows[0] or 0),
        "images_cropped": int(image_rows[1] or 0),
        "images_skipped": int(image_rows[2] or 0),
        "images_archived": int(image_rows[3] or 0),
        "images_failed": int(image_rows[4] or 0),
        "albums_seen": int(album_rows[0] or 0),
        "albums_done": int(album_rows[1] or 0),
        "albums_running": int(album_rows[2] or 0),
        "albums_stopped": int(album_rows[3] or 0),
        "albums_failed": int(album_rows[4] or 0),
    }


if __name__ == "__main__":
    raise SystemExit(main())
