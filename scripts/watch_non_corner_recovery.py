#!/usr/bin/env python3
"""Print or poll non-corner archive recovery progress."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time

DEFAULT_STATE_DIR = os.path.join("runs", "non-corner-recovery", "live")


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor a non-corner recovery run.")
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
    db_path = os.path.join(state_dir, "non-corner-recovery.sqlite3")
    if not os.path.exists(progress_path):
        print(f"No progress file yet: {progress_path}")
        return

    with open(progress_path, "r", encoding="utf-8") as handle:
        progress = json.load(handle)

    summary = dict(progress.get("summary") or {})
    if os.path.exists(db_path):
        summary.update(read_db_summary(db_path))

    print(
        "{updated_at} | phase={phase} | processed={processed}, recovered={recovered}, "
        "kept={kept}, failed={failed}".format(
            updated_at=progress.get("updated_at", "unknown"),
            phase=progress.get("phase", "unknown"),
            processed=summary.get("images_processed", 0),
            recovered=summary.get("images_recovered", 0),
            kept=summary.get("images_kept", 0),
            failed=summary.get("images_failed", 0),
        )
    )
    for key in (
        "completed_sources",
        "total_sources",
        "completion_ratio",
        "items_per_minute",
        "estimated_remaining_seconds",
        "target_workers",
        "read_mbps",
        "write_mbps",
    ):
        if key in progress:
            print(f"  {key}: {progress[key]}")


def read_db_summary(db_path: str) -> dict:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS images_processed,
                SUM(CASE WHEN action IN ('recovered', 'would_recover') THEN 1 ELSE 0 END) AS images_recovered,
                SUM(CASE WHEN action IN ('kept', 'would_keep') THEN 1 ELSE 0 END) AS images_kept,
                SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS images_failed
            FROM images
            """
        ).fetchone()
    finally:
        connection.close()

    return {
        "images_processed": int(row[0] or 0),
        "images_recovered": int(row[1] or 0),
        "images_kept": int(row[2] or 0),
        "images_failed": int(row[3] or 0),
    }


if __name__ == "__main__":
    raise SystemExit(main())
