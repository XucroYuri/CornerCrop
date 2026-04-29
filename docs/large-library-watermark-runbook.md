# Large image-library watermark processing runbook

This note records reusable operating lessons from a production-scale NAS image
library run. It intentionally avoids local paths, private collection names,
exact corpus sizes, and per-user environment details so it can be kept in a
public repository.

## Recognition Lessons

The first pass should stay conservative. It is better to archive ambiguous
non-corner branding than to crop away too much content in the active album tree.

The second pass can be more deliberate because it only sees already-isolated
non-corner candidates. The reliable pattern is:

1. Run OCR on the full image and focused edge regions.
2. Classify branding by both text content and broad edge placement.
3. Compute all plausible crop profiles before saving anything.
4. Sort viable crops by removed-area ratio, using profile preference only as a tie-breaker.
5. Verify the cropped image with OCR again.
6. Recover only when verification is clean and removed area is within the configured limit.
7. Leave the archive source untouched for residual, excessive, unreadable, or no-crop cases.

A clean crop should not be accepted merely because it was evaluated first. A
lower-loss clean crop is preferable, especially when an aggressive profile is
suggested by broad top/bottom branding but a single strip crop already removes
the detectable watermark.

The removed-area threshold should be treated as an upper boundary, not as a
target. Floating-point comparisons use a tiny epsilon so exact boundary cases
are not rejected due to rounding noise.

## Process Management Lessons

For large NAS-backed runs, the bottleneck is often OCR/CPU rather than network
storage throughput. Start with a small dry run, then increase worker count only
when CPU, memory, and disk metrics show stable headroom.

Useful defaults for this class of run:

- Start with a dry-run sample before any in-place write.
- Use local SQLite state under `runs/` rather than writing state into the NAS tree.
- Use WAL plus frequent per-image records so recovery is resumable.
- Keep progress JSON small and overwrite it atomically.
- Avoid repeated full-tree filesystem scans while the main job is running.
- Use a stop file for graceful interruption.
- Use heartbeat progress so long in-flight OCR batches do not look stalled.
- Audit after completion by cross-checking database rows against filesystem state.

## Post-Run Audit Checklist

For recovery passes:

1. `progress.json` phase is `finished`.
2. SQLite total equals the discovered source count.
3. `failed = 0`, or failed rows are explicitly triaged.
4. Every recovered row has an existing output path.
5. No recovered row still has its archive source path.
6. Every kept row still has its archive source path.
7. Remaining files in the archive reason directory equal the kept count.
8. Export kept rows to a local, ignored review artifact when manual review is needed.

## Current Automation

- `cornercrop-library` handles the first large album pass.
- `cornercrop-recover-non-corner` handles second-pass recovery from
  `_cornercrop_archive/non_corner_watermark`.
- `scripts/watch_library_cornercrop.py` and `scripts/watch_non_corner_recovery.py`
  read progress JSON and SQLite summaries.
- Batch scheduling adapts workers based on CPU, memory, disk throughput, load,
  and heartbeat progress.
