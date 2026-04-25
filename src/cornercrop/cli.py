"""Command-line interface for CornerCrop."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import sys
from dataclasses import dataclass

from . import __version__
from .batch import AdaptiveParallelismConfig, ResourceSnapshot, process_batch
from .cropper import CropProfile, CropStrategy
from .pipeline import (
    CropOverride,
    ProcessResult,
    VerificationStatus,
    apply_crop_override,
    process_image,
)

SUPPORTED_IMAGE_EXTENSIONS = {
    ".bmp",
    ".heic",
    ".heif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class InputItem:
    """A concrete image input plus enough context to preserve output structure."""

    path: str
    source_root: str
    relative_path: str


@dataclass(frozen=True)
class BatchFallbackConfig:
    """Configurable heuristics for batch-consensus fallback and review detection."""

    enabled: bool = True
    scope: str = "album"
    min_support: int = 5
    min_cropped_ratio: float = 0.6
    min_crop_px: int = 45
    max_crop_px: int = 90


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="cornercrop",
        description="Detect, crop, and verify branding text removal in images using macOS Vision OCR.",
    )
    parser.add_argument("input", nargs="+", help="Input image path(s) or directory path(s)")

    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--output", "-o", help="Output image path (single image mode only)")
    output_group.add_argument("--output-dir", help="Output directory for batch or directory mode")
    output_group.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite cropped images back to the source files",
    )
    parser.add_argument(
        "--backup-dir",
        help="Optional backup directory for originals when using --in-place",
    )

    parser.add_argument(
        "--profile",
        choices=[profile.value for profile in CropProfile],
        default=CropProfile.AUTO.value,
        help="Crop profile: auto (default), strip, cover, or corner",
    )
    parser.add_argument(
        "--strategy",
        choices=["strip", "corner"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--margin",
        type=_non_negative_int,
        default=10,
        help="Extra margin in pixels around detected branding text (default: 10)",
    )
    parser.add_argument(
        "--corner",
        type=_bounded_float(0.0, 0.5, "corner fraction"),
        default=0.20,
        help="Legacy corner region fraction for corner mode (0-0.5, default: 0.20)",
    )
    parser.add_argument(
        "--min-confidence",
        type=_bounded_float(0.0, 1.0, "minimum confidence"),
        default=0.3,
        help="Minimum OCR confidence threshold (default: 0.3)",
    )
    parser.add_argument(
        "--max-crop",
        type=_bounded_float(0.0, 1.0, "max crop fraction"),
        default=0.25,
        help="Maximum fraction of any edge to crop (default: 0.25)",
    )
    parser.add_argument(
        "--verify",
        dest="verify",
        action="store_true",
        default=True,
        help="Verify processed images for residual branding text (default: on)",
    )
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="Skip post-processing verification",
    )
    parser.add_argument(
        "--fail-on-residual",
        action="store_true",
        help="Return a non-zero exit code when verification still finds branding text",
    )
    parser.add_argument(
        "--report-json",
        help="Write a batch report JSON file with summary and per-image results",
    )
    parser.add_argument(
        "--review-candidates-dir",
        help="Export unresolved/suspicious images for manual review into this directory",
    )
    parser.add_argument(
        "--batch-fallback",
        dest="batch_fallback",
        action="store_true",
        default=True,
        help="Use batch-consensus fallback for low-contrast misses (default: on)",
    )
    parser.add_argument(
        "--no-batch-fallback",
        dest="batch_fallback",
        action="store_false",
        help="Disable batch-consensus fallback",
    )
    parser.add_argument(
        "--batch-fallback-scope",
        choices=["album", "global"],
        default="album",
        help="Consensus scope for batch fallback: album (default) or global",
    )
    parser.add_argument(
        "--batch-fallback-min-support",
        type=_non_negative_int,
        default=5,
        help="Minimum number of cropped peers required before fallback can trigger (default: 5)",
    )
    parser.add_argument(
        "--batch-fallback-min-ratio",
        type=_bounded_float(0.0, 1.0, "batch fallback min ratio"),
        default=0.6,
        help="Minimum cropped ratio inside a bucket before fallback can trigger (default: 0.6)",
    )
    parser.add_argument(
        "--batch-fallback-min-crop",
        type=_non_negative_int,
        default=45,
        help="Minimum consensus bottom crop in pixels for fallback (default: 45)",
    )
    parser.add_argument(
        "--batch-fallback-max-crop",
        type=_non_negative_int,
        default=90,
        help="Maximum consensus bottom crop in pixels for fallback (default: 90)",
    )
    parser.add_argument(
        "--adaptive-workers",
        dest="adaptive_workers",
        action="store_true",
        default=True,
        help="Use resource-aware adaptive worker scheduling for batch runs (default: on)",
    )
    parser.add_argument(
        "--no-adaptive-workers",
        dest="adaptive_workers",
        action="store_false",
        help="Disable adaptive worker scheduling",
    )
    parser.add_argument(
        "--min-workers",
        type=_positive_int,
        default=1,
        help="Minimum worker count for adaptive scheduling (default: 1)",
    )
    parser.add_argument(
        "--max-workers",
        type=_positive_int,
        help="Optional maximum worker count override for adaptive scheduling",
    )
    parser.add_argument(
        "--resource-profile",
        choices=["conservative", "balanced", "aggressive"],
        default="balanced",
        help="Adaptive scheduling profile tuned for different Macs (default: balanced)",
    )
    parser.add_argument(
        "--resource-poll-interval",
        type=_positive_float,
        default=2.0,
        help="Seconds between hardware resource samples (default: 2.0)",
    )
    parser.add_argument(
        "--progress-interval",
        type=_positive_int,
        default=25,
        help="Emit progress every N completed images during large batch runs (default: 25)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect and simulate crop only; do not save processed images",
    )
    parser.add_argument("--json", action="store_true", help="Output per-image results as JSON")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    args = parser.parse_args(argv)
    try:
        input_items = _collect_input_paths(args.input)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    if not input_items:
        parser.error("No supported image files found in the provided inputs")
    if args.output and len(input_items) != 1:
        parser.error("--output can only be used when exactly one input image is selected")
    if args.backup_dir and not args.in_place:
        parser.error("--backup-dir can only be used together with --in-place")
    if args.review_candidates_dir:
        os.makedirs(args.review_candidates_dir, exist_ok=True)

    if args.strategy:
        profile = CropProfile.CORNER if args.strategy == "corner" else CropProfile.STRIP
    else:
        profile = CropProfile(args.profile)

    fallback_config = BatchFallbackConfig(
        enabled=args.batch_fallback,
        scope=args.batch_fallback_scope,
        min_support=args.batch_fallback_min_support,
        min_cropped_ratio=args.batch_fallback_min_ratio,
        min_crop_px=args.batch_fallback_min_crop,
        max_crop_px=args.batch_fallback_max_crop,
    )
    adaptive_config = _adaptive_config_from_args(args)

    if args.output_dir and not args.dry_run:
        os.makedirs(args.output_dir, exist_ok=True)
    if args.backup_dir and not args.dry_run:
        os.makedirs(args.backup_dir, exist_ok=True)
    if args.report_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.report_json)) or ".", exist_ok=True)

    records = []
    results: list[ProcessResult] = []
    failure_count = 0
    residual_count = 0

    input_items_by_path = {item.path: item for item in input_items}
    if len(input_items) > 1:
        batch_outputs = process_batch(
            input_items,
            lambda item: _process_single_item(item, args, profile),
            adaptive_config,
            progress_callback=(
                None
                if args.json
                else lambda completed, total, target, snapshot: _print_progress(
                    completed,
                    total,
                    target,
                    snapshot,
                )
            ),
        )
    else:
        batch_outputs = [_process_single_item(input_items[0], args, profile)]

    for output in batch_outputs:
        if output["error"] is not None:
            failure_count += 1
            error_record = _error_to_dict(output["path"], output["error"])
            records.append(error_record)
            if not args.json:
                _print_error(output["path"], output["error"])
            continue

        result = output["result"]
        if result.verification_status == VerificationStatus.RESIDUAL:
            residual_count += 1
        results.append(result)
        records.append(_result_to_dict(result))
        if not args.json:
            _print_human(result)

    fallback_overrides = _build_batch_fallback_overrides(results, fallback_config)
    if fallback_overrides:
        results_by_input = {result.input_path: result for result in results}
        records_by_input = {
            record["input"]: record
            for record in records
            if record.get("status") == "ok"
        }
        for path, override in fallback_overrides.items():
            base_result = results_by_input[path]
            result = apply_crop_override(
                base_result,
                override,
                output_path=_build_output_path(
                    input_items_by_path[path],
                    output_path=args.output,
                    output_dir=args.output_dir,
                    in_place=args.in_place,
                ),
                dry_run=args.dry_run,
                verify=args.verify,
            )
            if base_result.verification_status == VerificationStatus.RESIDUAL:
                residual_count = max(0, residual_count - 1)
            if result.verification_status == VerificationStatus.RESIDUAL:
                residual_count += 1
            results_by_input[path] = result
            records_by_input[path] = _result_to_dict(result)
            if not args.json:
                print(f"🔁 Fallback crop applied: {os.path.basename(path)}")
                _print_human(result)

        results = [results_by_input[result.input_path] for result in results]
        records = [
            records_by_input.get(record["input"], record)
            if record.get("status") == "ok"
            else record
            for record in records
        ]

    review_candidates = _collect_review_candidates(results, fallback_config)
    if args.review_candidates_dir:
        _export_review_candidates(
            review_candidates,
            input_items_by_path,
            {result.input_path: result for result in results},
            args.review_candidates_dir,
        )

    summary = _build_summary(records)
    summary["review_candidates"] = len(review_candidates)

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "summary": summary,
                    "fallback_config": _fallback_config_to_dict(fallback_config),
                    "adaptive_config": _adaptive_config_to_dict(adaptive_config),
                    "review_candidates": review_candidates,
                    "results": records,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )

    if args.json:
        print(json.dumps(records, indent=2, ensure_ascii=False))
    else:
        _print_summary(summary)

    if failure_count:
        return 1
    if args.fail_on_residual and residual_count:
        return 1
    return 0


def _print_human(result: ProcessResult) -> None:
    """Print a human-readable processing result."""
    width, height = result.original_size
    print(f"📷 {result.input_path} ({width}×{height})")
    print(
        f"🔍 {len(result.text_regions)} text region(s), "
        f"{len(result.branding_candidates)} branding candidate(s)"
    )

    for candidate in result.branding_candidates:
        anchors = ", ".join(candidate.anchors) or "unanchored"
        rules = ", ".join(candidate.matched_rules)
        print(
            f"  ⚠️  [{anchors}] \"{candidate.text}\" "
            f"(conf={candidate.confidence:.2f}, rules={rules}) px={candidate.px_bbox}"
        )

    if result.crop_result.needs_crop:
        crop_result = result.crop_result
        out_w, out_h = result.output_size
        removed_w, removed_h = crop_result.removed_px
        print(f"✂️  Crop ({result.selected_profile.value}): {crop_result.crop_box}")
        print(f"   Output: {out_w}×{out_h} (removed {removed_w}×{removed_h}px)")
        if result.crop_reasons:
            print(f"   Reasons: {', '.join(result.crop_reasons)}")
    else:
        print("✅ No branding crop needed")

    if result.verification_status == VerificationStatus.CLEAN:
        print("🧪 Verification: clean")
    elif result.verification_status == VerificationStatus.RESIDUAL:
        print("🧪 Verification: residual branding detected")
        for match in result.residual_text_matches[:5]:
            rules = ", ".join(match.matched_rules)
            print(
                f"   ↳ [{match.source}] \"{match.text}\" "
                f"(conf={match.confidence:.2f}, rules={rules}) px={match.px_bbox}"
            )
    elif result.verification_status == VerificationStatus.NOT_RUN:
        print("🧪 Verification: skipped")

    if result.saved:
        print(f"💾 Saved: {result.output_path}")
    elif result.crop_result.needs_crop and not result.saved:
        print("🏃 Dry run — no file saved")

    print()


def _result_to_dict(result: ProcessResult) -> dict:
    """Convert a ProcessResult into a JSON-serializable dict."""
    return {
        "status": "ok",
        "input": result.input_path,
        "output": result.output_path,
        "original_size": list(result.original_size),
        "output_size": list(result.output_size),
        "detected_text_regions": len(result.text_regions),
        "branding_candidates": [
            {
                "text": candidate.text,
                "confidence": candidate.confidence,
                "anchors": candidate.anchors,
                "matched_rules": candidate.matched_rules,
                "px_bbox": candidate.px_bbox,
            }
            for candidate in result.branding_candidates
        ],
        "crop_box": list(result.crop_result.crop_box),
        "crop_reasons": result.crop_reasons,
        "selected_profile": result.selected_profile.value,
        "verification_status": result.verification_status,
        "residual_text_matches": [
            {
                "source": match.source,
                "text": match.text,
                "confidence": match.confidence,
                "matched_rules": match.matched_rules,
                "px_bbox": match.px_bbox,
            }
            for match in result.residual_text_matches
        ],
        "saved": result.saved,
    }


def _error_to_dict(path: str, exc: Exception) -> dict:
    """Convert a processing error into a JSON-serializable dict."""
    return {
        "status": "error",
        "input": path,
        "error": str(exc),
    }


def _build_summary(records: list[dict]) -> dict:
    """Build aggregate statistics for a batch run."""
    ok_records = [record for record in records if record.get("status") == "ok"]
    return {
        "processed": len(records),
        "failed": len(records) - len(ok_records),
        "cropped": sum(1 for record in ok_records if record["original_size"] != record["output_size"]),
        "verified_clean": sum(
            1 for record in ok_records if record["verification_status"] == VerificationStatus.CLEAN
        ),
        "residual_detected": sum(
            1
            for record in ok_records
            if record["verification_status"] == VerificationStatus.RESIDUAL
        ),
        "verification_skipped": sum(
            1
            for record in ok_records
            if record["verification_status"] == VerificationStatus.NOT_RUN
        ),
    }


def _process_single_item(
    item: InputItem,
    args,
    profile: CropProfile,
) -> dict:
    """Process one image item and capture its exception boundary."""
    try:
        if args.in_place and args.backup_dir and not args.dry_run:
            _backup_original(item, args.backup_dir)
        result = process_image(
            image_path=item.path,
            output_path=_build_output_path(
                item,
                output_path=args.output,
                output_dir=args.output_dir,
                in_place=args.in_place,
            ),
            strategy=profile,
            corner_frac=args.corner,
            min_confidence=args.min_confidence,
            margin=args.margin,
            max_crop_frac=args.max_crop,
            dry_run=args.dry_run,
            verify=args.verify,
        )
        return {"path": item.path, "result": result, "error": None}
    except Exception as exc:
        return {"path": item.path, "result": None, "error": exc}


def _build_batch_fallback_overrides(
    results: list[ProcessResult],
    config: BatchFallbackConfig,
) -> dict[str, CropOverride]:
    """Infer a consensus bottom crop for low-contrast misses inside the same batch."""
    if not config.enabled:
        return {}

    buckets = _group_results_for_batch_fallback(results, config)
    overrides: dict[str, CropOverride] = {}
    for bucket in buckets.values():
        bottom_crops = [
            result.original_size[1] - result.output_size[1]
            for result in bucket
            if _is_bottom_only_crop(result)
        ]
        if len(bottom_crops) < config.min_support:
            continue
        cropped_ratio = len(bottom_crops) / len(bucket)
        if cropped_ratio < config.min_cropped_ratio:
            continue
        consensus = int(round(statistics.median(bottom_crops)))
        if not config.min_crop_px <= consensus <= config.max_crop_px:
            continue

        for result in bucket:
            if result.original_size != result.output_size:
                continue
            if result.branding_candidates:
                continue
            if result.selected_profile == CropProfile.COVER:
                continue

            width, height = result.original_size
            overrides[result.input_path] = CropOverride(
                crop_box=(0, 0, width, max(0, height - consensus)),
                selected_profile=CropProfile.STRIP,
                crop_reasons=[
                    f"batch-bottom-fallback:{consensus}px",
                    f"batch-scope:{config.scope}",
                ],
            )

    return overrides


def _collect_review_candidates(
    results: list[ProcessResult],
    config: BatchFallbackConfig,
) -> list[dict]:
    """Collect unresolved or suspicious results for manual review."""
    candidates: list[dict] = []
    seen_paths = set()

    for result in results:
        if result.verification_status == VerificationStatus.RESIDUAL:
            candidates.append(
                {
                    "input": result.input_path,
                    "reason": "residual_branding_detected",
                    "selected_profile": result.selected_profile.value,
                    "output_size": list(result.output_size),
                    "crop_reasons": result.crop_reasons,
                }
            )
            seen_paths.add(result.input_path)

    if not config.enabled:
        return candidates

    buckets = _group_results_for_batch_fallback(results, config)
    for bucket in buckets.values():
        bottom_crops = [
            result.original_size[1] - result.output_size[1]
            for result in bucket
            if _is_bottom_only_crop(result)
        ]
        if len(bottom_crops) < config.min_support:
            continue
        cropped_ratio = len(bottom_crops) / len(bucket)
        if cropped_ratio < config.min_cropped_ratio:
            continue
        consensus = int(round(statistics.median(bottom_crops)))
        if not config.min_crop_px <= consensus <= config.max_crop_px:
            continue

        for result in bucket:
            if result.input_path in seen_paths:
                continue
            if result.original_size != result.output_size:
                continue
            if result.branding_candidates:
                continue
            candidates.append(
                {
                    "input": result.input_path,
                    "reason": "batch_consensus_suspect",
                    "consensus_crop_px": consensus,
                    "selected_profile": result.selected_profile.value,
                    "output_size": list(result.output_size),
                    "crop_reasons": result.crop_reasons,
                }
            )
            seen_paths.add(result.input_path)

    return sorted(candidates, key=lambda item: item["input"])


def _group_results_for_batch_fallback(
    results: list[ProcessResult],
    config: BatchFallbackConfig,
) -> dict[tuple, list[ProcessResult]]:
    """Bucket results for consensus analysis."""
    buckets: dict[tuple, list[ProcessResult]] = {}
    for result in results:
        if config.scope == "global":
            key = ("global", result.original_size)
        else:
            key = (os.path.dirname(result.input_path), result.original_size)
        buckets.setdefault(key, []).append(result)
    return buckets


def _export_review_candidates(
    review_candidates: list[dict],
    input_items_by_path: dict[str, InputItem],
    results_by_input: dict[str, ProcessResult],
    review_dir: str,
) -> None:
    """Export originals and processed outputs for manual review."""
    originals_root = os.path.join(review_dir, "originals")
    processed_root = os.path.join(review_dir, "processed")
    os.makedirs(originals_root, exist_ok=True)
    os.makedirs(processed_root, exist_ok=True)

    for candidate in review_candidates:
        item = input_items_by_path[candidate["input"]]
        result = results_by_input.get(candidate["input"])
        scoped_relative_path = _source_scoped_relative_path(item)
        original_target = os.path.join(originals_root, scoped_relative_path)
        processed_target = os.path.join(processed_root, scoped_relative_path)
        os.makedirs(os.path.dirname(original_target), exist_ok=True)
        os.makedirs(os.path.dirname(processed_target), exist_ok=True)
        shutil.copy2(candidate["input"], original_target)

        processed_source = result.output_path if result and result.output_path else None
        if processed_source and os.path.exists(processed_source):
            shutil.copy2(processed_source, processed_target)
            continue
        shutil.copy2(candidate["input"], processed_target)

    manifest_path = os.path.join(review_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(review_candidates, handle, indent=2, ensure_ascii=False)


def _fallback_config_to_dict(config: BatchFallbackConfig) -> dict:
    return {
        "enabled": config.enabled,
        "scope": config.scope,
        "min_support": config.min_support,
        "min_cropped_ratio": config.min_cropped_ratio,
        "min_crop_px": config.min_crop_px,
        "max_crop_px": config.max_crop_px,
    }


def _adaptive_config_from_args(args) -> AdaptiveParallelismConfig:
    profile_defaults = {
        "conservative": {
            "cpu_low_water": 45.0,
            "cpu_high_water": 78.0,
            "memory_low_water": 72.0,
            "memory_high_water": 82.0,
            "read_low_mbps": 80.0,
            "read_high_mbps": 200.0,
            "write_low_mbps": 35.0,
            "write_high_mbps": 90.0,
        },
        "balanced": {
            "cpu_low_water": 55.0,
            "cpu_high_water": 88.0,
            "memory_low_water": 78.0,
            "memory_high_water": 88.0,
            "read_low_mbps": 150.0,
            "read_high_mbps": 450.0,
            "write_low_mbps": 60.0,
            "write_high_mbps": 220.0,
        },
        "aggressive": {
            "cpu_low_water": 65.0,
            "cpu_high_water": 92.0,
            "memory_low_water": 75.0,
            "memory_high_water": 90.0,
            "read_low_mbps": 220.0,
            "read_high_mbps": 600.0,
            "write_low_mbps": 90.0,
            "write_high_mbps": 260.0,
        },
    }[args.resource_profile]
    return AdaptiveParallelismConfig(
        enabled=args.adaptive_workers,
        min_workers=args.min_workers,
        max_workers=args.max_workers,
        poll_interval=args.resource_poll_interval,
        progress_interval=args.progress_interval,
        **profile_defaults,
    )


def _adaptive_config_to_dict(config: AdaptiveParallelismConfig) -> dict:
    return {
        "enabled": config.enabled,
        "min_workers": config.min_workers,
        "max_workers": config.max_workers,
        "poll_interval": config.poll_interval,
        "cpu_low_water": config.cpu_low_water,
        "cpu_high_water": config.cpu_high_water,
        "memory_low_water": config.memory_low_water,
        "memory_high_water": config.memory_high_water,
        "read_low_mbps": config.read_low_mbps,
        "read_high_mbps": config.read_high_mbps,
        "write_low_mbps": config.write_low_mbps,
        "write_high_mbps": config.write_high_mbps,
        "progress_interval": config.progress_interval,
    }


def _print_progress(
    completed: int,
    total: int,
    target_workers: int,
    snapshot: ResourceSnapshot,
) -> None:
    """Print a compact progress line for long-running batch jobs."""
    print(
        "[progress] {}/{} complete | target_workers={} | cpu={:.0f}% | mem={:.0f}% | read={:.1f}MB/s | write={:.1f}MB/s".format(
            completed,
            total,
            target_workers,
            snapshot.cpu_percent,
            snapshot.memory_percent,
            snapshot.read_mbps,
            snapshot.write_mbps,
        ),
        file=sys.stderr,
    )


def _is_bottom_only_crop(result: ProcessResult) -> bool:
    """Return True when a result only removed pixels from the bottom edge."""
    crop_box = result.crop_result.crop_box
    width, height = result.original_size
    return (
        crop_box[0] == 0
        and crop_box[1] == 0
        and crop_box[2] == width
        and crop_box[3] < height
    )


def _print_summary(summary: dict) -> None:
    """Print aggregate statistics."""
    print(
        "Processed {processed} image(s): {cropped} cropped, {verified_clean} verified clean, "
        "{residual_detected} residual, {failed} failed".format(**summary)
    )
    if summary["verification_skipped"]:
        print(f"Verification skipped for {summary['verification_skipped']} image(s)")


def _print_error(path: str, exc: Exception) -> None:
    """Print a human-readable processing error."""
    print(f"❌ {path}: {exc}", file=sys.stderr)


def _collect_input_paths(inputs: list[str]) -> list[InputItem]:
    """Expand file and directory inputs into a sorted image input list."""
    collected: list[InputItem] = []
    seen = set()

    for raw_path in inputs:
        path = os.path.abspath(raw_path)
        if os.path.isdir(path):
            entries = []
            for root, _, files in os.walk(path):
                for filename in sorted(files):
                    entry = os.path.join(root, filename)
                    if _is_supported_image(entry):
                        entries.append(entry)
            for entry in entries:
                if entry not in seen:
                    seen.add(entry)
                    collected.append(
                        InputItem(
                            path=entry,
                            source_root=path,
                            relative_path=os.path.relpath(entry, path),
                        )
                    )
            continue

        if not os.path.exists(path):
            raise FileNotFoundError(f"Input path does not exist: {raw_path}")
        if not os.path.isfile(path):
            raise ValueError(f"Unsupported input path: {raw_path}")
        if not _is_supported_image(path):
            raise ValueError(f"Unsupported image file type: {raw_path}")
        if path not in seen:
            seen.add(path)
            collected.append(
                InputItem(
                    path=path,
                    source_root=os.path.dirname(path) or ".",
                    relative_path=os.path.basename(path),
                )
            )

    return sorted(collected, key=lambda item: item.path)


def _build_output_path(
    input_item: InputItem,
    output_path: str | None = None,
    output_dir: str | None = None,
    in_place: bool = False,
) -> str | None:
    """Resolve the output path for a single image."""
    if output_path:
        return output_path
    if in_place:
        return input_item.path
    if not output_dir:
        return None

    stem, ext = os.path.splitext(input_item.relative_path)
    ext = ext or ".png"
    resolved = os.path.join(output_dir, f"{stem}_nowm{ext}")
    os.makedirs(os.path.dirname(resolved), exist_ok=True)
    return resolved


def _backup_original(item: InputItem, backup_dir: str) -> None:
    """Copy the original input file into a collision-safe backup location once."""
    backup_path = os.path.join(backup_dir, _source_scoped_relative_path(item))
    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
    if not os.path.exists(backup_path):
        shutil.copy2(item.path, backup_path)


def _source_scoped_relative_path(item: InputItem) -> str:
    """Prefix relative paths with a stable source-root token to avoid collisions."""
    relative_path = item.relative_path or os.path.basename(item.path)
    source_root = os.path.abspath(item.source_root)
    source_name = os.path.basename(os.path.normpath(source_root)) or "input"
    source_hash = hashlib.sha1(source_root.encode("utf-8")).hexdigest()[:8]
    return os.path.join(f"{source_name}-{source_hash}", relative_path)


def _is_supported_image(path: str) -> bool:
    """Return True if the path looks like a supported image file."""
    return os.path.splitext(path)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS


def _non_negative_int(raw_value: str) -> int:
    """Argparse validator for non-negative integers."""
    value = int(raw_value)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return value


def _positive_int(raw_value: str) -> int:
    """Argparse validator for positive integers."""
    value = int(raw_value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def _positive_float(raw_value: str) -> float:
    """Argparse validator for positive floats."""
    value = float(raw_value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive number")
    return value


def _bounded_float(lower: float, upper: float, label: str):
    """Create an argparse validator for bounded floats."""

    def _validator(raw_value: str) -> float:
        value = float(raw_value)
        if not lower <= value <= upper:
            raise argparse.ArgumentTypeError(
                f"{label} must be between {lower} and {upper}"
            )
        return value

    return _validator


if __name__ == "__main__":
    sys.exit(main())
