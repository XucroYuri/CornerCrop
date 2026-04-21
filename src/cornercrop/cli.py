"""Command-line interface for CornerCrop."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .pipeline import process_image
from .cropper import CropStrategy


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="cornercrop",
        description="Detect and crop corner watermarks from images using macOS Vision OCR.",
    )
    parser.add_argument("input", nargs="+", help="Input image path(s)")
    parser.add_argument("--output", "-o", help="Output image path (single image mode)")
    parser.add_argument(
        "--strategy",
        choices=["strip", "corner"],
        default="strip",
        help="Crop strategy: strip (remove shorter edge) or corner (remove both edges). Default: strip",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=10,
        help="Extra margin in pixels around detected watermark. Default: 10",
    )
    parser.add_argument(
        "--corner",
        type=float,
        default=0.20,
        help="Corner region fraction (0-0.5). Default: 0.20",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.3,
        help="Minimum OCR confidence threshold. Default: 0.3",
    )
    parser.add_argument(
        "--max-crop",
        type=float,
        default=0.25,
        help="Maximum fraction of any edge to crop (safety). Default: 0.25",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect only, don't save cropped images",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    args = parser.parse_args(argv)

    strategy = CropStrategy.STRIP if args.strategy == "strip" else CropStrategy.CORNER

    all_results = []
    for path in args.input:
        result = process_image(
            image_path=path,
            output_path=args.output if len(args.input) == 1 else None,
            strategy=strategy,
            corner_frac=args.corner,
            min_confidence=args.min_confidence,
            margin=args.margin,
            max_crop_frac=args.max_crop,
            dry_run=args.dry_run,
        )
        all_results.append(result)

        if not args.json:
            _print_human(result)

    if args.json:
        print(json.dumps([_result_to_dict(r) for r in all_results], indent=2, ensure_ascii=False))

    # Exit code: 0 if all images processed, 1 if any had issues
    return 0


def _print_human(result):
    """Print human-readable result."""
    w, h = result.original_size
    print(f"📷 {result.input_path} ({w}×{h})")
    print(f"🔍 {len(result.text_regions)} text region(s), {len(result.watermarks)} corner watermark(s)")

    for wm in result.watermarks:
        corners_str = ", ".join(c.value for c in wm.corners)
        print(f"  ⚠️  [{corners_str}] \"{wm.text}\" (conf={wm.confidence:.2f}) px={wm.px_bbox}")

    if result.watermarks:
        cr = result.crop_result
        ow, oh = result.output_size
        rw, rh = cr.removed_px
        print(f"✂️  Crop ({cr.strategy.value}): {cr.crop_box}")
        print(f"   Output: {ow}×{oh} (removed {rw}×{rh}px)")
    else:
        print("✅ No corner watermarks — no cropping needed")

    if result.saved:
        print(f"💾 Saved: {result.output_path}")
    elif result.watermarks and not result.saved:
        print("🏃 Dry run — no file saved")

    print()


def _result_to_dict(result):
    """Convert ProcessResult to JSON-serializable dict."""
    return {
        "input": result.input_path,
        "output": result.output_path,
        "original_size": list(result.original_size),
        "output_size": list(result.output_size),
        "text_regions": len(result.text_regions),
        "watermarks": [
            {
                "corners": [c.value for c in wm.corners],
                "text": wm.text,
                "confidence": wm.confidence,
                "px_bbox": wm.px_bbox,
            }
            for wm in result.watermarks
        ],
        "crop_box": list(result.crop_result.crop_box),
        "crop_strategy": result.crop_result.strategy.value,
        "saved": result.saved,
    }


if __name__ == "__main__":
    sys.exit(main())
