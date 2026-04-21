"""High-level API: detect + classify + crop in one call."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

from PIL import Image

from .detector import TextRegion, detect_text
from .cropper import (
    CropResult,
    CropStrategy,
    WatermarkCandidate,
    compute_crop,
    find_corner_watermarks,
)


@dataclass
class ProcessResult:
    """Complete result of watermark processing."""

    input_path: str
    output_path: Optional[str]
    original_size: Tuple[int, int]
    output_size: Tuple[int, int]
    text_regions: List[TextRegion]
    watermarks: List[WatermarkCandidate]
    crop_result: CropResult
    saved: bool = False


def process_image(
    image_path: str,
    output_path: Optional[str] = None,
    strategy: CropStrategy = CropStrategy.STRIP,
    corner_frac: float = 0.20,
    min_confidence: float = 0.3,
    margin: int = 10,
    max_crop_frac: float = 0.25,
    dry_run: bool = False,
) -> ProcessResult:
    """
    Full pipeline: detect text → find corner watermarks → crop → save.

    Args:
        image_path: Input image path.
        output_path: Output path (default: <input>_nowm.<ext>).
        strategy: Crop strategy (strip or corner).
        corner_frac: Fraction of image dimension to consider "corner".
        min_confidence: Minimum OCR confidence.
        margin: Extra margin pixels around watermark.
        max_crop_frac: Maximum fraction to crop from any edge (safety).
        dry_run: If True, don't save the cropped image.

    Returns:
        ProcessResult with full detection and crop details.
    """
    img = Image.open(image_path)
    img_w, img_h = img.size

    # Detect text
    text_regions = detect_text(image_path, min_confidence=min_confidence)

    # Find corner watermarks
    watermarks = find_corner_watermarks(text_regions, img_w, img_h, corner_frac)

    # Compute crop
    crop_result = compute_crop(
        watermarks, img_w, img_h, strategy, margin, max_crop_frac
    )

    # Save
    saved = False
    out_path = None
    if crop_result.needs_crop and not dry_run:
        ext = os.path.splitext(image_path)[1] or ".png"
        out_path = output_path or os.path.splitext(image_path)[0] + "_nowm" + ext
        cropped = img.crop(crop_result.crop_box)
        save_kwargs = {}
        if ext.lower() in (".jpg", ".jpeg"):
            save_kwargs["quality"] = 95
        cropped.save(out_path, **save_kwargs)
        saved = True

    return ProcessResult(
        input_path=image_path,
        output_path=out_path,
        original_size=(img_w, img_h),
        output_size=crop_result.output_size,
        text_regions=text_regions,
        watermarks=watermarks,
        crop_result=crop_result,
        saved=saved,
    )
