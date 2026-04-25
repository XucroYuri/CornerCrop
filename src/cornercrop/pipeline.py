"""High-level API: detect, classify, crop, verify, and save in one call."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from PIL import Image

from .cropper import (
    BOTTOM_BRANDING_ZONE_FRACTION,
    EDGE_BRANDING_ZONE_FRACTION,
    TOP_BRANDING_ZONE_FRACTION,
    BrandingCandidate,
    CropProfile,
    CropResult,
    CropStrategy,
    compute_crop,
    find_branding_candidates,
    find_corner_watermarks,
    matched_branding_rules,
    should_use_cover_profile,
)
from .detector import detect_text
from .models import TextRegion

VERIFICATION_TOP_FRACTION = 0.35
VERIFICATION_BOTTOM_FRACTION = 0.25
VERIFICATION_SIDE_FRACTION = 0.25
VERIFICATION_MIN_CONFIDENCE = 0.1
FOCUSED_DETECTION_MIN_CONFIDENCE = 0.1


class VerificationStatus(str):
    NOT_RUN = "not_run"
    CLEAN = "clean"
    RESIDUAL = "residual_branding_detected"


@dataclass
class ResidualTextMatch:
    """A branding text match found during post-processing verification."""

    source: str
    text: str
    confidence: float
    matched_rules: List[str]
    px_bbox: dict


@dataclass
class ProcessResult:
    """Complete result of branding text processing."""

    input_path: str
    output_path: Optional[str]
    original_size: Tuple[int, int]
    output_size: Tuple[int, int]
    text_regions: List[TextRegion]
    branding_candidates: List[BrandingCandidate]
    crop_result: CropResult
    selected_profile: CropProfile
    verification_status: str = VerificationStatus.NOT_RUN
    residual_text_matches: List[ResidualTextMatch] = field(default_factory=list)
    saved: bool = False

    @property
    def watermarks(self) -> List[BrandingCandidate]:
        """Backward-compatible alias for older callers."""
        return self.branding_candidates

    @property
    def crop_reasons(self) -> List[str]:
        return self.crop_result.crop_reasons


@dataclass
class CropOverride:
    """An explicit crop override computed outside the OCR pipeline."""

    crop_box: Tuple[int, int, int, int]
    selected_profile: CropProfile
    crop_reasons: List[str]


def process_image(
    image_path: str,
    output_path: Optional[str] = None,
    strategy: CropStrategy = CropProfile.AUTO,
    corner_frac: float = 0.20,
    min_confidence: float = 0.3,
    margin: int = 10,
    max_crop_frac: float = 0.25,
    dry_run: bool = False,
    verify: bool = False,
) -> ProcessResult:
    """
    Full pipeline: detect text → find branding candidates → crop → verify → save.

    Args:
        image_path: Input image path.
        output_path: Output path (default: <input>_nowm.<ext>).
        strategy: Crop profile (auto, strip, cover, or corner).
        corner_frac: Legacy corner fraction used by corner mode.
        min_confidence: Minimum OCR confidence.
        margin: Extra margin pixels around detected branding text.
        max_crop_frac: Maximum fraction to crop from any edge.
        dry_run: If True, don't save the processed image.
        verify: If True, OCR-scan the processed output for residual branding text.

    Returns:
        ProcessResult with crop and verification details.
    """
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if not 0.0 <= corner_frac <= 0.5:
        raise ValueError("corner_frac must be between 0.0 and 0.5")
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0.0 and 1.0")
    if not 0.0 <= max_crop_frac <= 1.0:
        raise ValueError("max_crop_frac must be between 0.0 and 1.0")

    with Image.open(image_path) as img:
        img_w, img_h = img.size
        text_regions = _collect_text_regions(
            image_path,
            img,
            min_confidence=min_confidence,
        )

        requested_profile = CropProfile(strategy)
        if requested_profile == CropProfile.CORNER:
            branding_candidates = find_corner_watermarks(
                text_regions, img_w, img_h, corner_frac=corner_frac
            )
            selected_profile = CropProfile.CORNER
        else:
            branding_candidates = find_branding_candidates(
                text_regions,
                img_w,
                img_h,
                top_frac=TOP_BRANDING_ZONE_FRACTION,
                bottom_frac=BOTTOM_BRANDING_ZONE_FRACTION,
                edge_frac=EDGE_BRANDING_ZONE_FRACTION,
            )
            if requested_profile == CropProfile.AUTO:
                selected_profile = (
                    CropProfile.COVER
                    if should_use_cover_profile(branding_candidates, img_w)
                    else CropProfile.STRIP
                )
            else:
                selected_profile = requested_profile

        crop_result = compute_crop(
            branding_candidates,
            img_w,
            img_h,
            strategy=selected_profile,
            margin=margin,
            max_crop_frac=max_crop_frac,
        )

        final_image = _build_processed_image(img, crop_result)
        saved = False
        out_path = None
        if crop_result.needs_crop and not dry_run:
            ext = os.path.splitext(image_path)[1] or ".png"
            out_path = output_path or os.path.splitext(image_path)[0] + "_nowm" + ext
            save_kwargs = {}
            if ext.lower() in (".jpg", ".jpeg"):
                save_kwargs["quality"] = 95
            _save_image_atomic(final_image, out_path, **save_kwargs)
            saved = True

        verification_status = VerificationStatus.NOT_RUN
        residual_text_matches: List[ResidualTextMatch] = []
        if verify:
            verification_status, residual_text_matches = _verify_processed_image(
                final_image,
                min_confidence=min(min_confidence, VERIFICATION_MIN_CONFIDENCE),
            )

        final_image.close()

    return ProcessResult(
        input_path=image_path,
        output_path=out_path,
        original_size=(img_w, img_h),
        output_size=crop_result.output_size,
        text_regions=text_regions,
        branding_candidates=branding_candidates,
        crop_result=crop_result,
        selected_profile=selected_profile,
        verification_status=verification_status,
        residual_text_matches=residual_text_matches,
        saved=saved,
    )


def apply_crop_override(
    base_result: ProcessResult,
    override: CropOverride,
    output_path: Optional[str] = None,
    dry_run: bool = False,
    verify: bool = False,
) -> ProcessResult:
    """Apply an explicit crop override to an already-inspected image."""
    crop_box = override.crop_box
    original_size = base_result.original_size
    output_size = (
        max(0, crop_box[2] - crop_box[0]),
        max(0, crop_box[3] - crop_box[1]),
    )
    crop_result = CropResult(
        crop_box=crop_box,
        strategy=override.selected_profile,
        branding_candidates=base_result.branding_candidates,
        original_size=original_size,
        output_size=output_size,
        crop_reasons=override.crop_reasons,
    )

    with Image.open(base_result.input_path) as img:
        final_image = _build_processed_image(img, crop_result)
        saved = False
        out_path = None
        if crop_result.needs_crop and not dry_run:
            ext = os.path.splitext(base_result.input_path)[1] or ".png"
            out_path = output_path or os.path.splitext(base_result.input_path)[0] + "_nowm" + ext
            save_kwargs = {}
            if ext.lower() in (".jpg", ".jpeg"):
                save_kwargs["quality"] = 95
            _save_image_atomic(final_image, out_path, **save_kwargs)
            saved = True

        verification_status = VerificationStatus.NOT_RUN
        residual_text_matches: List[ResidualTextMatch] = []
        if verify:
            verification_status, residual_text_matches = _verify_processed_image(
                final_image,
                min_confidence=VERIFICATION_MIN_CONFIDENCE,
            )
        final_image.close()

    return ProcessResult(
        input_path=base_result.input_path,
        output_path=out_path,
        original_size=original_size,
        output_size=output_size,
        text_regions=base_result.text_regions,
        branding_candidates=base_result.branding_candidates,
        crop_result=crop_result,
        selected_profile=override.selected_profile,
        verification_status=verification_status,
        residual_text_matches=residual_text_matches,
        saved=saved,
    )


def _build_processed_image(img: Image.Image, crop_result: CropResult) -> Image.Image:
    """Return a detached image representing the processed output."""
    if crop_result.needs_crop:
        return img.crop(crop_result.crop_box).copy()
    return img.copy()


def _verify_processed_image(
    image: Image.Image,
    min_confidence: float,
) -> Tuple[str, List[ResidualTextMatch]]:
    """Run full-image and focused region OCR to find residual branding text."""
    matches: List[ResidualTextMatch] = []
    seen = set()

    for source, region_image, offset in _verification_regions(image):
        regions = _detect_text_in_image(region_image, min_confidence=min_confidence)
        for region in regions:
            matched_rules = matched_branding_rules(region.text)
            if not matched_rules:
                continue
            bbox = region.to_pixel(*region_image.size)
            bbox["x"] += offset[0]
            bbox["y"] += offset[1]
            key = (
                source,
                region.text.strip().lower(),
                tuple(sorted(matched_rules)),
                bbox["x"],
                bbox["y"],
                bbox["w"],
                bbox["h"],
            )
            if key in seen:
                continue
            seen.add(key)
            matches.append(
                ResidualTextMatch(
                    source=source,
                    text=region.text,
                    confidence=region.confidence,
                    matched_rules=matched_rules,
                    px_bbox=bbox,
                )
            )

    status = VerificationStatus.CLEAN if not matches else VerificationStatus.RESIDUAL
    return status, matches


def _collect_text_regions(
    image_path: str,
    image: Image.Image,
    min_confidence: float,
) -> List[TextRegion]:
    """Collect OCR results from the full image plus focused edge scans."""
    full_size = image.size
    focused_min_confidence = min(min_confidence, FOCUSED_DETECTION_MIN_CONFIDENCE)
    seen = set()
    regions: List[TextRegion] = []

    for source, region_image, offset in _verification_regions(image):
        if source == "full":
            detected = detect_text(image_path, min_confidence=min_confidence)
        else:
            detected = _detect_text_in_image(region_image, min_confidence=focused_min_confidence)

        for region in detected:
            absolute = _region_to_full_coordinates(region, region_image.size, full_size, offset)
            px_bbox = absolute.to_pixel(*full_size)
            key = (
                absolute.text.strip().lower(),
                px_bbox["x"],
                px_bbox["y"],
                px_bbox["w"],
                px_bbox["h"],
            )
            if key in seen:
                continue
            seen.add(key)
            regions.append(absolute)

    return regions


def _verification_regions(image: Image.Image):
    """Yield OCR verification regions with absolute offsets."""
    width, height = image.size
    yield "full", image, (0, 0)

    top_height = max(1, int(height * VERIFICATION_TOP_FRACTION))
    bottom_height = max(1, int(height * VERIFICATION_BOTTOM_FRACTION))
    side_width = max(1, int(width * VERIFICATION_SIDE_FRACTION))

    yield "top", image.crop((0, 0, width, top_height)), (0, 0)
    yield "bottom", image.crop((0, height - bottom_height, width, height)), (0, height - bottom_height)
    yield "left", image.crop((0, 0, side_width, height)), (0, 0)
    yield "right", image.crop((width - side_width, 0, width, height)), (width - side_width, 0)


def _region_to_full_coordinates(
    region: TextRegion,
    region_size: tuple[int, int],
    full_size: tuple[int, int],
    offset: tuple[int, int],
) -> TextRegion:
    """Convert a region-local OCR result into full-image normalized coordinates."""
    region_w, region_h = region_size
    full_w, full_h = full_size
    local_px = region.to_pixel(region_w, region_h)
    abs_x = local_px["x"] + offset[0]
    abs_y = local_px["y"] + offset[1]
    return TextRegion(
        text=region.text,
        confidence=region.confidence,
        bbox_x=abs_x / full_w,
        bbox_y=(full_h - abs_y - local_px["h"]) / full_h,
        bbox_w=local_px["w"] / full_w,
        bbox_h=local_px["h"] / full_h,
    )


def _detect_text_in_image(image: Image.Image, min_confidence: float) -> List[TextRegion]:
    """Run OCR on an in-memory image by writing it to a temporary file."""
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            temp_path = handle.name
        image.save(temp_path)
        return detect_text(temp_path, min_confidence=min_confidence)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def _save_image_atomic(image: Image.Image, output_path: str, **save_kwargs) -> None:
    """Save an image atomically so in-place writes do not corrupt the source."""
    output_dir = os.path.dirname(output_path) or "."
    suffix = os.path.splitext(output_path)[1] or ".png"
    fd, temp_path = tempfile.mkstemp(prefix=".cornercrop-", suffix=suffix, dir=output_dir)
    os.close(fd)
    try:
        image.save(temp_path, **save_kwargs)
        os.replace(temp_path, output_path)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise
