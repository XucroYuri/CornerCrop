"""Watermark classification and crop computation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple

from .detector import TextRegion


class Corner(str, Enum):
    TOP_LEFT = "top-left"
    TOP_RIGHT = "top-right"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_RIGHT = "bottom-right"


class CropStrategy(str, Enum):
    STRIP = "strip"     # Remove the shorter edge strip
    CORNER = "corner"   # Remove both edge strips at the corner


@dataclass
class WatermarkCandidate:
    """A text region classified as a corner watermark."""

    text: str
    confidence: float
    corners: List[Corner]
    px_bbox: dict  # {"x", "y", "w", "h"} in pixels

    def __repr__(self):
        return (
            f"Watermark({self.corners}, \"{self.text}\", "
            f"conf={self.confidence:.2f}, px={self.px_bbox})"
        )


@dataclass
class CropResult:
    """Result of crop computation."""

    crop_box: Tuple[int, int, int, int]  # (left, top, right, bottom)
    strategy: CropStrategy
    watermarks: List[WatermarkCandidate]
    original_size: Tuple[int, int]
    output_size: Tuple[int, int]

    @property
    def removed_px(self) -> Tuple[int, int]:
        w_removed = self.original_size[0] - self.output_size[0]
        h_removed = self.original_size[1] - self.output_size[1]
        return (w_removed, h_removed)

    @property
    def needs_crop(self) -> bool:
        return self.output_size != self.original_size


def find_corner_watermarks(
    text_regions: List[TextRegion],
    img_w: int,
    img_h: int,
    corner_frac: float = 0.20,
) -> List[WatermarkCandidate]:
    """
    Classify text regions in image corners as watermark candidates.

    Args:
        text_regions: Detected text regions.
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        corner_frac: Fraction of each dimension considered "corner" (0-0.5).

    Returns:
        List of WatermarkCandidate objects.
    """
    margin_x = corner_frac * img_w
    margin_y = corner_frac * img_h
    watermarks: List[WatermarkCandidate] = []

    for r in text_regions:
        px = r.to_pixel(img_w, img_h)
        right_edge = img_w - (px["x"] + px["w"])
        bottom_edge = img_h - (px["y"] + px["h"])

        corners: List[Corner] = []
        if px["x"] < margin_x and px["y"] < margin_y:
            corners.append(Corner.TOP_LEFT)
        if right_edge < margin_x and px["y"] < margin_y:
            corners.append(Corner.TOP_RIGHT)
        if px["x"] < margin_x and bottom_edge < margin_y:
            corners.append(Corner.BOTTOM_LEFT)
        if right_edge < margin_x and bottom_edge < margin_y:
            corners.append(Corner.BOTTOM_RIGHT)

        if corners:
            watermarks.append(
                WatermarkCandidate(
                    text=r.text,
                    confidence=r.confidence,
                    corners=corners,
                    px_bbox=px,
                )
            )

    return watermarks


def compute_crop(
    watermarks: List[WatermarkCandidate],
    img_w: int,
    img_h: int,
    strategy: CropStrategy = CropStrategy.STRIP,
    margin: int = 10,
    max_crop_frac: float = 0.25,
) -> CropResult:
    """
    Compute the optimal crop box to remove corner watermarks.

    Args:
        watermarks: List of detected corner watermark candidates.
        img_w: Original image width.
        img_h: Original image height.
        strategy: Crop strategy (strip or corner).
        margin: Extra pixels to remove around watermark bounds.
        max_crop_frac: Maximum fraction of each dimension to crop (safety).

    Returns:
        CropResult with crop box and metadata.
    """
    if not watermarks:
        return CropResult(
            crop_box=(0, 0, img_w, img_h),
            strategy=strategy,
            watermarks=[],
            original_size=(img_w, img_h),
            output_size=(img_w, img_h),
        )

    crop_top = 0
    crop_bottom = 0
    crop_left = 0
    crop_right = 0

    for wm in watermarks:
        bb = wm.px_bbox
        for corner in wm.corners:
            if strategy == CropStrategy.STRIP:
                _apply_strip_crop(
                    corner, bb, img_w, img_h, margin,
                    crop_top, crop_bottom, crop_left, crop_right,
                    # Use mutable refs via list trick
                    result := [crop_top, crop_bottom, crop_left, crop_right],
                )
                crop_top, crop_bottom, crop_left, crop_right = result
            else:
                _apply_corner_crop(
                    corner, bb, img_w, img_h, margin,
                    result := [crop_top, crop_bottom, crop_left, crop_right],
                )
                crop_top, crop_bottom, crop_left, crop_right = result

    # Safety clamp
    max_h = int(img_h * max_crop_frac)
    max_w = int(img_w * max_crop_frac)
    crop_top = min(crop_top, max_h)
    crop_bottom = min(crop_bottom, max_h)
    crop_left = min(crop_left, max_w)
    crop_right = min(crop_right, max_w)

    box = (crop_left, crop_top, img_w - crop_right, img_h - crop_bottom)
    out_w = box[2] - box[0]
    out_h = box[3] - box[1]

    return CropResult(
        crop_box=box,
        strategy=strategy,
        watermarks=watermarks,
        original_size=(img_w, img_h),
        output_size=(out_w, out_h),
    )


def _apply_strip_crop(corner, bb, img_w, img_h, margin, _refs_unused, result):
    """Apply strip crop strategy — choose the shorter strip per corner."""
    crop_top, crop_bottom, crop_left, crop_right = result

    if corner == Corner.TOP_LEFT:
        strip_h = bb["y"] + bb["h"] + margin
        strip_w = bb["x"] + bb["w"] + margin
        if strip_h <= strip_w:
            crop_top = max(crop_top, strip_h)
        else:
            crop_left = max(crop_left, strip_w)

    elif corner == Corner.TOP_RIGHT:
        strip_h = bb["y"] + bb["h"] + margin
        strip_w = img_w - bb["x"] + margin
        if strip_h <= strip_w:
            crop_top = max(crop_top, strip_h)
        else:
            crop_right = max(crop_right, strip_w)

    elif corner == Corner.BOTTOM_LEFT:
        strip_h = img_h - bb["y"] + margin
        strip_w = bb["x"] + bb["w"] + margin
        if strip_h <= strip_w:
            crop_bottom = max(crop_bottom, strip_h)
        else:
            crop_left = max(crop_left, strip_w)

    elif corner == Corner.BOTTOM_RIGHT:
        strip_h = img_h - bb["y"] + margin
        strip_w = img_w - bb["x"] + margin
        if strip_h <= strip_w:
            crop_bottom = max(crop_bottom, strip_h)
        else:
            crop_right = max(crop_right, strip_w)

    result[:] = [crop_top, crop_bottom, crop_left, crop_right]


def _apply_corner_crop(corner, bb, img_w, img_h, margin, result):
    """Apply corner crop strategy — remove both strips at each corner."""
    crop_top, crop_bottom, crop_left, crop_right = result

    if corner == Corner.TOP_LEFT:
        crop_top = max(crop_top, bb["y"] + bb["h"] + margin)
        crop_left = max(crop_left, bb["x"] + bb["w"] + margin)

    elif corner == Corner.TOP_RIGHT:
        crop_top = max(crop_top, bb["y"] + bb["h"] + margin)
        crop_right = max(crop_right, img_w - bb["x"] + margin)

    elif corner == Corner.BOTTOM_LEFT:
        crop_bottom = max(crop_bottom, img_h - bb["y"] + margin)
        crop_left = max(crop_left, bb["x"] + bb["w"] + margin)

    elif corner == Corner.BOTTOM_RIGHT:
        crop_bottom = max(crop_bottom, img_h - bb["y"] + margin)
        crop_right = max(crop_right, img_w - bb["x"] + margin)

    result[:] = [crop_top, crop_bottom, crop_left, crop_right]
