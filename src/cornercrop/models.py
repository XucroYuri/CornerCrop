"""Shared data models that do not depend on Vision/PyObjC."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TextRegion:
    """A detected text region with a normalized bounding box."""

    text: str
    confidence: float
    # Normalized bbox (0-1), Vision convention: origin bottom-left
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float

    def to_pixel(self, img_w: int, img_h: int) -> dict[str, int]:
        """Convert normalized bbox to pixel coords using top-left origin."""
        px_x = int(self.bbox_x * img_w)
        px_y = int((1.0 - self.bbox_y - self.bbox_h) * img_h)
        px_w = int(self.bbox_w * img_w)
        px_h = int(self.bbox_h * img_h)
        return {"x": px_x, "y": px_y, "w": px_w, "h": px_h}
