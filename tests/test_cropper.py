"""Unit tests for CornerCrop."""

import json
import os
import sys
import tempfile
import unittest

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PIL import Image, ImageDraw

from cornercrop.detector import TextRegion
from cornercrop.cropper import (
    Corner,
    CropStrategy,
    WatermarkCandidate,
    compute_crop,
    find_corner_watermarks,
)
from cornercrop.pipeline import process_image


def _create_test_image(path, width=800, height=600, watermarks=None):
    """Create a test image with optional watermark text in corners."""
    img = Image.new("RGB", (width, height), color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    draw.rectangle([200, 150, 600, 450], fill=(100, 150, 200))

    if watermarks:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except Exception:
            font = ImageFont.load_default()
        for pos, text in watermarks:
            draw.text(pos, text, fill=(180, 180, 180), font=font)

    img.save(path)
    return path


class TestCropperLogic(unittest.TestCase):
    """Test crop computation without needing Vision OCR."""

    def test_no_watermarks(self):
        result = compute_crop([], 800, 600)
        self.assertFalse(result.needs_crop)
        self.assertEqual(result.crop_box, (0, 0, 800, 600))

    def test_strip_bottom_right(self):
        wm = WatermarkCandidate(
            text="©Test",
            confidence=1.0,
            corners=[Corner.BOTTOM_RIGHT],
            px_bbox={"x": 700, "y": 550, "w": 80, "h": 30},
        )
        result = compute_crop([wm], 800, 600, CropStrategy.STRIP, margin=10)
        self.assertTrue(result.needs_crop)
        # Bottom strip: 600-550+30+10 = 90px from bottom → less than right strip
        self.assertEqual(result.output_size[0], 800)  # width unchanged
        self.assertLess(result.output_size[1], 600)    # height reduced

    def test_corner_bottom_right(self):
        wm = WatermarkCandidate(
            text="©Test",
            confidence=1.0,
            corners=[Corner.BOTTOM_RIGHT],
            px_bbox={"x": 700, "y": 550, "w": 80, "h": 30},
        )
        result = compute_crop([wm], 800, 600, CropStrategy.CORNER, margin=10)
        self.assertTrue(result.needs_crop)
        # Both bottom and right should be cropped
        self.assertLess(result.output_size[0], 800)
        self.assertLess(result.output_size[1], 600)

    def test_safety_clamp(self):
        """Crop should never exceed max_crop_frac."""
        wm = WatermarkCandidate(
            text="BigWatermark",
            confidence=1.0,
            corners=[Corner.BOTTOM_RIGHT],
            px_bbox={"x": 100, "y": 100, "w": 600, "h": 400},
        )
        result = compute_crop([wm], 800, 600, max_crop_frac=0.25)
        # Should be clamped to 25% even though watermark is huge
        max_h = int(600 * 0.25)
        max_w = int(800 * 0.25)
        self.assertLessEqual(600 - result.output_size[1], max_h)
        self.assertLessEqual(800 - result.output_size[0], max_w)

    def test_find_corner_watermarks_classification(self):
        """Test that only corner-region text is classified as watermark."""
        # Create text regions manually
        center_text = TextRegion(
            text="Center",
            confidence=0.9,
            bbox_x=0.4, bbox_y=0.45, bbox_w=0.2, bbox_h=0.05,
        )
        corner_text = TextRegion(
            text="©Corner",
            confidence=0.9,
            bbox_x=0.85, bbox_y=0.02, bbox_w=0.1, bbox_h=0.03,
        )
        watermarks = find_corner_watermarks([center_text, corner_text], 800, 600)
        self.assertEqual(len(watermarks), 1)
        self.assertIn(Corner.BOTTOM_RIGHT, watermarks[0].corners)


class TestTextRegion(unittest.TestCase):
    """Test TextRegion coordinate conversion."""

    def test_to_pixel(self):
        tr = TextRegion(text="test", confidence=1.0, bbox_x=0.5, bbox_y=0.5, bbox_w=0.2, bbox_h=0.1)
        px = tr.to_pixel(800, 600)
        # Vision: origin bottom-left, so y=0.5 → top-left pixel y = (1-0.5-0.1)*600 = 240
        self.assertEqual(px["x"], 400)
        self.assertEqual(px["y"], 240)
        self.assertEqual(px["w"], 160)
        self.assertEqual(px["h"], 60)


class TestCLI(unittest.TestCase):
    """Test CLI argument parsing."""

    def test_version(self):
        from cornercrop.cli import main
        with self.assertRaises(SystemExit) as ctx:
            main(["--version"])
        self.assertEqual(ctx.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
