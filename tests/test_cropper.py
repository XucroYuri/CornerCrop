"""Unit tests for CornerCrop."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PIL import Image

from cornercrop.cli import main
from cornercrop.cropper import (
    BrandingCandidate,
    Corner,
    CropProfile,
    CropResult,
    compute_crop,
    find_branding_candidates,
    matched_branding_rules,
    should_use_cover_profile,
)
from cornercrop.models import TextRegion
from cornercrop.pipeline import (
    ProcessResult,
    ResidualTextMatch,
    VerificationStatus,
    process_image,
)


def _create_test_image(path: str, width: int = 800, height: int = 600) -> str:
    img = Image.new("RGB", (width, height), color=(200, 200, 200))
    img.save(path)
    return path


def _candidate(
    text: str,
    *,
    anchors: list[str],
    bbox: dict,
    rules: list[str] | None = None,
    confidence: float = 1.0,
    corners: list[Corner] | None = None,
) -> BrandingCandidate:
    return BrandingCandidate(
        text=text,
        confidence=confidence,
        px_bbox=bbox,
        anchors=anchors,
        matched_rules=rules or ["brand"],
        corners=corners or [],
    )


def _result(
    input_path: str,
    *,
    selected_profile: CropProfile = CropProfile.AUTO,
    verification_status: str = VerificationStatus.CLEAN,
    residual_text_matches: list[ResidualTextMatch] | None = None,
    branding_candidates: list[BrandingCandidate] | None = None,
    original_size: tuple[int, int] = (800, 600),
    output_size: tuple[int, int] = (800, 550),
    saved: bool = False,
    output_path: str | None = None,
) -> ProcessResult:
    crop_result = CropResult(
        crop_box=(0, 0, output_size[0], output_size[1]),
        strategy=selected_profile,
        branding_candidates=branding_candidates or [],
        original_size=original_size,
        output_size=output_size,
        crop_reasons=["bottom:copyright"] if output_size != original_size else [],
    )
    return ProcessResult(
        input_path=input_path,
        output_path=output_path,
        original_size=original_size,
        output_size=output_size,
        text_regions=[],
        branding_candidates=branding_candidates or [],
        crop_result=crop_result,
        selected_profile=selected_profile,
        verification_status=verification_status,
        residual_text_matches=residual_text_matches or [],
        saved=saved,
    )


class TestBrandingRules(unittest.TestCase):
    def test_matched_branding_rules_detects_keywords_and_issue_id(self):
        rules = matched_branding_rules("Copyright @2024 xiuren.com XR20240327N08296")
        self.assertIn("copyright", rules)
        self.assertIn("brand", rules)
        self.assertIn("issue_id", rules)

    def test_find_branding_candidates_uses_content_and_position(self):
        regions = [
            TextRegion("XIUREN", 0.9, 0.35, 0.92, 0.2, 0.05),
            TextRegion("Copyright @2024 xiuren.com All Rights Reserved", 1.0, 0.01, 0.01, 0.5, 0.03),
            TextRegion("Unrelated", 0.9, 0.4, 0.5, 0.1, 0.05),
        ]
        candidates = find_branding_candidates(regions, 1200, 1800)
        self.assertEqual(len(candidates), 2)
        self.assertIn("top", candidates[0].anchors)
        self.assertIn("bottom", candidates[1].anchors)

    def test_cover_profile_detection_triggers_on_top_and_bottom_branding(self):
        candidates = [
            _candidate("XIUREN", anchors=["top"], bbox={"x": 100, "y": 20, "w": 400, "h": 80}),
            _candidate(
                "Copyright @2024 xiuren.com All Rights Reserved",
                anchors=["bottom", "left"],
                bbox={"x": 10, "y": 1750, "w": 600, "h": 30},
                rules=["copyright", "brand", "rights"],
            ),
        ]
        self.assertTrue(should_use_cover_profile(candidates, 1200))


class TestCropProfiles(unittest.TestCase):
    def test_strip_crop_prefers_bottom_for_bottom_branding(self):
        candidates = [
            _candidate(
                "Copyright @2024 xiuren.com All Rights Reserved",
                anchors=["bottom", "left", "right"],
                bbox={"x": 10, "y": 550, "w": 500, "h": 30},
                rules=["copyright", "brand", "rights"],
            )
        ]
        result = compute_crop(candidates, 800, 600, CropProfile.STRIP, margin=10)
        self.assertTrue(result.needs_crop)
        self.assertEqual(result.crop_box[0], 0)
        self.assertEqual(result.crop_box[2], 800)
        self.assertLess(result.output_size[1], 600)

    def test_cover_crop_removes_top_and_bottom_branding(self):
        candidates = [
            _candidate("XIUREN", anchors=["top"], bbox={"x": 100, "y": 20, "w": 500, "h": 100}),
            _candidate(
                "Copyright @2024 xiuren.com All Rights Reserved",
                anchors=["bottom", "left"],
                bbox={"x": 10, "y": 1750, "w": 700, "h": 30},
                rules=["copyright", "brand", "rights"],
            ),
        ]
        result = compute_crop(candidates, 1200, 1800, CropProfile.COVER, margin=10)
        self.assertTrue(result.needs_crop)
        self.assertGreater(result.crop_box[1], 0)
        self.assertLess(result.crop_box[3], 1800)


class TestPipeline(unittest.TestCase):
    def test_process_image_auto_selects_cover_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = _create_test_image(os.path.join(tmpdir, "sample.jpg"), 1200, 1800)
            detected = [
                TextRegion("XIUREN", 0.9, 0.3, 0.92, 0.25, 0.05),
                TextRegion(
                    "Copyright @2024 xiuren.com All Rights Reserved",
                    1.0,
                    0.01,
                    0.01,
                    0.5,
                    0.03,
                ),
            ]
            with mock.patch("cornercrop.pipeline.detect_text", return_value=detected):
                result = process_image(image_path, strategy=CropProfile.AUTO, dry_run=True, verify=False)

        self.assertEqual(result.selected_profile, CropProfile.COVER)
        self.assertTrue(result.crop_result.needs_crop)
        self.assertEqual(result.verification_status, VerificationStatus.NOT_RUN)

    def test_process_image_verification_flags_residual_branding(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = _create_test_image(os.path.join(tmpdir, "sample.jpg"))
            with mock.patch("cornercrop.pipeline.detect_text", return_value=[]):
                with mock.patch(
                    "cornercrop.pipeline._verify_processed_image",
                    return_value=(
                        VerificationStatus.RESIDUAL,
                        [
                            ResidualTextMatch(
                                source="top",
                                text="XIUREN",
                                confidence=0.9,
                                matched_rules=["brand"],
                                px_bbox={"x": 10, "y": 10, "w": 100, "h": 30},
                            )
                        ],
                    ),
                ):
                    result = process_image(image_path, dry_run=True, verify=True)

        self.assertEqual(result.verification_status, VerificationStatus.RESIDUAL)
        self.assertTrue(result.residual_text_matches)


class TestCLI(unittest.TestCase):
    def test_directory_input_and_report_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = os.path.join(tmpdir, "report.json")
            first = _create_test_image(os.path.join(tmpdir, "001.jpg"))
            second = _create_test_image(os.path.join(tmpdir, "002.jpg"))
            results = [
                _result(first, verification_status=VerificationStatus.CLEAN),
                _result(
                    second,
                    verification_status=VerificationStatus.RESIDUAL,
                    residual_text_matches=[
                        ResidualTextMatch(
                            source="top",
                            text="XIUREN",
                            confidence=0.9,
                            matched_rules=["brand"],
                            px_bbox={"x": 10, "y": 10, "w": 100, "h": 30},
                        )
                    ],
                ),
            ]
            stdout = io.StringIO()
            with mock.patch("cornercrop.cli.process_image", side_effect=results):
                with redirect_stdout(stdout):
                    exit_code = main([tmpdir, "--report-json", report_path, "--fail-on-residual"])

            with open(report_path, "r", encoding="utf-8") as handle:
                report = json.load(handle)

        self.assertEqual(exit_code, 1)
        self.assertEqual(report["summary"]["processed"], 2)
        self.assertEqual(report["summary"]["residual_detected"], 1)
        self.assertIn("Processed 2 image(s): 2 cropped, 1 verified clean, 1 residual, 0 failed", stdout.getvalue())

    def test_in_place_backup_copies_original_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = _create_test_image(os.path.join(tmpdir, "001.jpg"))
            backup_dir = os.path.join(tmpdir, "backup")
            result = _result(
                image_path,
                selected_profile=CropProfile.STRIP,
                output_path=image_path,
                saved=True,
            )

            with mock.patch("cornercrop.cli.process_image", return_value=result):
                exit_code = main([image_path, "--in-place", "--backup-dir", backup_dir])

            backed_up = []
            for root, _, files in os.walk(backup_dir):
                for filename in files:
                    backed_up.append(os.path.join(root, filename))

            self.assertEqual(len(backed_up), 1)
            self.assertTrue(backed_up[0].endswith(os.path.join("001.jpg")))

        self.assertEqual(exit_code, 0)

    def test_in_place_backup_keeps_same_basenames_separate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first_dir = os.path.join(tmpdir, "album-a")
            second_dir = os.path.join(tmpdir, "album-b")
            os.makedirs(first_dir, exist_ok=True)
            os.makedirs(second_dir, exist_ok=True)

            first = _create_test_image(os.path.join(first_dir, "001.jpg"))
            second = _create_test_image(os.path.join(second_dir, "001.jpg"))
            backup_dir = os.path.join(tmpdir, "backup")
            results = [
                _result(first, selected_profile=CropProfile.STRIP, output_path=first, saved=True),
                _result(second, selected_profile=CropProfile.STRIP, output_path=second, saved=True),
            ]

            with mock.patch("cornercrop.cli.process_image", side_effect=results):
                exit_code = main([first, second, "--in-place", "--backup-dir", backup_dir])

            backed_up = []
            for root, _, files in os.walk(backup_dir):
                for filename in files:
                    backed_up.append(os.path.relpath(os.path.join(root, filename), backup_dir))

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(backed_up), 2)
        self.assertNotEqual(backed_up[0], backed_up[1])

    def test_review_candidates_export_uses_processed_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = _create_test_image(os.path.join(tmpdir, "001.jpg"))
            processed_path = os.path.join(tmpdir, "001_nowm.jpg")
            processed_image = Image.new("RGB", (640, 480), color=(10, 20, 30))
            processed_image.save(processed_path)
            processed_image.close()

            result = _result(
                image_path,
                selected_profile=CropProfile.STRIP,
                verification_status=VerificationStatus.RESIDUAL,
                original_size=(800, 600),
                output_size=(640, 480),
                saved=True,
                output_path=processed_path,
            )
            review_dir = os.path.join(tmpdir, "review")

            with mock.patch("cornercrop.cli.process_image", return_value=result):
                exit_code = main([image_path, "--review-candidates-dir", review_dir])

            processed_exports = []
            for root, _, files in os.walk(os.path.join(review_dir, "processed")):
                for filename in files:
                    processed_exports.append(os.path.join(root, filename))

            self.assertEqual(len(processed_exports), 1)
            with Image.open(processed_exports[0]) as exported:
                self.assertEqual(exported.size, (640, 480))

        self.assertEqual(exit_code, 0)

    def test_json_output_includes_verification_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = _create_test_image(os.path.join(tmpdir, "001.jpg"))
            result = _result(
                image_path,
                selected_profile=CropProfile.COVER,
                verification_status=VerificationStatus.CLEAN,
            )
            stdout = io.StringIO()
            with mock.patch("cornercrop.cli.process_image", return_value=result):
                with redirect_stdout(stdout):
                    exit_code = main([image_path, "--json"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload[0]["selected_profile"], "cover")
        self.assertEqual(payload[0]["verification_status"], "clean")

    def test_batch_consensus_fallback_crops_uncaught_low_contrast_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = [_create_test_image(os.path.join(tmpdir, f"{idx:03d}.jpg"), 1200, 1799) for idx in range(6)]
            results = [
                _result(
                    path,
                    selected_profile=CropProfile.STRIP,
                    original_size=(1200, 1799),
                    output_size=(1200, 1739),
                    verification_status=VerificationStatus.CLEAN,
                )
                for path in paths[:5]
            ]
            results.append(
                _result(
                    paths[5],
                    selected_profile=CropProfile.STRIP,
                    original_size=(1200, 1799),
                    output_size=(1200, 1799),
                    verification_status=VerificationStatus.CLEAN,
                )
            )

            stdout = io.StringIO()
            with mock.patch("cornercrop.cli.process_image", side_effect=results):
                with mock.patch(
                    "cornercrop.cli.apply_crop_override",
                    return_value=_result(
                        paths[5],
                        selected_profile=CropProfile.STRIP,
                        original_size=(1200, 1799),
                        output_size=(1200, 1739),
                        verification_status=VerificationStatus.CLEAN,
                    ),
                ) as override_mock:
                    with redirect_stdout(stdout):
                        exit_code = main([tmpdir, "--dry-run"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(override_mock.call_count, 1)
        self.assertIn("Fallback crop applied", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
