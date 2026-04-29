"""Unit tests for CornerCrop."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PIL import Image, UnidentifiedImageError

from cornercrop.batch import AdaptiveParallelismConfig, process_batch
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
from cornercrop.library_runner import (
    ArchiveReason,
    ImageAction,
    JobDatabase,
    LibraryRunConfig,
    _process_image_path,
    _process_album,
    build_image_decision,
    iter_album_dirs,
    safe_archive_path,
)
from cornercrop.non_corner_recovery import (
    RecoveryConfig,
    album_dir_for_archive_image,
    iter_non_corner_archive_images,
    recover_image,
    safe_recovered_path,
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

    def test_matched_branding_rules_detects_common_album_brands(self):
        samples = [
            "XINGYAN星颜社 VOL.025",
            "HuaYang花漾 Vol.270",
            "MFStar模范学院 Vol.383",
            "YiTuYu艺图语 Vol.8874",
        ]

        for sample in samples:
            with self.subTest(sample=sample):
                self.assertIn("brand", matched_branding_rules(sample))

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


class TestLibraryRunnerPolicy(unittest.TestCase):
    def test_iter_album_dirs_finds_image_dirs_and_ignores_archives(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = os.path.join(tmpdir, "Model")
            album_dir = os.path.join(model_dir, "Album")
            archive_dir = os.path.join(album_dir, "_cornercrop_archive", "non_corner_watermark")
            empty_dir = os.path.join(model_dir, "Empty")
            os.makedirs(archive_dir, exist_ok=True)
            os.makedirs(empty_dir, exist_ok=True)
            _create_test_image(os.path.join(album_dir, "001.jpg"))
            _create_test_image(os.path.join(archive_dir, "archived.jpg"))

            albums = list(iter_album_dirs(model_dir))

        self.assertEqual(albums, [album_dir])

    def test_build_image_decision_skips_when_no_watermark_candidates(self):
        result = _result(
            "sample.jpg",
            branding_candidates=[],
            original_size=(1000, 1000),
            output_size=(1000, 1000),
        )

        decision = build_image_decision(result, max_removed_area_ratio=0.30)

        self.assertEqual(decision.action, ImageAction.SKIP)

    def test_build_image_decision_crops_corner_watermark_under_limit(self):
        candidate = _candidate(
            "XIUREN",
            anchors=["bottom", "right"],
            bbox={"x": 900, "y": 940, "w": 80, "h": 30},
            corners=[Corner.BOTTOM_RIGHT],
        )
        result = _result(
            "sample.jpg",
            selected_profile=CropProfile.CORNER,
            branding_candidates=[candidate],
            original_size=(1000, 1000),
            output_size=(900, 900),
        )

        decision = build_image_decision(result, max_removed_area_ratio=0.30)

        self.assertEqual(decision.action, ImageAction.CROP)
        self.assertAlmostEqual(decision.removed_area_ratio, 0.19)

    def test_build_image_decision_archives_non_corner_watermark(self):
        candidate = _candidate(
            "XIUREN",
            anchors=["top"],
            bbox={"x": 420, "y": 10, "w": 160, "h": 30},
            corners=[],
        )
        result = _result(
            "sample.jpg",
            branding_candidates=[candidate],
            original_size=(1000, 1000),
            output_size=(1000, 940),
        )

        decision = build_image_decision(result, max_removed_area_ratio=0.30)

        self.assertEqual(decision.action, ImageAction.ARCHIVE)
        self.assertEqual(decision.archive_reason, ArchiveReason.NON_CORNER_WATERMARK)

    def test_build_image_decision_archives_excessive_removed_area(self):
        candidate = _candidate(
            "XIUREN",
            anchors=["top", "left"],
            bbox={"x": 0, "y": 0, "w": 320, "h": 320},
            corners=[Corner.TOP_LEFT],
        )
        result = _result(
            "sample.jpg",
            branding_candidates=[candidate],
            original_size=(1000, 1000),
            output_size=(800, 800),
        )

        decision = build_image_decision(result, max_removed_area_ratio=0.30)

        self.assertEqual(decision.action, ImageAction.ARCHIVE)
        self.assertEqual(decision.archive_reason, ArchiveReason.EXCESSIVE_CROP_AREA)

    def test_safe_archive_path_keeps_conflicting_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_dir = os.path.join(tmpdir, "archive")
            os.makedirs(archive_dir, exist_ok=True)
            existing = os.path.join(archive_dir, "001.jpg")
            _create_test_image(existing)

            resolved = safe_archive_path(os.path.join(tmpdir, "001.jpg"), archive_dir)

        self.assertTrue(resolved.endswith("001__cornercrop_1.jpg"))

    def test_process_image_path_archives_unreadable_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            album_dir = os.path.join(tmpdir, "Album")
            os.makedirs(album_dir, exist_ok=True)
            image_path = os.path.join(album_dir, "broken.jpg")
            with open(image_path, "wb") as handle:
                handle.write(b"not an image")
            db = JobDatabase(os.path.join(tmpdir, "state", "job.sqlite3"))
            config = LibraryRunConfig(root=tmpdir, state_dir=os.path.join(tmpdir, "state"))
            try:
                with mock.patch(
                    "cornercrop.library_runner.inspect_image",
                    side_effect=UnidentifiedImageError("cannot identify image file"),
                ):
                    action = _process_image_path(album_dir, image_path, config, db)
                row = db._connection.execute(
                    "SELECT action, reason, output_path, error FROM images WHERE path = ?",
                    (image_path,),
                ).fetchone()
                archive_exists = os.path.exists(row["output_path"])
            finally:
                db.close()

            self.assertEqual(action, "archived")
            self.assertEqual(row["action"], "archived")
            self.assertEqual(row["reason"], ArchiveReason.UNREADABLE_IMAGE.value)
            self.assertIsNone(row["error"])
            self.assertFalse(os.path.exists(image_path))
            self.assertTrue(archive_exists)

    def test_process_album_records_album_failure_without_stopping_batch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            album_dir = os.path.join(tmpdir, "Album")
            os.makedirs(album_dir, exist_ok=True)
            _create_test_image(os.path.join(album_dir, "001.jpg"))
            db = JobDatabase(os.path.join(tmpdir, "state", "job.sqlite3"))
            config = LibraryRunConfig(root=tmpdir, state_dir=os.path.join(tmpdir, "state"), dry_run=True)
            try:
                with mock.patch(
                    "cornercrop.library_runner._process_image_path",
                    side_effect=RuntimeError("album worker boom"),
                ):
                    counts = _process_album(album_dir, config, db)
                row = db._connection.execute(
                    "SELECT status, error FROM albums WHERE path = ?",
                    (album_dir,),
                ).fetchone()
            finally:
                db.close()

        self.assertEqual(counts["processed"], 0)
        self.assertEqual(row["status"], "failed")
        self.assertIn("album worker boom", row["error"])

    def test_process_album_marks_partial_stop_without_claiming_done(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            album_dir = os.path.join(tmpdir, "Album")
            os.makedirs(album_dir, exist_ok=True)
            _create_test_image(os.path.join(album_dir, "001.jpg"))
            stop_file = os.path.join(tmpdir, "STOP")
            open(stop_file, "w", encoding="utf-8").close()
            db = JobDatabase(os.path.join(tmpdir, "state", "job.sqlite3"))
            config = LibraryRunConfig(
                root=tmpdir,
                state_dir=os.path.join(tmpdir, "state"),
                dry_run=True,
                stop_file=stop_file,
            )
            try:
                counts = _process_album(album_dir, config, db)
                row = db._connection.execute(
                    "SELECT status, processed, total FROM albums WHERE path = ?",
                    (album_dir,),
                ).fetchone()
            finally:
                db.close()

        self.assertEqual(counts["processed"], 0)
        self.assertEqual(row["status"], "stopped")
        self.assertEqual(row["processed"], 0)
        self.assertEqual(row["total"], 1)


class TestNonCornerRecovery(unittest.TestCase):
    def test_iter_non_corner_archive_images_finds_only_reason_dir_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            album_dir = os.path.join(tmpdir, "Model", "Album")
            reason_dir = os.path.join(album_dir, "_cornercrop_archive", "non_corner_watermark")
            other_archive = os.path.join(album_dir, "_cornercrop_archive", "excessive_crop_area")
            os.makedirs(reason_dir, exist_ok=True)
            os.makedirs(other_archive, exist_ok=True)
            wanted = _create_test_image(os.path.join(reason_dir, "001.jpg"))
            _create_test_image(os.path.join(other_archive, "002.jpg"))
            _create_test_image(os.path.join(album_dir, "003.jpg"))

            images = list(iter_non_corner_archive_images(tmpdir))

        self.assertEqual(images, [os.path.abspath(wanted)])

    def test_recover_image_saves_cropped_output_to_album_and_removes_archive_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            album_dir = os.path.join(tmpdir, "Album")
            reason_dir = os.path.join(album_dir, "_cornercrop_archive", "non_corner_watermark")
            os.makedirs(reason_dir, exist_ok=True)
            source_path = _create_test_image(os.path.join(reason_dir, "001.jpg"), 800, 600)
            config = RecoveryConfig(root=tmpdir, state_dir=os.path.join(tmpdir, "state"))
            regions = [TextRegion("XIUREN", 0.9, 0.35, 0.92, 0.2, 0.05)]

            with mock.patch("cornercrop.non_corner_recovery._collect_text_regions", return_value=regions):
                with mock.patch(
                    "cornercrop.non_corner_recovery._verify_processed_image",
                    return_value=(VerificationStatus.CLEAN, []),
                ):
                    result = recover_image(source_path, config)

            self.assertEqual(result.action, "recovered")
            self.assertEqual(album_dir_for_archive_image(source_path), album_dir)
            self.assertFalse(os.path.exists(source_path))
            self.assertTrue(os.path.exists(result.output_path))
            with Image.open(result.output_path) as recovered:
                self.assertLess(recovered.size[1], 600)

    def test_recover_image_keeps_source_when_verification_finds_residual_branding(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            album_dir = os.path.join(tmpdir, "Album")
            reason_dir = os.path.join(album_dir, "_cornercrop_archive", "non_corner_watermark")
            os.makedirs(reason_dir, exist_ok=True)
            source_path = _create_test_image(os.path.join(reason_dir, "001.jpg"), 800, 600)
            config = RecoveryConfig(root=tmpdir, state_dir=os.path.join(tmpdir, "state"))
            regions = [TextRegion("XIUREN", 0.9, 0.35, 0.92, 0.2, 0.05)]
            residual = ResidualTextMatch(
                source="top",
                text="XIUREN",
                confidence=0.9,
                matched_rules=["brand"],
                px_bbox={"x": 10, "y": 10, "w": 80, "h": 30},
            )

            with mock.patch("cornercrop.non_corner_recovery._collect_text_regions", return_value=regions):
                with mock.patch(
                    "cornercrop.non_corner_recovery._verify_processed_image",
                    return_value=(VerificationStatus.RESIDUAL, [residual]),
                ):
                    result = recover_image(source_path, config)

            self.assertEqual(result.action, "kept")
            self.assertEqual(result.reason, "residual_watermark_after_second_pass_crop")
            self.assertTrue(os.path.exists(source_path))
            self.assertIsNone(result.output_path)

    def test_safe_recovered_path_keeps_conflicting_album_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            album_dir = os.path.join(tmpdir, "Album")
            reason_dir = os.path.join(album_dir, "_cornercrop_archive", "non_corner_watermark")
            os.makedirs(reason_dir, exist_ok=True)
            _create_test_image(os.path.join(album_dir, "001.jpg"))
            source_path = _create_test_image(os.path.join(reason_dir, "001.jpg"))

            resolved = safe_recovered_path(source_path, album_dir)

        self.assertTrue(resolved.endswith("001__cornercrop_recovered_1.jpg"))

    def test_recover_image_prefers_lower_area_clean_crop_across_profiles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            album_dir = os.path.join(tmpdir, "Album")
            reason_dir = os.path.join(album_dir, "_cornercrop_archive", "non_corner_watermark")
            os.makedirs(reason_dir, exist_ok=True)
            source_path = _create_test_image(os.path.join(reason_dir, "001.jpg"), 800, 600)
            config = RecoveryConfig(
                root=tmpdir,
                state_dir=os.path.join(tmpdir, "state"),
                dry_run=True,
            )
            regions = [
                TextRegion("XIUREN", 0.9, 0.30, 0.92, 0.25, 0.05),
                TextRegion("Copyright xiuren.com", 0.9, 0.30, 0.01, 0.25, 0.05),
            ]

            with mock.patch("cornercrop.non_corner_recovery._collect_text_regions", return_value=regions):
                with mock.patch(
                    "cornercrop.non_corner_recovery._verify_processed_image",
                    return_value=(VerificationStatus.CLEAN, []),
                ):
                    result = recover_image(source_path, config)

            self.assertEqual(result.action, "would_recover")
            self.assertEqual(result.selected_profile, CropProfile.STRIP.value)
            self.assertTrue(os.path.exists(source_path))


class TestAdaptiveBatch(unittest.TestCase):
    def test_process_batch_emits_heartbeat_before_completion(self):
        callbacks = []

        def worker(item):
            time.sleep(0.06)
            return item

        results = process_batch(
            [1],
            worker,
            AdaptiveParallelismConfig(
                min_workers=1,
                max_workers=1,
                poll_interval=0.01,
                progress_interval=100,
                heartbeat_interval=0.01,
            ),
            progress_callback=lambda completed, total, target, snapshot: callbacks.append(
                (completed, total, target)
            ),
        )

        self.assertEqual(results, [1])
        self.assertIn((0, 1, 1), callbacks)
        self.assertIn((1, 1, 1), callbacks)


if __name__ == "__main__":
    unittest.main()
