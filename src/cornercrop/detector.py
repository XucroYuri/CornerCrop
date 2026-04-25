"""Vision.framework OCR backend for text region detection on macOS."""

from __future__ import annotations

import os
import sys
from typing import List

from .models import TextRegion


def detect_text(image_path: str, min_confidence: float = 0.3) -> List[TextRegion]:
    """
    Detect all text regions in an image using macOS Vision.framework OCR.

    Args:
        image_path: Path to the image file.
        min_confidence: Minimum confidence threshold (0-1).

    Returns:
        List of TextRegion objects.
    """
    try:
        from Foundation import NSURL, NSMutableDictionary
        from Vision import (
            VNImageRequestHandler,
            VNRecognizeTextRequest,
            VNRequestTextRecognitionLevelAccurate,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Vision.framework bindings are unavailable. Install CornerCrop's "
            "PyObjC dependencies on macOS before running OCR."
        ) from exc

    url = NSURL.fileURLWithPath_(os.path.abspath(image_path))
    handler = VNImageRequestHandler.alloc().initWithURL_options_(
        url, NSMutableDictionary.dictionary()
    )

    results: List[TextRegion] = []

    def _callback(request, error):
        if error:
            print(f"Vision error: {error.localizedDescription()}", file=sys.stderr)
            return
        for obs in request.results():
            bbox = obs.boundingBox()
            candidates = obs.topCandidates_(1)
            if candidates and len(candidates) > 0:
                candidate = candidates[0]
                text = candidate.string() if hasattr(candidate, "string") else str(candidate)
                confidence = float(
                    candidate.confidence() if hasattr(candidate, "confidence") else 0.0
                )
            else:
                continue

            if confidence >= min_confidence:
                results.append(
                    TextRegion(
                        text=text,
                        confidence=confidence,
                        bbox_x=float(bbox.origin.x),
                        bbox_y=float(bbox.origin.y),
                        bbox_w=float(bbox.size.width),
                        bbox_h=float(bbox.size.height),
                    )
                )

    request = VNRecognizeTextRequest.alloc().initWithCompletionHandler_(_callback)
    request.setRecognitionLevel_(VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(True)

    handler.performRequests_error_([request], None)
    return results
