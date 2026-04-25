"""Branding classification and crop computation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Tuple

from .models import TextRegion

TOP_BRANDING_ZONE_FRACTION = 0.30
BOTTOM_BRANDING_ZONE_FRACTION = 0.25
EDGE_BRANDING_ZONE_FRACTION = 0.25
TOP_WIDE_BRANDING_THRESHOLD = 0.20
MIN_STRIP_RETAINED_AREA_RATIO = 0.75
COVER_TOP_MAX_CROP_FRACTION = 0.35
COVER_BOTTOM_MAX_CROP_FRACTION = 0.25
COVER_SIDE_MAX_CROP_FRACTION = 0.25

_BRANDING_PATTERNS = (
    ("copyright", re.compile(r"copyright", re.IGNORECASE)),
    ("rights", re.compile(r"all\s+rights\s+reserved", re.IGNORECASE)),
    ("brand", re.compile(r"xiuren(?:\.com)?", re.IGNORECASE)),
    ("url", re.compile(r"https?://\S*xiuren\.com/?", re.IGNORECASE)),
    ("issue_id", re.compile(r"\bxr[0-9a-z]{6,}\b", re.IGNORECASE)),
)


class Corner(str, Enum):
    TOP_LEFT = "top-left"
    TOP_RIGHT = "top-right"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_RIGHT = "bottom-right"


class CropProfile(str, Enum):
    AUTO = "auto"
    STRIP = "strip"
    COVER = "cover"
    CORNER = "corner"


CropStrategy = CropProfile


@dataclass
class BrandingCandidate:
    """A detected text region classified as removable branding text."""

    text: str
    confidence: float
    px_bbox: dict
    anchors: List[str]
    matched_rules: List[str]
    corners: List[Corner] = field(default_factory=list)

    def __repr__(self):
        anchors = ",".join(self.anchors)
        rules = ",".join(self.matched_rules)
        return (
            f"BrandingCandidate(text={self.text!r}, conf={self.confidence:.2f}, "
            f"anchors={anchors}, rules={rules}, px={self.px_bbox})"
        )


WatermarkCandidate = BrandingCandidate


@dataclass
class CropResult:
    """Result of crop computation."""

    crop_box: Tuple[int, int, int, int]
    strategy: CropProfile
    branding_candidates: List[BrandingCandidate]
    original_size: Tuple[int, int]
    output_size: Tuple[int, int]
    crop_reasons: List[str] = field(default_factory=list)

    @property
    def removed_px(self) -> Tuple[int, int]:
        return (
            self.original_size[0] - self.output_size[0],
            self.original_size[1] - self.output_size[1],
        )

    @property
    def needs_crop(self) -> bool:
        return self.output_size != self.original_size

    @property
    def watermarks(self) -> List[BrandingCandidate]:
        """Backward-compatible alias for older callers."""
        return self.branding_candidates


@dataclass
class StripEdgeProposal:
    """Aggregated strip-crop proposal for one image edge."""

    edge: str
    amount: int
    score: float = 0.0
    hits: int = 0
    rules: set[str] = field(default_factory=set)

    def register(self, amount: int, confidence: float, matched_rules: Iterable[str]) -> None:
        self.amount = max(self.amount, amount)
        self.score += confidence
        self.hits += 1
        self.rules.update(matched_rules)

    def efficiency(self, full_extent: int) -> float:
        crop_fraction = self.amount / full_extent if full_extent else 1.0
        return self.score / max(crop_fraction, 1e-6)


def matched_branding_rules(text: str) -> List[str]:
    """Return the set of branding rules matched by a text snippet."""
    matches = [name for name, pattern in _BRANDING_PATTERNS if pattern.search(text or "")]
    return list(dict.fromkeys(matches))


def find_branding_candidates(
    text_regions: List[TextRegion],
    img_w: int,
    img_h: int,
    top_frac: float = TOP_BRANDING_ZONE_FRACTION,
    bottom_frac: float = BOTTOM_BRANDING_ZONE_FRACTION,
    edge_frac: float = EDGE_BRANDING_ZONE_FRACTION,
) -> List[BrandingCandidate]:
    """Classify removable branding text based on content and broad edge/top zones."""
    candidates: List[BrandingCandidate] = []

    for region in text_regions:
        matched_rules = matched_branding_rules(region.text)
        if not matched_rules:
            continue

        px = region.to_pixel(img_w, img_h)
        anchors = _anchors_for_bbox(px, img_w, img_h, top_frac, bottom_frac, edge_frac)
        if not anchors:
            continue

        candidates.append(
            BrandingCandidate(
                text=region.text,
                confidence=region.confidence,
                px_bbox=px,
                anchors=anchors,
                matched_rules=matched_rules,
                corners=_corners_from_anchors(anchors),
            )
        )

    return candidates


def find_corner_watermarks(
    text_regions: List[TextRegion],
    img_w: int,
    img_h: int,
    corner_frac: float = 0.20,
) -> List[WatermarkCandidate]:
    """Legacy corner-only classification kept for backward compatibility."""
    if not 0.0 <= corner_frac <= 0.5:
        raise ValueError("corner_frac must be between 0.0 and 0.5")

    margin_x = corner_frac * img_w
    margin_y = corner_frac * img_h
    watermarks: List[WatermarkCandidate] = []

    for region in text_regions:
        px = region.to_pixel(img_w, img_h)
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
                BrandingCandidate(
                    text=region.text,
                    confidence=region.confidence,
                    px_bbox=px,
                    anchors=_anchors_from_corners(corners),
                    matched_rules=["corner-text"],
                    corners=corners,
                )
            )

    return watermarks


def should_use_cover_profile(
    branding_candidates: List[BrandingCandidate],
    img_w: int,
) -> bool:
    """Return True when a page looks like a heavily annotated cover/information page."""
    top_candidates = [candidate for candidate in branding_candidates if "top" in candidate.anchors]
    bottom_candidates = [
        candidate for candidate in branding_candidates if "bottom" in candidate.anchors
    ]

    if len(top_candidates) >= 2:
        return True
    if top_candidates and bottom_candidates:
        return True
    return any(
        candidate.px_bbox["w"] / img_w >= TOP_WIDE_BRANDING_THRESHOLD
        for candidate in top_candidates
    )


def compute_crop(
    branding_candidates: List[BrandingCandidate],
    img_w: int,
    img_h: int,
    strategy: CropProfile = CropProfile.STRIP,
    margin: int = 10,
    max_crop_frac: float = 0.25,
) -> CropResult:
    """Compute the optimal crop box for the selected branding candidates."""
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if not 0.0 <= max_crop_frac <= 1.0:
        raise ValueError("max_crop_frac must be between 0.0 and 1.0")
    if strategy == CropProfile.AUTO:
        raise ValueError("AUTO must be resolved before calling compute_crop")

    if not branding_candidates:
        return CropResult(
            crop_box=(0, 0, img_w, img_h),
            strategy=strategy,
            branding_candidates=[],
            original_size=(img_w, img_h),
            output_size=(img_w, img_h),
            crop_reasons=[],
        )

    if strategy == CropProfile.STRIP:
        crop_edges, reasons = _compute_strip_edges(branding_candidates, img_w, img_h, margin)
    elif strategy == CropProfile.COVER:
        crop_edges, reasons = _compute_cover_edges(branding_candidates, img_w, img_h, margin)
    else:
        crop_edges, reasons = _compute_corner_edges(branding_candidates, img_w, img_h, margin)

    crop_edges = _clamp_edges(crop_edges, img_w, img_h, strategy, max_crop_frac)
    box = (
        crop_edges["left"],
        crop_edges["top"],
        img_w - crop_edges["right"],
        img_h - crop_edges["bottom"],
    )
    out_w = max(0, box[2] - box[0])
    out_h = max(0, box[3] - box[1])

    return CropResult(
        crop_box=box,
        strategy=strategy,
        branding_candidates=branding_candidates,
        original_size=(img_w, img_h),
        output_size=(out_w, out_h),
        crop_reasons=reasons,
    )


def _compute_strip_edges(
    branding_candidates: List[BrandingCandidate],
    img_w: int,
    img_h: int,
    margin: int,
) -> Tuple[Dict[str, int], List[str]]:
    """Pick one best edge per axis to remove branding with limited content loss."""
    proposals: Dict[str, StripEdgeProposal] = {}

    for candidate in branding_candidates:
        for edge, amount in _strip_options(candidate, img_w, img_h, margin):
            proposal = proposals.setdefault(edge, StripEdgeProposal(edge=edge, amount=0))
            proposal.register(amount, candidate.confidence, candidate.matched_rules)

    selected: Dict[str, StripEdgeProposal] = {}
    for edge_names, full_extent in ((("top", "bottom"), img_h), (("left", "right"), img_w)):
        candidates = [proposals[name] for name in edge_names if name in proposals]
        if not candidates:
            continue
        selected_edge = max(
            candidates,
            key=lambda proposal: (
                proposal.efficiency(full_extent),
                proposal.score,
                proposal.hits,
                -proposal.amount,
            ),
        )
        selected[selected_edge.edge] = selected_edge

    while len(selected) > 1 and _retained_area_ratio(selected, img_w, img_h) < MIN_STRIP_RETAINED_AREA_RATIO:
        edge_to_drop = min(
            selected.values(),
            key=lambda proposal: (
                proposal.score,
                proposal.hits,
                proposal.efficiency(img_h if proposal.edge in ("top", "bottom") else img_w),
                -proposal.amount,
            ),
        )
        selected.pop(edge_to_drop.edge, None)

    crop_edges = _empty_edges()
    reasons: List[str] = []
    for edge, proposal in selected.items():
        crop_edges[edge] = proposal.amount
        reasons.append(f"{edge}:{','.join(sorted(proposal.rules))}")
    return crop_edges, reasons


def _compute_cover_edges(
    branding_candidates: List[BrandingCandidate],
    img_w: int,
    img_h: int,
    margin: int,
) -> Tuple[Dict[str, int], List[str]]:
    """Aggressively remove full branding zones on cover/info pages."""
    crop_edges = _empty_edges()
    reasons: List[str] = []

    top_candidates = [candidate for candidate in branding_candidates if "top" in candidate.anchors]
    bottom_candidates = [
        candidate for candidate in branding_candidates if "bottom" in candidate.anchors
    ]
    side_only_candidates = [
        candidate
        for candidate in branding_candidates
        if "top" not in candidate.anchors and "bottom" not in candidate.anchors
    ]

    if top_candidates:
        crop_edges["top"] = max(
            candidate.px_bbox["y"] + candidate.px_bbox["h"] + margin
            for candidate in top_candidates
        )
        reasons.append(
            "top:" + ",".join(sorted({rule for candidate in top_candidates for rule in candidate.matched_rules}))
        )

    if bottom_candidates:
        crop_edges["bottom"] = max(
            img_h - candidate.px_bbox["y"] + margin for candidate in bottom_candidates
        )
        reasons.append(
            "bottom:"
            + ",".join(sorted({rule for candidate in bottom_candidates for rule in candidate.matched_rules}))
        )

    for edge in ("left", "right"):
        edge_candidates = [
            candidate for candidate in side_only_candidates if edge in candidate.anchors
        ]
        if not edge_candidates:
            continue
        if edge == "left":
            crop_edges["left"] = max(
                candidate.px_bbox["x"] + candidate.px_bbox["w"] + margin
                for candidate in edge_candidates
            )
        else:
            crop_edges["right"] = max(
                img_w - candidate.px_bbox["x"] + margin for candidate in edge_candidates
            )
        reasons.append(
            f"{edge}:"
            + ",".join(sorted({rule for candidate in edge_candidates for rule in candidate.matched_rules}))
        )

    return crop_edges, reasons


def _compute_corner_edges(
    branding_candidates: List[BrandingCandidate],
    img_w: int,
    img_h: int,
    margin: int,
) -> Tuple[Dict[str, int], List[str]]:
    """Legacy corner crop that removes both strips at each detected corner."""
    crop_edges = _empty_edges()
    reasons: List[str] = []

    for candidate in branding_candidates:
        if not candidate.corners:
            continue
        for corner in candidate.corners:
            crop_edges = _apply_corner_crop(corner, candidate.px_bbox, img_w, img_h, margin, crop_edges)
            reasons.append(f"{corner.value}:{','.join(candidate.matched_rules)}")

    return crop_edges, reasons


def _strip_options(
    candidate: BrandingCandidate,
    img_w: int,
    img_h: int,
    margin: int,
) -> List[Tuple[str, int]]:
    """Return strip crop options for one branding candidate."""
    bbox = candidate.px_bbox
    if "top" in candidate.anchors and "bottom" not in candidate.anchors:
        return [("top", bbox["y"] + bbox["h"] + margin)]
    if "bottom" in candidate.anchors and "top" not in candidate.anchors:
        return [("bottom", img_h - bbox["y"] + margin)]

    options: List[Tuple[str, int]] = []
    if "left" in candidate.anchors:
        options.append(("left", bbox["x"] + bbox["w"] + margin))
    if "right" in candidate.anchors:
        options.append(("right", img_w - bbox["x"] + margin))
    return options


def _apply_corner_crop(
    corner: Corner,
    bbox: dict,
    img_w: int,
    img_h: int,
    margin: int,
    crop_edges: Dict[str, int],
) -> Dict[str, int]:
    """Apply legacy corner crop geometry to the current edge totals."""
    crop_edges = dict(crop_edges)
    if corner == Corner.TOP_LEFT:
        crop_edges["top"] = max(crop_edges["top"], bbox["y"] + bbox["h"] + margin)
        crop_edges["left"] = max(crop_edges["left"], bbox["x"] + bbox["w"] + margin)
    elif corner == Corner.TOP_RIGHT:
        crop_edges["top"] = max(crop_edges["top"], bbox["y"] + bbox["h"] + margin)
        crop_edges["right"] = max(crop_edges["right"], img_w - bbox["x"] + margin)
    elif corner == Corner.BOTTOM_LEFT:
        crop_edges["bottom"] = max(crop_edges["bottom"], img_h - bbox["y"] + margin)
        crop_edges["left"] = max(crop_edges["left"], bbox["x"] + bbox["w"] + margin)
    elif corner == Corner.BOTTOM_RIGHT:
        crop_edges["bottom"] = max(crop_edges["bottom"], img_h - bbox["y"] + margin)
        crop_edges["right"] = max(crop_edges["right"], img_w - bbox["x"] + margin)
    return crop_edges


def _clamp_edges(
    crop_edges: Dict[str, int],
    img_w: int,
    img_h: int,
    strategy: CropProfile,
    max_crop_frac: float,
) -> Dict[str, int]:
    """Clamp crop sizes to safe per-edge maxima."""
    crop_edges = dict(crop_edges)
    if strategy == CropProfile.COVER:
        edge_limits = {
            "top": int(img_h * max(max_crop_frac, COVER_TOP_MAX_CROP_FRACTION)),
            "bottom": int(img_h * max(max_crop_frac, COVER_BOTTOM_MAX_CROP_FRACTION)),
            "left": int(img_w * max(max_crop_frac, COVER_SIDE_MAX_CROP_FRACTION)),
            "right": int(img_w * max(max_crop_frac, COVER_SIDE_MAX_CROP_FRACTION)),
        }
    else:
        edge_limits = {
            "top": int(img_h * max_crop_frac),
            "bottom": int(img_h * max_crop_frac),
            "left": int(img_w * max_crop_frac),
            "right": int(img_w * max_crop_frac),
        }

    for edge, limit in edge_limits.items():
        crop_edges[edge] = min(crop_edges[edge], limit)
    return crop_edges


def _anchors_for_bbox(
    bbox: dict,
    img_w: int,
    img_h: int,
    top_frac: float,
    bottom_frac: float,
    edge_frac: float,
) -> List[str]:
    """Return broad placement anchors for a pixel bbox."""
    anchors: List[str] = []
    right_edge = img_w - (bbox["x"] + bbox["w"])
    bottom_edge = img_h - (bbox["y"] + bbox["h"])

    if bbox["y"] <= img_h * top_frac:
        anchors.append("top")
    if bottom_edge <= img_h * bottom_frac:
        anchors.append("bottom")
    if bbox["x"] <= img_w * edge_frac:
        anchors.append("left")
    if right_edge <= img_w * edge_frac:
        anchors.append("right")

    return anchors


def _corners_from_anchors(anchors: List[str]) -> List[Corner]:
    """Derive corner labels from anchors when possible."""
    corners: List[Corner] = []
    anchor_set = set(anchors)
    if {"top", "left"} <= anchor_set:
        corners.append(Corner.TOP_LEFT)
    if {"top", "right"} <= anchor_set:
        corners.append(Corner.TOP_RIGHT)
    if {"bottom", "left"} <= anchor_set:
        corners.append(Corner.BOTTOM_LEFT)
    if {"bottom", "right"} <= anchor_set:
        corners.append(Corner.BOTTOM_RIGHT)
    return corners


def _anchors_from_corners(corners: List[Corner]) -> List[str]:
    """Backward-compatible anchor derivation for legacy corner watermarks."""
    anchors: set[str] = set()
    for corner in corners:
        if corner in (Corner.TOP_LEFT, Corner.TOP_RIGHT):
            anchors.add("top")
        if corner in (Corner.BOTTOM_LEFT, Corner.BOTTOM_RIGHT):
            anchors.add("bottom")
        if corner in (Corner.TOP_LEFT, Corner.BOTTOM_LEFT):
            anchors.add("left")
        if corner in (Corner.TOP_RIGHT, Corner.BOTTOM_RIGHT):
            anchors.add("right")
    return sorted(anchors)


def _retained_area_ratio(selected: Dict[str, StripEdgeProposal], img_w: int, img_h: int) -> float:
    """Compute retained image area for a strip-crop selection."""
    crop_top = selected.get("top", StripEdgeProposal("top", 0)).amount
    crop_bottom = selected.get("bottom", StripEdgeProposal("bottom", 0)).amount
    crop_left = selected.get("left", StripEdgeProposal("left", 0)).amount
    crop_right = selected.get("right", StripEdgeProposal("right", 0)).amount
    out_w = max(0, img_w - crop_left - crop_right)
    out_h = max(0, img_h - crop_top - crop_bottom)
    if img_w == 0 or img_h == 0:
        return 0.0
    return (out_w * out_h) / (img_w * img_h)


def _empty_edges() -> Dict[str, int]:
    return {"top": 0, "bottom": 0, "left": 0, "right": 0}
