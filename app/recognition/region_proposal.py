"""
app.recognition.region_proposal — Candidate Visual Region Extractor
===================================================================

Extracts candidate regions of interest (ROI) from video frames for targeted
object recognition:
1. Full-frame holistic candidate.
2. Center-focus candidate (user holding an object toward camera).
3. Salient foreground contour proposals using OpenCV edge and contour analysis.

Decouples visual region proposal from embedding model inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CandidateRegion:
    """A candidate visual region extracted from a frame.

    Attributes
    ----------
    bbox : tuple of (int, int, int, int)
        Bounding box (x, y, width, height) relative to source frame.
    crop : np.ndarray
        BGR image crop corresponding to the bounding box.
    region_type : str
        Source identifier: ``'full_frame'``, ``'center_focus'``, or ``'salient_contour'``.
    area : int
        Area in pixels (w * h).
    """

    bbox: Tuple[int, int, int, int]
    crop: np.ndarray
    region_type: str
    area: int


class CandidateRegionExtractor:
    """Extracts candidate visual regions of interest from video frames.

    Parameters
    ----------
    include_full_frame : bool, optional
        Whether to include the uncropped full frame as candidate. Default True.
    include_center_crop : bool, optional
        Whether to include a central focus crop where held objects typically reside. Default True.
    center_crop_ratio : float, optional
        Fraction of width and height for central crop. Default 0.65.
    include_contours : bool, optional
        Whether to extract salient foreground contours using OpenCV. Default True.
    min_contour_area : int, optional
        Minimum pixel area for contour bounding boxes to filter noise. Default 4000.
    max_contour_proposals : int, optional
        Maximum number of contour proposals to retain per frame. Default 2.
    """

    def __init__(
        self,
        include_full_frame: bool = True,
        include_center_crop: bool = True,
        center_crop_ratio: float = 0.65,
        include_contours: bool = True,
        min_contour_area: int = 4000,
        max_contour_proposals: int = 2,
    ) -> None:
        self.include_full_frame = include_full_frame
        self.include_center_crop = include_center_crop
        self.center_crop_ratio = min(max(center_crop_ratio, 0.2), 0.95)
        self.include_contours = include_contours
        self.min_contour_area = min_contour_area
        self.max_contour_proposals = max_contour_proposals

    def extract(self, frame: np.ndarray) -> List[CandidateRegion]:
        """Extract candidate regions from a BGR video frame.

        Parameters
        ----------
        frame : np.ndarray
            OpenCV BGR image array (H, W, 3).

        Returns
        -------
        list of CandidateRegion
            Candidate visual crops extracted from the frame.
        """
        if frame is None or frame.size == 0 or len(frame.shape) < 2:
            return []

        h, w = frame.shape[:2]
        regions: List[CandidateRegion] = []

        # 1. Full-frame candidate
        if self.include_full_frame:
            regions.append(
                CandidateRegion(
                    bbox=(0, 0, w, h),
                    crop=frame,
                    region_type="full_frame",
                    area=w * h,
                )
            )

        # 2. Central focus candidate
        if self.include_center_crop and w > 40 and h > 40:
            cw = int(w * self.center_crop_ratio)
            ch = int(h * self.center_crop_ratio)
            cx = max(0, (w - cw) // 2)
            cy = max(0, (h - ch) // 2)

            center_crop = frame[cy : cy + ch, cx : cx + cw]
            if center_crop.size > 0:
                regions.append(
                    CandidateRegion(
                        bbox=(cx, cy, cw, ch),
                        crop=center_crop,
                        region_type="center_focus",
                        area=cw * ch,
                    )
                )

        # 3. Salient foreground contours
        if self.include_contours and w > 80 and h > 80:
            contour_regions = self._extract_contours(frame, w, h)
            regions.extend(contour_regions)

        return regions

    def _extract_contours(
        self, frame: np.ndarray, frame_w: int, frame_h: int
    ) -> List[CandidateRegion]:
        """Detect salient foreground contours using edge and gradient analysis."""
        proposals: List[CandidateRegion] = []
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            edged = cv2.Canny(blurred, 40, 120)

            # Dilate edges to close gaps
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dilated = cv2.dilate(edged, kernel, iterations=2)

            contours, _ = cv2.findContours(
                dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Sort contours by area descending
            sorted_contours = sorted(
                contours, key=lambda c: cv2.contourArea(c), reverse=True
            )

            max_allowed_area = int(frame_w * frame_h * 0.90)  # skip near-full-screen border
            count = 0

            for cnt in sorted_contours:
                area = int(cv2.contourArea(cnt))
                if area < self.min_contour_area or area > max_allowed_area:
                    continue

                x, y, cw, ch = cv2.boundingRect(cnt)
                # Ensure proposal has meaningful aspect ratio and padding
                pad = 10
                px = max(0, x - pad)
                py = max(0, y - pad)
                pw = min(frame_w - px, cw + 2 * pad)
                ph = min(frame_h - py, ch + 2 * pad)

                crop = frame[py : py + ph, px : px + pw]
                if crop.size > 0:
                    proposals.append(
                        CandidateRegion(
                            bbox=(px, py, pw, ph),
                            crop=crop,
                            region_type="salient_contour",
                            area=pw * ph,
                        )
                    )
                    count += 1
                    if count >= self.max_contour_proposals:
                        break

        except Exception as exc:
            logger.debug("Contour candidate extraction encountered an error: %s", exc)

        return proposals
