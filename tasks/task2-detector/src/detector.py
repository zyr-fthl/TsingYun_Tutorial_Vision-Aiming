"""Task 2 MNIST-board detector helpers with student TODO extension points.

This file belongs to Task 2. The simulator runner imports it so that a Task 2
implementation can be tested both offline and inside the Unity simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import sys
import builtins
from pathlib import Path
import cv2
import numpy as np

from simulator_client.protocol import Matrix3x3
from model import classify_mnist_digit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_SRC = str(PROJECT_ROOT / "tasks" / "task2-detector" / "src")
if TARGET_SRC not in sys.path:
    sys.path.insert(0, TARGET_SRC)

try:
    from train import MNISTClassifier
    builtins.MNISTClassifier = MNISTClassifier
except Exception as e:
    print(f"⏰ [Gimbal Pipeline] Global model inject note: {e}")

Point2D = tuple[float, float]
CornerSet = tuple[Point2D, Point2D, Point2D, Point2D]
RgbPixel = tuple[int, int, int]
ImageLike = np.ndarray
WARP_OUTPUT_SIZE = 128
MNIST_INNER_RATIO = 0.69


@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> Point2D:
        return (self.x + self.width * 0.5, self.y + self.height * 0.5)


@dataclass
class Detection:
    class_id: int
    confidence: float
    bbox: BoundingBox
    corners: CornerSet
    rvec: object | None = None
    tvec: object | None = None


def _bbox_from_corners(corners: Sequence[Point2D]) -> BoundingBox:
    if len(corners) != 4:
        raise ValueError(f"Expected 4 corners, got {len(corners)}")

    xs = [float(point[0]) for point in corners]
    ys = [float(point[1]) for point in corners]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    return BoundingBox(
        x=min_x,
        y=min_y,
        width=max_x - min_x + 1.0,
        height=max_y - min_y + 1.0,
    )


def _crop_bounds(corners: Sequence[Point2D], image_width: int, image_height: int) -> tuple[int, int, int, int]:
    bbox = _bbox_from_corners(corners)
    x0 = max(0, min(image_width, int(np.floor(bbox.x))))
    y0 = max(0, min(image_height, int(np.floor(bbox.y))))
    x1 = max(0, min(image_width, int(np.ceil(bbox.x + bbox.width))))
    y1 = max(0, min(image_height, int(np.ceil(bbox.y + bbox.height))))
    return x0, y0, x1, y1


def crop_bbox(image: np.ndarray, corner_candidates: Sequence[Sequence[Point2D]]) -> list[np.ndarray]:
    crops: list[np.ndarray] = []
    for corners in corner_candidates:
        if len(corners) != 4:
            continue

        # `corners` are expected in LU, RU, RD, LD order.
        src = np.array(corners, dtype=np.float32)

        # Shrink the source quad toward its center so the warp removes the outer red border.
        center = np.mean(src, axis=0)
        src = center + (src - center) * MNIST_INNER_RATIO

        dst = np.array(
            [
                [0, 0],
                [WARP_OUTPUT_SIZE - 1, 0],
                [WARP_OUTPUT_SIZE - 1, WARP_OUTPUT_SIZE - 1],
                [0, WARP_OUTPUT_SIZE - 1],
            ],
            dtype=np.float32,
        )

        perspective = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(
            image,
            perspective,
            (WARP_OUTPUT_SIZE, WARP_OUTPUT_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        crops.append(warped)
    return crops


def order_corners(corners: Sequence[Point2D]) -> CornerSet:
    pts = np.array(corners, dtype=np.float32)
    
    sum_pts = pts.sum(axis=1)
    diff_pts = np.diff(pts, axis=1).flatten()  

    lu = corners[np.argmin(sum_pts)]
    rd = corners[np.argmax(sum_pts)]
    ru = corners[np.argmin(diff_pts)]
    ld = corners[np.argmax(diff_pts)]
    
    return (lu, ru, rd, ld)


def detect_bbox(image: ImageLike, threshold: int = 200) -> list[CornerSet]:
    img_uint8 = np.asarray(image, dtype=np.uint8)
    
    if len(img_uint8.shape) == 3:
        c0 = img_uint8[:, :, 0]
        c2 = img_uint8[:, :, 2]
        
        diff1 = cv2.subtract(c2, c0)  
        diff2 = cv2.subtract(c0, c2)  
        
        red_diff = diff1 if np.max(diff1) >= np.max(diff2) else diff2
    else:
        red_diff = img_uint8.copy()
    
    max_val = np.max(red_diff)
    actual_thresh = int(max_val * 0.5) if threshold > max_val and max_val > 30 else threshold

    _, red_mask = cv2.threshold(red_diff, actual_thresh, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    corner_candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 30:  
            continue
            
        peri = cv2.arcLength(contour, True)

        polygon = None
        for eps_coef in [0.02, 0.025, 0.03, 0.035, 0.04, 0.05]:
            poly_try = cv2.approxPolyDP(contour, eps_coef * peri, True)
            if len(poly_try) == 4:
                polygon = poly_try
                break
                
        if polygon is None:
            continue
            
        if not cv2.isContourConvex(polygon):
            continue
            
        pts_list = [tuple(p[0]) for p in polygon]
        ordered = order_corners(pts_list)
        
        corner_candidates.append(ordered)
        
    return corner_candidates

def detect_mnist_board(image: ImageLike, threshold: int = 200) -> list[Detection]:
    corner_candidates = detect_bbox(image, threshold)
    
    crops = crop_bbox(image, corner_candidates)
    
    detections = []
    for i, (corners, crop) in enumerate(zip(corner_candidates, crops)):
        digit, confidence = classify_mnist_digit(crop)
        
        if confidence >= 0.0:  
            bbox = _bbox_from_corners(corners)

            det = Detection(
                class_id=digit,
                confidence=confidence,
                bbox=bbox,
                corners=corners
            )
            detections.append(det)
            
    return detections


def solve_pnp(
    detections: Sequence[Detection],
    camera_matrix: Matrix3x3,
    board_width_meters: float,
    board_height_meters: float,
    dist_coeffs: Sequence[float] | None = None,
) -> list[Detection]:
    half_w = board_width_meters / 2.0
    half_h = board_height_meters / 2.0
    
    object_points = np.array([
        [-half_w, -half_h, 0.0],  # LU
        [ half_w, -half_h, 0.0],  # RU
        [ half_w,  half_h, 0.0],  # RD
        [-half_w,  half_h, 0.0]   # LD
    ], dtype=np.float32)
    
    camera_array = np.array(camera_matrix, dtype=np.float64).reshape(3, 3)
    
    if dist_coeffs is not None:
        dist_array = np.array(dist_coeffs, dtype=np.float64)
    else:
        dist_array = np.zeros(5, dtype=np.float64)
        
    result = []
    for det in detections:
        image_points = np.array(det.corners, dtype=np.float32)
        
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_array,
            dist_array,
            flags=cv2.SOLVEPNP_IPPE_SQUARE  
        )
        
        if not success:
            continue
            
        det.rvec = rvec
        det.tvec = tvec
        result.append(det)
        
    return result
