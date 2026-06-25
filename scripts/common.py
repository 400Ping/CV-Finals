from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    def clip(self, width: int, height: int) -> "Box":
        return Box(
            max(0, min(self.x1, width - 1)),
            max(0, min(self.y1, height - 1)),
            max(0, min(self.x2, width)),
            max(0, min(self.y2, height)),
        )

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


def iter_images(image_dir: Path) -> list[Path]:
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def label_path_for(image_path: Path, labels_dir: Path) -> Path:
    return labels_dir / f"{image_path.stem}.txt"


def read_yolo_boxes(label_path: Path, image_width: int, image_height: int) -> list[Box]:
    if not label_path.exists():
        return []

    boxes: list[Box] = []
    for raw in label_path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Bad label line in {label_path}: {line}")
        class_id, cx, cy, w, h = parts
        if int(class_id) != 0:
            continue
        cx_f, cy_f, w_f, h_f = map(float, (cx, cy, w, h))
        bw = w_f * image_width
        bh = h_f * image_height
        x1 = int(round(cx_f * image_width - bw / 2))
        y1 = int(round(cy_f * image_height - bh / 2))
        x2 = int(round(cx_f * image_width + bw / 2))
        y2 = int(round(cy_f * image_height + bh / 2))
        box = Box(x1, y1, x2, y2).clip(image_width, image_height)
        if box.area > 0:
            boxes.append(box)
    return boxes


def write_yolo_boxes(label_path: Path, boxes: Iterable[Box], image_width: int, image_height: int) -> None:
    lines = []
    for box in boxes:
        box = box.clip(image_width, image_height)
        if box.area <= 0:
            continue
        cx = ((box.x1 + box.x2) / 2) / image_width
        cy = ((box.y1 + box.y2) / 2) / image_height
        w = (box.x2 - box.x1) / image_width
        h = (box.y2 - box.y1) / image_height
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def iou(a: Box, b: Box) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union else 0.0


def non_max_suppression(boxes: list[Box], scores: list[float], threshold: float) -> list[int]:
    if not boxes:
        return []

    order = np.argsort(scores)[::-1]
    keep: list[int] = []
    suppressed = np.zeros(len(boxes), dtype=bool)

    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(int(idx))
        for other in order:
            if other == idx or suppressed[other]:
                continue
            if iou(boxes[idx], boxes[other]) >= threshold:
                suppressed[other] = True
    return keep


def make_hog() -> cv2.HOGDescriptor:
    return cv2.HOGDescriptor(
        _winSize=(64, 128),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )


def preprocess_crop(crop: np.ndarray) -> np.ndarray:
    resized = cv2.resize(crop, (64, 128), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)
