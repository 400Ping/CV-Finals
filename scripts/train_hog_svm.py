from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from common import Box, iou, iter_images, label_path_for, make_hog, preprocess_crop, read_yolo_boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an OpenCV HOG + linear SVM bottle detector.")
    parser.add_argument("--images", type=Path, default=Path("dataset/images"))
    parser.add_argument("--labels", type=Path, default=Path("dataset/labels"))
    parser.add_argument("--model", type=Path, default=Path("models/bottle_hog_svm.yml"))
    parser.add_argument("--negatives-per-image", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def random_negative_boxes(width: int, height: int, positives: list[Box], count: int) -> list[Box]:
    boxes: list[Box] = []
    attempts = 0
    while len(boxes) < count and attempts < count * 80:
        attempts += 1
        crop_h = random.randint(max(80, height // 8), max(100, height // 2))
        crop_w = int(crop_h * 0.5)
        if crop_w >= width or crop_h >= height:
            continue
        x1 = random.randint(0, width - crop_w)
        y1 = random.randint(0, height - crop_h)
        candidate = Box(x1, y1, x1 + crop_w, y1 + crop_h)
        if all(iou(candidate, pos) < 0.05 for pos in positives):
            boxes.append(candidate)
    return boxes


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.model.parent.mkdir(parents=True, exist_ok=True)

    hog = make_hog()
    features: list[np.ndarray] = []
    labels: list[int] = []
    positive_count = 0
    negative_count = 0

    image_paths = iter_images(args.images)
    if not image_paths:
        raise SystemExit(f"No images found in {args.images}")

    for image_path in tqdm(image_paths, desc="Extract HOG features"):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        boxes = read_yolo_boxes(label_path_for(image_path, args.labels), width, height)
        if not boxes:
            continue

        for box in boxes:
            crop = image[box.y1 : box.y2, box.x1 : box.x2]
            if crop.size == 0:
                continue
            features.append(hog.compute(preprocess_crop(crop)).ravel())
            labels.append(1)
            positive_count += 1

            flipped = cv2.flip(crop, 1)
            features.append(hog.compute(preprocess_crop(flipped)).ravel())
            labels.append(1)
            positive_count += 1

        for box in random_negative_boxes(width, height, boxes, args.negatives_per_image):
            crop = image[box.y1 : box.y2, box.x1 : box.x2]
            features.append(hog.compute(preprocess_crop(crop)).ravel())
            labels.append(-1)
            negative_count += 1

    if positive_count == 0:
        raise SystemExit("No positive labels found. Run: python scripts/annotate.py")
    if negative_count == 0:
        raise SystemExit("No negative samples were generated.")

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(0.1)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 2000, 1e-6))
    svm.train(x, cv2.ml.ROW_SAMPLE, y)
    svm.save(str(args.model))

    metadata = {
        "detector": "OpenCV HOG + linear SVM",
        "class": "pet_bottle",
        "positive_samples_with_flip": positive_count,
        "negative_samples": negative_count,
        "window_size": [64, 128],
    }
    args.model.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved model to {args.model}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
