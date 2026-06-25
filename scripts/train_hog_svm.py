from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from common import Box, HOG_WINDOW_SIZE, iou, iter_images, label_path_for, make_hog, preprocess_crop, read_yolo_boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an OpenCV HOG + linear SVM drink cup detector.")
    parser.add_argument("--images", type=Path, default=Path("dataset/images"))
    parser.add_argument("--labels", type=Path, default=Path("dataset/labels"))
    parser.add_argument("--model", type=Path, default=Path("models/cup_hog_svm.yml"))
    parser.add_argument("--negatives-per-image", type=int, default=40)
    parser.add_argument("--hard-negative-rounds", type=int, default=1)
    parser.add_argument("--hard-negatives-per-image", type=int, default=20)
    parser.add_argument("--hard-negative-step", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def random_negative_boxes(width: int, height: int, positives: list[Box], count: int) -> list[Box]:
    boxes: list[Box] = []
    attempts = 0
    while len(boxes) < count and attempts < count * 80:
        attempts += 1
        crop_h = random.randint(max(80, height // 8), max(100, height // 2))
        crop_w = int(crop_h * random.uniform(0.8, 1.25))
        if crop_w >= width or crop_h >= height:
            continue
        x1 = random.randint(0, width - crop_w)
        y1 = random.randint(0, height - crop_h)
        candidate = Box(x1, y1, x1 + crop_w, y1 + crop_h)
        if all(iou(candidate, pos) < 0.05 for pos in positives):
            boxes.append(candidate)
    return boxes


def train_svm(features: list[np.ndarray], targets: list[int]) -> cv2.ml_SVM:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.int32)

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(0.1)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 2000, 1e-6))
    svm.train(x, cv2.ml.ROW_SAMPLE, y)
    return svm


def mine_hard_negatives(
    image_paths: list[Path],
    labels_dir: Path,
    svm: cv2.ml_SVM,
    hog: cv2.HOGDescriptor,
    step: int,
    max_per_image: int,
) -> list[np.ndarray]:
    win_w, win_h = HOG_WINDOW_SIZE
    hard_features: list[np.ndarray] = []

    for image_path in tqdm(image_paths, desc="Mine hard negatives"):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        positives = read_yolo_boxes(label_path_for(image_path, labels_dir), width, height)
        mined_for_image = 0

        scale = 1.0
        current = image
        while current.shape[0] >= win_h and current.shape[1] >= win_w:
            gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            h, w = gray.shape[:2]
            inv = 1.0 / scale

            for y in range(0, h - win_h + 1, step):
                for x in range(0, w - win_w + 1, step):
                    box = Box(
                        int(x * inv),
                        int(y * inv),
                        int((x + win_w) * inv),
                        int((y + win_h) * inv),
                    ).clip(width, height)
                    if any(iou(box, pos) >= 0.05 for pos in positives):
                        continue

                    crop = gray[y : y + win_h, x : x + win_w]
                    feature = hog.compute(crop).reshape(1, -1).astype(np.float32)
                    _, pred = svm.predict(feature)
                    if int(pred[0, 0]) == 1:
                        hard_features.append(feature.ravel())
                        mined_for_image += 1
                        if mined_for_image >= max_per_image:
                            break
                if mined_for_image >= max_per_image:
                    break

            if mined_for_image >= max_per_image:
                break

            scale /= 1.25
            new_w = int(width * scale)
            new_h = int(height * scale)
            if new_w < win_w or new_h < win_h:
                break
            current = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return hard_features


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.model.parent.mkdir(parents=True, exist_ok=True)

    hog = make_hog()
    features: list[np.ndarray] = []
    targets: list[int] = []
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

        for box in boxes:
            crop = image[box.y1 : box.y2, box.x1 : box.x2]
            if crop.size == 0:
                continue
            features.append(hog.compute(preprocess_crop(crop)).ravel())
            targets.append(1)
            positive_count += 1

            flipped = cv2.flip(crop, 1)
            features.append(hog.compute(preprocess_crop(flipped)).ravel())
            targets.append(1)
            positive_count += 1

        for box in random_negative_boxes(width, height, boxes, args.negatives_per_image):
            crop = image[box.y1 : box.y2, box.x1 : box.x2]
            features.append(hog.compute(preprocess_crop(crop)).ravel())
            targets.append(-1)
            negative_count += 1

    if positive_count == 0:
        raise SystemExit("No positive labels found. Run: python scripts/annotate.py")
    if negative_count == 0:
        raise SystemExit("No negative samples were generated.")

    svm = train_svm(features, targets)
    hard_negative_count = 0
    for round_idx in range(args.hard_negative_rounds):
        hard_features = mine_hard_negatives(
            image_paths,
            args.labels,
            svm,
            hog,
            args.hard_negative_step,
            args.hard_negatives_per_image,
        )
        if not hard_features:
            print(f"Hard negative round {round_idx + 1}: no false positives found")
            break
        features.extend(hard_features)
        targets.extend([-1] * len(hard_features))
        hard_negative_count += len(hard_features)
        print(f"Hard negative round {round_idx + 1}: added {len(hard_features)} samples")
        svm = train_svm(features, targets)

    svm.save(str(args.model))

    metadata = {
        "detector": "OpenCV HOG + linear SVM",
        "class": "drink_cup",
        "positive_samples_with_flip": positive_count,
        "random_negative_samples": negative_count,
        "hard_negative_samples": hard_negative_count,
        "negative_samples": negative_count + hard_negative_count,
        "window_size": list(HOG_WINDOW_SIZE),
    }
    args.model.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved model to {args.model}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
