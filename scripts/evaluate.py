from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from common import iou, iter_images, label_path_for, read_yolo_boxes
from detect import detect_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate detector precision and recall against YOLO labels.")
    parser.add_argument("--images", type=Path, default=Path("dataset/images"))
    parser.add_argument("--labels", type=Path, default=Path("dataset/labels"))
    parser.add_argument("--model", type=Path, default=Path("models/bottle_hog_svm.yml"))
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--step", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    svm = cv2.ml.SVM_load(str(args.model))
    tp = fp = fn = 0
    labeled_images = 0

    for image_path in iter_images(args.images):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        ground_truth = read_yolo_boxes(label_path_for(image_path, args.labels), width, height)
        if not ground_truth:
            continue
        labeled_images += 1

        detections = detect_frame(image, svm, args.step, 0.0, 0.25)
        matched = set()
        for pred_box, _score in detections:
            best_idx = -1
            best_iou = 0.0
            for idx, gt_box in enumerate(ground_truth):
                if idx in matched:
                    continue
                value = iou(pred_box, gt_box)
                if value > best_iou:
                    best_iou = value
                    best_idx = idx
            if best_iou >= args.iou:
                tp += 1
                matched.add(best_idx)
            else:
                fp += 1
        fn += len(ground_truth) - len(matched)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    print(f"Labeled images: {labeled_images}")
    print(f"TP: {tp}  FP: {fp}  FN: {fn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")


if __name__ == "__main__":
    main()
