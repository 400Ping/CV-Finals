from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from common import Box, HOG_WINDOW_SIZE, make_hog, non_max_suppression


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run drink cup detection on an image, video, or camera.")
    parser.add_argument("--model", type=Path, default=Path("models/cup_hog_svm.yml"))
    parser.add_argument("--source", default="0", help="Camera index, image path, or video path.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--step", type=int, default=24)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--nms-threshold", type=float, default=0.25)
    return parser.parse_args()


def detect_frame(frame: np.ndarray, svm: cv2.ml_SVM, step: int, score_threshold: float, nms_threshold: float) -> list[tuple[Box, float]]:
    hog = make_hog()
    frame_h, frame_w = frame.shape[:2]
    win_w, win_h = HOG_WINDOW_SIZE
    boxes: list[Box] = []
    scores: list[float] = []

    scale = 1.0
    current = frame
    while current.shape[0] >= win_h and current.shape[1] >= win_w:
        gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        h, w = gray.shape[:2]
        for y in range(0, h - win_h + 1, step):
            for x in range(0, w - win_w + 1, step):
                crop = gray[y : y + win_h, x : x + win_w]
                feature = hog.compute(crop).reshape(1, -1).astype(np.float32)
                _, pred = svm.predict(feature)
                if int(pred[0, 0]) != 1:
                    continue
                _, raw = svm.predict(feature, flags=cv2.ml.StatModel_RAW_OUTPUT)
                score = abs(float(raw[0, 0]))
                if score < score_threshold:
                    continue
                inv = 1.0 / scale
                box = Box(
                    int(x * inv),
                    int(y * inv),
                    int((x + win_w) * inv),
                    int((y + win_h) * inv),
                ).clip(frame_w, frame_h)
                boxes.append(box)
                scores.append(score)

        scale /= 1.25
        new_w = int(frame_w * scale)
        new_h = int(frame_h * scale)
        if new_w < win_w or new_h < win_h:
            break
        current = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    keep = non_max_suppression(boxes, scores, nms_threshold)
    return [(boxes[i], scores[i]) for i in keep]


def draw_detections(frame: np.ndarray, detections: list[tuple[Box, float]]) -> np.ndarray:
    canvas = frame.copy()
    for box, score in detections:
        cv2.rectangle(canvas, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), 3)
        cv2.putText(
            canvas,
            f"Drink cup {score:.2f}",
            (box.x1, max(24, box.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return canvas


def open_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}. Run python scripts/train_hog_svm.py first.")
    svm = cv2.ml.SVM_load(str(args.model))

    source = open_source(args.source)
    if isinstance(source, str) and Path(source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        frame = cv2.imread(source)
        if frame is None:
            raise SystemExit(f"Unable to read image: {source}")
        result = draw_detections(frame, detect_frame(frame, svm, args.step, args.score_threshold, args.nms_threshold))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.output), result)
        cv2.imshow("Drink cup detector", result)
        cv2.waitKey(0)
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Unable to open source: {args.source}")

    writer = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = draw_detections(frame, detect_frame(frame, svm, args.step, args.score_threshold, args.nms_threshold))
        if writer:
            writer.write(result)
        cv2.imshow("Drink cup detector", result)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
