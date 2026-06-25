from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from common import Box, iter_images, label_path_for, read_yolo_boxes, write_yolo_boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate PET bottle bounding boxes in YOLO text format.")
    parser.add_argument("--images", type=Path, default=Path("dataset/images"))
    parser.add_argument("--labels", type=Path, default=Path("dataset/labels"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.labels.mkdir(parents=True, exist_ok=True)
    image_paths = iter_images(args.images)
    if not image_paths:
        raise SystemExit(f"No images found in {args.images}")

    window = "Bottle annotator"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skip unreadable image: {image_path}")
            continue
        height, width = image.shape[:2]
        label_path = label_path_for(image_path, args.labels)
        boxes = read_yolo_boxes(label_path, width, height)
        drawing = {"active": False, "start": (0, 0), "current": (0, 0)}

        def render() -> None:
            canvas = image.copy()
            for box in boxes:
                cv2.rectangle(canvas, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), 3)
            if drawing["active"]:
                x1, y1 = drawing["start"]
                x2, y2 = drawing["current"]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(
                canvas,
                f"{image_path.name} | drag=box s=save n=next u=undo q=quit",
                (24, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window, canvas)

        def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing["active"] = True
                drawing["start"] = (x, y)
                drawing["current"] = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and drawing["active"]:
                drawing["current"] = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and drawing["active"]:
                drawing["active"] = False
                x1, y1 = drawing["start"]
                x2, y2 = x, y
                box = Box(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)).clip(width, height)
                if box.area > 100:
                    boxes.append(box)
            render()

        cv2.setMouseCallback(window, on_mouse)
        render()

        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == ord("u") and boxes:
                boxes.pop()
                render()
            elif key == ord("s"):
                write_yolo_boxes(label_path, boxes, width, height)
                print(f"Saved {label_path} ({len(boxes)} boxes)")
            elif key == ord("n"):
                write_yolo_boxes(label_path, boxes, width, height)
                break
            elif key == ord("q"):
                write_yolo_boxes(label_path, boxes, width, height)
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
