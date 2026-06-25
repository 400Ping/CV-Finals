from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from common import iter_images, label_path_for


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create low-light augmented training images.")
    parser.add_argument("--images", type=Path, default=Path("dataset/images"))
    parser.add_argument("--labels", type=Path, default=Path("dataset/labels"))
    parser.add_argument("--brightness", type=float, default=0.45)
    parser.add_argument("--gamma", type=float, default=1.6)
    return parser.parse_args()


def low_light(image: np.ndarray, brightness: float, gamma: float) -> np.ndarray:
    dark = np.clip(image.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(dark, table)


def main() -> None:
    args = parse_args()
    for image_path in iter_images(args.images):
        if image_path.stem.endswith("_lowlight"):
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        output_image = image_path.with_name(f"{image_path.stem}_lowlight{image_path.suffix}")
        cv2.imwrite(str(output_image), low_light(image, args.brightness, args.gamma))

        source_label = label_path_for(image_path, args.labels)
        output_label = args.labels / f"{image_path.stem}_lowlight.txt"
        if source_label.exists():
            shutil.copyfile(source_label, output_label)
        print(f"Created {output_image}")


if __name__ == "__main__":
    main()
