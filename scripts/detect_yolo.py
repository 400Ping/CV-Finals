from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO drink cup detection on an image, video, or camera.")
    parser.add_argument("--weights", type=Path, default=Path("models/yolo/drink_cup/weights/best.pt"))
    parser.add_argument("--source", default="0", help="Camera index, image path, video path, or folder.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cpu", help="Use cpu, mps, cuda, or a CUDA device index.")
    parser.add_argument("--project", type=Path, default=Path("outputs/yolo"))
    parser.add_argument("--name", default="predict")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def configure_matplotlib_cache() -> None:
    cache_dir = Path(".cache/matplotlib").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


def main() -> None:
    configure_matplotlib_cache()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {exc}. Run: pip install -r requirements-yolo.txt") from exc

    args = parse_args()
    if not args.weights.exists():
        raise SystemExit(f"Weights not found: {args.weights}. Run scripts/train_yolo.py first.")

    source: int | str = int(args.source) if args.source.isdigit() else args.source
    model = YOLO(str(args.weights))
    model.predict(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=str(args.project),
        name=args.name,
        show=args.show,
        save=True,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
