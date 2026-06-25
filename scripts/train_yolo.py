from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO drink cup detector.")
    parser.add_argument("--data", type=Path, default=Path("dataset.yaml"))
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO checkpoint or model yaml.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="Use cpu, mps, cuda, or a CUDA device index.")
    parser.add_argument("--project", type=Path, default=Path("models/yolo"))
    parser.add_argument("--name", default="drink_cup")
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
    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
