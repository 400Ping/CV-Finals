from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO drink cup detector.")
    parser.add_argument("--data", type=Path, default=Path("dataset.yaml"))
    parser.add_argument("--weights", type=Path, default=Path("models/yolo/drink_cup/weights/best.pt"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu", help="Use cpu, mps, cuda, or a CUDA device index.")
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

    model = YOLO(str(args.weights))
    metrics = model.val(data=str(args.data), imgsz=args.imgsz, device=args.device)
    box = metrics.box
    print(f"Precision: {box.mp:.3f}")
    print(f"Recall: {box.mr:.3f}")
    print(f"mAP50: {box.map50:.3f}")
    print(f"mAP50-95: {box.map:.3f}")


if __name__ == "__main__":
    main()
