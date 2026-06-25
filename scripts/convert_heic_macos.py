from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert HEIC photos to JPG with macOS sips.")
    parser.add_argument("--input", type=Path, default=Path("Data"))
    parser.add_argument("--output", type=Path, default=Path("dataset/images"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    heic_paths = sorted(args.input.glob("*.HEIC"))
    if not heic_paths:
        raise SystemExit(f"No HEIC files found in {args.input}")

    temp_path = Path("/tmp")
    for source in heic_paths:
        target = args.output / f"{source.stem}.jpg"
        subprocess.run(
            ["qlmanage", "-t", "-s", "1600", "-o", str(temp_path), str(source)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        thumbnail = temp_path / f"{source.name}.png"
        if not thumbnail.exists():
            raise RuntimeError(f"Quick Look did not create thumbnail for {source}")
        subprocess.run(
            ["sips", "-s", "format", "jpeg", str(thumbnail), "--out", str(target)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        print(f"{source} -> {target}")


if __name__ == "__main__":
    main()
