from __future__ import annotations

import modal


app = modal.App("drink-cup-yolo-training")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("ultralytics>=8.2")
    .add_local_dir("dataset", remote_path="/root/dataset")
    .add_local_file("dataset.yaml", remote_path="/root/dataset.yaml")
)

volume = modal.Volume.from_name("drink-cup-yolo-runs", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10",
    timeout=60 * 60 * 2,
    volumes={"/runs": volume},
)
def train_yolo(
    model_name: str = "yolov8n.pt",
    epochs: int = 80,
    imgsz: int = 640,
    batch: int = 8,
) -> None:
    from ultralytics import YOLO

    model = YOLO(model_name)
    model.train(
        data="/root/dataset.yaml",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0,
        project="/runs/yolo",
        name="drink_cup",
        exist_ok=True,
    )
    volume.commit()


@app.local_entrypoint()
def main(
    model_name: str = "yolov8n.pt",
    epochs: int = 80,
    imgsz: int = 640,
    batch: int = 8,
) -> None:
    train_yolo.remote(model_name=model_name, epochs=epochs, imgsz=imgsz, batch=batch)
    print("Training finished.")
    print("Download best weights with:")
    print(
        "modal volume get drink-cup-yolo-runs "
        "yolo/drink_cup/weights/best.pt "
        "models/yolo/drink_cup/weights/best.pt"
    )
