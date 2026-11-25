
from ultralytics import YOLO
import multiprocessing

def main():
    # Load pretrained YOLO11 nano model
    model = YOLO("yolo11n.pt")

    # Train the model
    results = model.train(
    data=r"Face-Detection-1\data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    workers=8,
    patience=20,
    amp=True,
    cache=ram,
    plots=True
)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
