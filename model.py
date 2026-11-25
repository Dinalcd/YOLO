
from ultralytics import YOLO
import multiprocessing

def main():
    # Load your best trained model
    model = YOLO(
    r"runs\detect\train-2\weights\best.pt"
    )

    # Evaluate on the validation dataset
    metrics = model.val(
        data=r"Face-Detection-1/data.yaml",
        device=0,
        imgsz=640,
        batch=16,
        workers=4
    )

    # Display evaluation metrics
    print("\n===== MODEL PERFORMANCE =====")
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)
    print("Precision:", metrics.box.mp)
    print("Recall:", metrics.box.mr)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
