from ultralytics import YOLO

model = YOLO(
    r"runs\detect\train-2\weights\best.pt"
    )

results = model.predict(
    source="-I1-MS09uaqsLdGTFkgnS0Rcg1mmPyAj95ySg_eckoM_jpeg_jpg.rf.1d7e1149c77ef9aa5b2407ee7b24e2a9.jpg",
    conf=0.4,
    device=0
)

results[0].show()