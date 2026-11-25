# Face Detection with YOLO11

This project implements a face detection system using the **YOLO11** architecture. It leverages the `ultralytics` library to train, evaluate, and run inference on a custom face detection dataset.

## Project Structure

- `dataset.py`: Script to download the dataset from Roboflow.
- `train.py`: Script to train the YOLO11 nano model on the provided dataset.
- `model.py`: Script to evaluate the performance of the best-trained model.
- `runmodel.py`: Script to run inference on a single sample image and display the results.
- `Face-Detection-1/`: Contains the dataset configuration and local image/label structure.
- `yolo11n.pt`: The pre-trained YOLO11 nano model weights.

## Requirements

To run this project, you need to have Python installed and the required libraries. You can install them using:

```bash
pip install -r requirements.txt
```

## Getting Started

### 1. Download the Dataset
Run the following script to download the face detection dataset from Roboflow:
```bash
python dataset.py
```

### 2. Train the Model
Use the `train.py` script to start the training process. This will use the `yolo11n.pt` weights as a starting point and train for 100 epochs.
```bash
python train.py
```

### 3. Evaluate the Model
After training is complete, you can evaluate the model's performance (mAP50, mAP50-95, Precision, and Recall) using `model.py`.
```bash
python model.py
```

### 4. Run Inference
To test the model on a sample image, run:
```bash
python runmodel.py
```

## Configuration

- **Dataset Configuration**: The dataset paths are defined in `Face-Detection-1/data.yaml`.
- **Model Weights**: After training, the best weights are saved in the `runs/detect/train_X/weights/` directory.
- **Script Updates**: If you start a new training session (e.g., `train-3`), you must update the path in `model.py` and `runmodel.py` to point to the new `best.pt` file.

## License
This project is for academic purposes.
