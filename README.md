# Deepfake Detection using WildDeepfake Dataset

This project implements a deepfake detection model using the WildDeepfake dataset. The model is based on a ResNet50 architecture fine-tuned for binary classification (real vs. fake images).

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. The project structure:
- `model.py`: Contains the model architecture
- `data_utils.py`: Handles dataset loading and preprocessing
- `train.py`: Main training script
- `requirements.txt`: Project dependencies

## Training the Model

To train the model, simply run:
```bash
python train.py
```

The script will:
- Load and preprocess the WildDeepfake dataset
- Train the model for 10 epochs
- Save the best model based on validation accuracy
- Generate training curves showing loss and accuracy

## Model Architecture

The model uses a ResNet50 backbone with the following modifications:
- Final fully connected layer replaced with a custom head
- Binary classification output (real vs. fake)
- Sigmoid activation for probability output

## Training Details

- Batch size: 32
- Learning rate: 0.001
- Optimizer: Adam
- Loss function: Binary Cross Entropy
- Image size: 224x224
- Data augmentation: Standard ImageNet normalization

## Output

The training script will generate:
- `best_model.pth`: The best model weights based on validation accuracy
- `training_curves.png`: Visualization of training and validation metrics 