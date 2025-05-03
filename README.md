<<<<<<< HEAD
# DeepGuardAI - Deepfake Detection System

A web-based deepfake detection system that uses machine learning to analyze images for signs of manipulation.

## Project Structure

```
deepguard_backend/
├── deepguard_backend/     # Main project settings
├── detector/              # Deepfake detection app
│   ├── models.py         # Database models
│   ├── views.py          # API endpoints
│   └── urls.py           # URL routing
├── media/                 # Uploaded files
├── static/               # Static files
└── requirements.txt      # Project dependencies
```

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
=======
# Deepfake Detection using WildDeepfake Dataset

This project implements a deepfake detection model using the WildDeepfake dataset. The model is based on a ResNet50 architecture fine-tuned for binary classification (real vs. fake images).

## Setup

1. Install the required dependencies:
>>>>>>> 121b4dace8f3a3dfac0ddcf8b5d01ab725b33cb4
```bash
pip install -r requirements.txt
```

<<<<<<< HEAD
3. Run migrations:
```bash
python manage.py makemigrations
python manage.py migrate
```

4. Create a superuser (optional):
```bash
python manage.py createsuperuser
```

5. Run the development server:
```bash
python manage.py runserver
```

## API Endpoints

- `POST /api/analyze/` - Upload and analyze an image
  - Request: Form data with 'image' field
  - Response: Analysis results in JSON format

- `GET /api/result/<id>/` - Get analysis result by ID
  - Response: Analysis results in JSON format

## Frontend Integration

The frontend is located in the `Mini Project` directory. To use it with the backend:

1. Make sure the Django server is running on `http://localhost:8000`
2. Open the frontend HTML files in a web browser
3. The frontend will automatically connect to the backend API

## Model Integration

To integrate your deepfake detection model:

1. Replace the `predict_deepfake()` function in `detector/views.py` with your model's prediction code
2. Make sure your model is compatible with the expected input/output format
3. Update the requirements.txt file with any additional dependencies your model needs

## Development

- Backend API runs on `http://localhost:8000`
- Admin interface available at `http://localhost:8000/admin`
- API documentation available at `http://localhost:8000/api/docs`

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
=======
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
>>>>>>> 121b4dace8f3a3dfac0ddcf8b5d01ab725b33cb4
