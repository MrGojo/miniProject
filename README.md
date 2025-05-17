
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
# Deepfake Detection using kaggle Dataset

This project implements a deepfake detection model using the kaggle 140k real and fake image dataset. The model is is based on Custom Cnn architecture fine-tuned for binary classification (real vs. fake images).
https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces/data
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
# 🎭 Deepfake Detection using Convolutional Neural Networks (CNN)

This project uses a CNN-based model to detect deepfake content by analyzing visual artifacts in images or video frames. The model processes RGB images resized to 256x256 and outputs a binary prediction — **real (0)** or **deepfake (1)**.

---

## Training the Model

To train the model:

1. Prepare the dataset:
   - Collect labeled images or extract frames from real and deepfake videos.
   - Resize all frames to **256x256**.
   - Split data into training, validation, and test sets.

2. Load and preprocess:
   - Normalize pixel values to range [0,1].
   - Use `ImageDataGenerator` or manual data loaders.

3. Train the model:
   ```python
   model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
   model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

#Model Architecture
              Input (256x256x3)
              ↓
              Conv2D(32, 3x3) → ReLU → BatchNorm → MaxPooling(2x2)
              ↓
              Conv2D(64, 3x3) → ReLU → BatchNorm → MaxPooling(2x2)
              ↓
              Conv2D(128, 3x3) → ReLU → BatchNorm → MaxPooling(2x2)
              ↓
              Flatten
              ↓
              Dense(128) → ReLU → Dropout(0.1)
              ↓
              Dense(64) → ReLU → Dropout(0.1)
              ↓
              Dense(1) → Sigmoid



