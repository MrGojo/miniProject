import torch
from torchvision import transforms
from PIL import Image
import argparse
import os
from model import DeepfakeDetector

# Define image transforms (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

def load_model(model_path, device):
    model = DeepfakeDetector(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict(image_path, model_path='checkpoints/best_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = load_model(model_path, device)
    image = load_image(image_path).to(device)

    with torch.no_grad():
        output = model(image)
        prob = torch.sigmoid(output).item()
        label = 1 if prob > 0.5 else 0

    print(f"\n Image: {image_path}")
    print(f"Prediction: {'Fake' if label == 1 else 'Real'}")
    print(f"Confidence: {prob:.4f}")

    return label, prob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--model', default='checkpoints/best_model.pth', help='Path to trained model')
    args = parser.parse_args()

    predict(args.image, args.model)