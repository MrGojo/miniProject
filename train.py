
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datasets import load_dataset
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

from model import DeepfakeDetector
from data_utils import get_data_loaders

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in tqdm(loader, desc='Training'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predicted.squeeze() == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1))

            total_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted.squeeze() == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds)
    return total_loss / len(loader), correct / total, f1

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

def main(args):
    set_seed()

    # Load dataset and inspect structure
    dataset = load_dataset("xingjunm/WildDeepfake", split="train[:5%]")

    print("First sample in dataset:")
    print(dataset[0])
    print("Dataset feature keys:", dataset.features)

    
    # Manually assign label based on __key__ path
    label_counts = Counter()

    def extract_label(example):
        if "fake" in example["__key__"]:
            label = 1
        else:
            label = 0
        label_counts[label] += 1
        return {**example, "label": label}

    dataset = dataset.map(extract_label)


    # Count labels
    label_counts = Counter([example[label_key] for example in dataset])
    for label, count in sorted(label_counts.items()):
        label_name = "Fake" if label == 1 else "Real"
        print(f"Label {label} ({label_name}): {count} samples")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = DeepfakeDetector(pretrained=True).to(device)

    # Compute pos_weight for class imbalance
    real = label_counts[0]
    fake = label_counts[1]
    pos_weight = torch.tensor([real / fake]).to(device)
    print(f"Imbalance Ratio: {real / fake:.2f}")
    print(f"Applied pos_weight = {pos_weight.item():.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    train_loader, val_loader, test_loader = get_data_loaders(batch_size=args.batch_size)

    best_val_acc, patience, no_improve = 0, args.patience, 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs with no improvement.")
                break

        scheduler.step()

    # Plotting
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Val')
    plt.title("Accuracy")
    plt.legend()

    plt.savefig("logs/training_curves.png")
    plt.close()

    # Final Evaluation
    model.load_state_dict(torch.load('checkpoints/best_model.pth'))
    evaluate(model, test_loader, device)

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--patience', type=int, default=3)
args = parser.parse_args()

main(args)
