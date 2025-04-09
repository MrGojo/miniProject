from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
import torch
from PIL import Image

class WildDeepfakeDataset(Dataset):
    def __init__(self, split='train', transform=None):
        self.dataset = load_dataset("xingjunm/WildDeepfake", split=split)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert('RGB')
        label = item['label']

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

def get_data_loaders(batch_size=32):
    train_dataset = WildDeepfakeDataset(split='train')
    val_dataset = WildDeepfakeDataset(split='validation')
    test_dataset = WildDeepfakeDataset(split='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader