import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm

# Hyperparameters
batch_size = 16
learning_rate = 0.001
num_epochs = 20
num_classes = 5  # DR levels
checkpoint_path = './best_model.pth'

# Dataset Class
class RetinopathyDataset(Dataset):
    def __init__(self, ann_file, image_dir, transform=None, mode='single', test=False):
        self.ann_file = ann_file
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.test = test
        self.data = self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        img = Image.open(data['img_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)

        if not self.test:
            label = torch.tensor(data['dr_level'], dtype=torch.int64)
            return img, label
        else:
            return img

    def load_data(self):
        df = pd.read_csv(self.ann_file)
        data = []
        for _, row in df.iterrows():
            data.append({
                'img_path': os.path.join(self.image_dir, row['img_path']),
                'dr_level': int(row['patient_DR_Level']) if not self.test else None,
            })
        return data

# Augmentations
transform_train = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Model Architecture
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.efficientnet_b0(pretrained=True)
        self.backbone.classifier = nn.Sequential(
            nn.Linear(self.backbone.classifier[1].in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# Loss Function
class_weights = torch.tensor([1.0, 2.0, 1.5, 1.5, 2.0])  # Example weights for imbalance
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Training Function
def train_model(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_kappa = -1.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
        print(f"Train Loss: {running_loss / len(train_loader):.4f}, Kappa: {train_kappa:.4f}")

        # Validation
        val_kappa = evaluate_model(model, val_loader, device)
        print(f"Validation Kappa: {val_kappa:.4f}")

        scheduler.step(val_kappa)

        if val_kappa > best_kappa:
            best_kappa = val_kappa
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, checkpoint_path)

    print(f"Best Validation Kappa: {best_kappa:.4f}")
    model.load_state_dict(best_model_wts)
    return model

# Evaluation Function
def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return cohen_kappa_score(all_labels, all_preds, weights='quadratic')

# Main
if __name__ == "__main__":
    train_dataset = RetinopathyDataset('./DeepDRiD/train.csv', './DeepDRiD/train/', transform_train)
    val_dataset = RetinopathyDataset('./DeepDRiD/val.csv', './DeepDRiD/val/', transform_test)
    test_dataset = RetinopathyDataset('./DeepDRiD/test.csv', './DeepDRiD/test/', transform_test, test=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel(num_classes=num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5, verbose=True)

    model = train_model(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs)

    # Test Evaluation
    evaluate_model(model, test_loader, device)
