import numpy as np
import pandas as pd
import os
import random
import copy
import sys
from PIL import Image

import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

# Hyperparameters
batch_size = 24
num_classes = 5
learning_rate = 0.0001
num_epochs = 20

# Dataset Definition
class DiabeticRetinopathyDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name, level = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, f"{img_name}.jpeg")
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(level, dtype=torch.long)
        return image, label

# Data Preparation
def load_and_balance_data(csv_path, image_dir):
    df = pd.read_csv(csv_path)

    # Filter out rows where the image file does not exist
    df['exists'] = df['image'].apply(lambda x: os.path.exists(os.path.join(image_dir, f"{x}.jpeg")))
    df = df[df['exists']].drop(columns=['exists'])

    # Balance the dataset
    stage_counts = df['level'].value_counts()
    min_samples = stage_counts.min()
    balanced_df = df.groupby('level').apply(lambda x: x.sample(min_samples, random_state=42)).reset_index(drop=True)

    # Split the balanced dataset for training
    train_df, _ = train_test_split(balanced_df, test_size=0.2, stratify=balanced_df['level'], random_state=42)
    return train_df, df

# Transformations
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Model Definition
class DRModel(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.5):
        super(DRModel, self).__init__()
        self.backbone = models.resnet50(pretrained=False)
        self.backbone.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# Training and Evaluation Functions
def train_model(model, train_loader, val_loader, device, criterion, optimizer, scheduler, num_epochs):
    best_model_weights = copy.deepcopy(model.state_dict())
    best_kappa = -1

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        running_loss = 0.0
        train_preds, train_labels = [], []

        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            train_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        train_kappa = cohen_kappa_score(train_labels, train_preds, weights='quadratic')
        print(f"Train Loss: {epoch_loss:.4f} | Train Kappa: {train_kappa:.4f}")

        # Validation phase
        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_kappa = cohen_kappa_score(val_labels, val_preds, weights='quadratic')
        print(f"Val Loss: {epoch_val_loss:.4f} | Val Kappa: {val_kappa:.4f}")

        if val_kappa > best_kappa:
            best_kappa = val_kappa
            best_model_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_weights)
    return model

# Main Pipeline
def main():
    csv_path = "./7/trainLabels.csv"
    image_dir = "./7/resized_train_cropped/resized_train_cropped"

    # Load and balance the data
    train_df, val_df = load_and_balance_data(csv_path, image_dir)

    # Create datasets and dataloaders
    train_dataset = DiabeticRetinopathyDataset(train_df, image_dir, transform=transform_train)
    val_dataset = DiabeticRetinopathyDataset(val_df, image_dir, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check for GPU
    model = DRModel(num_classes=num_classes).to(device)  # Move model to device

    # Define loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Train and validate the model
    model = train_model(model, train_loader, val_loader, device, criterion, optimizer, scheduler, num_epochs)

    # Save the trained model
    torch.save(model.state_dict(), "dr_model.pth")
    print("Model training complete and saved.")

if __name__ == "__main__":
    main()
