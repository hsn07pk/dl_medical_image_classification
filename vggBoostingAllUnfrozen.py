import copy
import os
import random
import sys

import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
import torch.nn.functional as F 
from sklearn.ensemble import GradientBoostingClassifier


# Hyper Parameters
batch_size = 24
num_classes = 5  # 5 DR levels
learning_rate = 0.0001
num_epochs = 10


class RetinopathyDataset(Dataset):
    def __init__(self, ann_file, image_dir, transform=None, mode='single', test=False):
        self.ann_file = ann_file
        self.image_dir = image_dir
        self.transform = transform

        self.test = test
        self.mode = mode

        if self.mode == 'single':
            self.data = self.load_data()
        else:
            self.data = self.load_data_dual()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.mode == 'single':
            return self.get_item(index)
        else:
            return self.get_item_dual(index)

    # 1. single image
    def load_data(self):
        df = pd.read_csv(self.ann_file)

        data = []
        for _, row in df.iterrows():
            file_info = dict()
            file_info['img_path'] = os.path.join(self.image_dir, row['img_path'])
            if not self.test:
                file_info['dr_level'] = int(row['patient_DR_Level'])
            data.append(file_info)
        return data

    def get_item(self, index):
        data = self.data[index]
        img = Image.open(data['img_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)

        if not self.test:
            label = torch.tensor(data['dr_level'], dtype=torch.int64)
            return img, label
        else:
            return img

    # 2. dual image
    def load_data_dual(self):
        df = pd.read_csv(self.ann_file)

        df['prefix'] = df['image_id'].str.split('_').str[0]  # The patient id of each image
        df['suffix'] = df['image_id'].str.split('_').str[1].str[0]  # The left or right eye
        grouped = df.groupby(['prefix', 'suffix'])

        data = []
        for (prefix, suffix), group in grouped:
            file_info = dict()
            file_info['img_path1'] = os.path.join(self.image_dir, group.iloc[0]['img_path'])
            file_info['img_path2'] = os.path.join(self.image_dir, group.iloc[1]['img_path'])
            if not self.test:
                file_info['dr_level'] = int(group.iloc[0]['patient_DR_Level'])
            data.append(file_info)
        return data

    def get_item_dual(self, index):
        data = self.data[index]
        img1 = Image.open(data['img_path1']).convert('RGB')
        img2 = Image.open(data['img_path2']).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if not self.test:
            label = torch.tensor(data['dr_level'], dtype=torch.int64)
            return [img1, img2], label
        else:
            return [img1, img2]


class CutOut(object):
    def __init__(self, mask_size, p=0.5):
        self.mask_size = mask_size
        self.p = p

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img

        # Ensure the image is a tensor
        if not isinstance(img, torch.Tensor):
            raise TypeError('Input image must be a torch.Tensor')

        # Get height and width of the image
        h, w = img.shape[1], img.shape[2]
        mask_size_half = self.mask_size // 2
        offset = 1 if self.mask_size % 2 == 0 else 0

        cx = np.random.randint(mask_size_half, w + offset - mask_size_half)
        cy = np.random.randint(mask_size_half, h + offset - mask_size_half)

        xmin, xmax = cx - mask_size_half, cx + mask_size_half + offset
        ymin, ymax = cy - mask_size_half, cy + mask_size_half + offset
        xmin, xmax = max(0, xmin), min(w, xmax)
        ymin, ymax = max(0, ymin), min(h, ymax)

        img[:, ymin:ymax, xmin:xmax] = 0
        return img


class SLORandomPad:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        pad_width = max(0, self.size[0] - img.width)
        pad_height = max(0, self.size[1] - img.height)
        pad_left = random.randint(0, pad_width)
        pad_top = random.randint(0, pad_height)
        pad_right = pad_width - pad_left
        pad_bottom = pad_height - pad_top
        return transforms.functional.pad(img, (pad_left, pad_top, pad_right, pad_bottom))


class FundRandomRotate:
    def __init__(self, prob, degree):
        self.prob = prob
        self.degree = degree

    def __call__(self, img):
        if random.random() < self.prob:
            angle = random.uniform(-self.degree, self.degree)
            return transforms.functional.rotate(img, angle)
        return img


transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((210, 210)),
    SLORandomPad((224, 224)),
    # FundRandomRotate(prob=0.5, degree=30),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=(0.1, 0.9)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def train_model(model, train_loader, val_loader, device, criterion, optimizer, lr_scheduler, num_epochs=25,
                checkpoint_path='model.pth'):
    best_model = model.state_dict()
    best_epoch = None
    best_val_kappa = -1.0  # Initialize the best kappa score

    for epoch in range(1, num_epochs + 1):
        print(f'\nEpoch {epoch}/{num_epochs}')
        running_loss = []
        all_preds = []
        all_labels = []

        model.train()

        with tqdm(total=len(train_loader), desc=f'Training', unit=' batch', file=sys.stdout) as pbar:
            for images, labels in train_loader:
                if not isinstance(images, list):
                    images = images.to(device)  # single image case
                else:
                    images = [x.to(device) for x in images]  # dual images case

                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels.long())

                loss.backward()
                optimizer.step()

                preds = torch.argmax(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                running_loss.append(loss.item())

                pbar.set_postfix({'lr': f'{optimizer.param_groups[0]["lr"]:.1e}', 'Loss': f'{loss.item():.4f}'})
                pbar.update(1)

        lr_scheduler.step()

        epoch_loss = sum(running_loss) / len(running_loss)

        train_metrics = compute_metrics(all_preds, all_labels, per_class=True)
        kappa, accuracy, precision, recall = train_metrics[:4]

        print(f'[Train] Kappa: {kappa:.4f} Accuracy: {accuracy:.4f} '
              f'Precision: {precision:.4f} Recall: {recall:.4f} Loss: {epoch_loss:.4f}')

        if len(train_metrics) > 4:
            precision_per_class, recall_per_class = train_metrics[4:]
            for i, (precision, recall) in enumerate(zip(precision_per_class, recall_per_class)):
                print(f'[Train] Class {i}: Precision: {precision:.4f}, Recall: {recall:.4f}')

        # Evaluation on the validation set at the end of each epoch
        val_metrics = evaluate_model(model, val_loader, device)
        val_kappa, val_accuracy, val_precision, val_recall = val_metrics[:4]
        print(f'[Val] Kappa: {val_kappa:.4f} Accuracy: {val_accuracy:.4f} '
              f'Precision: {val_precision:.4f} Recall: {val_recall:.4f}')

        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            best_epoch = epoch
            best_model = model.state_dict()
            torch.save(best_model, checkpoint_path)

    print(f'[Val] Best kappa: {best_val_kappa:.4f}, Epoch {best_epoch}')

    return model



def evaluate_model(model, test_loader, device, test_only=False, prediction_path='./test_predictions.csv'):
    model.eval()

    all_preds = []
    all_labels = []
    all_image_ids = []

    with tqdm(total=len(test_loader), desc=f'Evaluating', unit=' batch', file=sys.stdout) as pbar:
        for i, data in enumerate(test_loader):

            if test_only:
                images = data
            else:
                images, labels = data

            if not isinstance(images, list):
                images = images.to(device)  # single image case
            else:
                images = [x.to(device) for x in images]  # dual images case

            with torch.no_grad():
                outputs = model(images)
                preds = torch.argmax(outputs, 1)

            if not isinstance(images, list):
                # single image case
                all_preds.extend(preds.cpu().numpy())
                image_ids = [
                    os.path.basename(test_loader.dataset.data[idx]['img_path']) for idx in
                    range(i * test_loader.batch_size, i * test_loader.batch_size + len(images))
                ]
                all_image_ids.extend(image_ids)
                if not test_only:
                    all_labels.extend(labels.numpy())
            else:
                # dual images case
                for k in range(2):
                    all_preds.extend(preds.cpu().numpy())
                    image_ids = [
                        os.path.basename(test_loader.dataset.data[idx][f'img_path{k + 1}']) for idx in
                        range(i * test_loader.batch_size, i * test_loader.batch_size + len(images[k]))
                    ]
                    all_image_ids.extend(image_ids)
                    if not test_only:
                        all_labels.extend(labels.numpy())

            pbar.update(1)

    # Save predictions to csv file for Kaggle online evaluation
    if test_only:
        df = pd.DataFrame({
            'ID': all_image_ids,
            'TARGET': all_preds
        })
        df.to_csv(prediction_path, index=False)
        print(f'[Test] Save predictions to {os.path.abspath(prediction_path)}')
    else:
        metrics = compute_metrics(all_preds, all_labels)
        return metrics


def compute_metrics(preds, labels, per_class=False):
    kappa = cohen_kappa_score(labels, preds, weights='quadratic')
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)

    # Calculate and print precision and recall for each class
    if per_class:
        precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
        recall_per_class = recall_score(labels, preds, average=None, zero_division=0)
        return kappa, accuracy, precision, recall, precision_per_class, recall_per_class

    return kappa, accuracy, precision, recall


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Average pooling along channel axis
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Max pooling along channel axis
        x = torch.cat([avg_out, max_out], dim=1)  # Concatenate along channel axis
        x = self.conv(x)  # Learn spatial importance
        return self.sigmoid(x)  # Scale spatial importance
    
    
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scaling parameter

    def forward(self, x):
        batch_size, C, H, W = x.size()  # Input feature map dimensions: (B, C, H, W)

        # Query, Key, and Value transformations
        query = self.query_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)  # Shape: (B, H*W, C//8)
        key = self.key_conv(x).view(batch_size, -1, H * W)  # Shape: (B, C//8, H*W)
        value = self.value_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)  # Shape: (B, H*W, C)

        # Compute attention weights
        attention = torch.bmm(query, key)  # Shape: (B, H*W, H*W)
        attention = F.softmax(attention, dim=-1)  # Normalize attention weights across spatial dimensions

        # Weighted sum of values
        out = torch.bmm(attention, value).permute(0, 2, 1)  # Shape: (B, C, H*W)
        out = out.view(batch_size, C, H, W)  # Reshape back to spatial dimensions

        # Apply learnable scaling and residual connection
        out = self.gamma * out + x
        return out


class MyModel(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.52):
        super().__init__()

        # Load the pretrained VGG16 model
        self.backbone = models.vgg16(pretrained=True)
        
        # Unfreeze all layers
        for param in self.backbone.parameters():
            param.requires_grad = True
            
        self.self_attention = SelfAttention(in_channels=512)

        # Get the input features for the classifier dynamically
        in_features = self.backbone.classifier[0].in_features
        
        # Replace the classifier with a custom one
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Forward pass through the VGG16 backbone
        features = self.backbone.features(x)

        # Apply self-attention
        features = self.self_attention(features)

        # Flatten features and pass through the classifier
        features = features.reshape(features.size(0), -1)  # Flatten
        x = self.backbone.classifier(features)
        return x


class BoostingEnsemble(nn.Module):
    def __init__(self, models, num_classes=5):
        super(BoostingEnsemble, self).__init__()
        self.models = nn.ModuleList(models)  # Convert to ModuleList for proper registration
        self.num_classes = num_classes
        self.boosting_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
        self.is_fitted = False

    def forward(self, x):
        if not isinstance(x, list):
            # For single image input
            model_outputs = []
            for model in self.models:
                output = model(x)
                model_outputs.append(output)
            # Average the predictions from all models
            ensemble_output = torch.mean(torch.stack(model_outputs), dim=0)
            return ensemble_output
        else:
            # For dual image input (if your dataset returns a list of two images)
            model_outputs = []
            for model in self.models:
                # Process each image separately and then combine
                output1 = model(x[0])
                output2 = model(x[1])
                # Average the predictions for the two images
                output = (output1 + output2) / 2
                model_outputs.append(output)
            # Average the predictions from all models
            ensemble_output = torch.mean(torch.stack(model_outputs), dim=0)
            return ensemble_output

    def get_predictions_for_boosting(self, dataloader, device):
        all_preds = []
        all_labels = []
        
        for models in self.models:
            models.eval()
        
        with torch.no_grad():
            for images, labels in dataloader:
                if isinstance(images, list):
                    images = [img.to(device) for img in images]
                else:
                    images = images.to(device)
                
                batch_preds = []
                for model in self.models:
                    outputs = model(images)
                    preds = torch.argmax(outputs, dim=1)
                    batch_preds.append(preds.cpu().numpy())
                
                all_preds.append(np.array(batch_preds).T)
                all_labels.extend(labels.cpu().numpy())
        
        return np.vstack(all_preds), np.array(all_labels)

    def fit_boosting(self, train_loader, device):
        X_train, y_train = self.get_predictions_for_boosting(train_loader, device)
        self.boosting_model.fit(X_train, y_train)
        self.is_fitted = True
    
    def predict_boosting(self, X_test):
        if not self.is_fitted:
            raise RuntimeError("Boosting model must be fitted before making predictions")
        return self.boosting_model.predict(X_test)



def train_model_with_boosting(
    model, train_loader, val_loader, device, criterion, optimizer, lr_scheduler, 
    num_epochs=25, checkpoint_path='model.pth'
):
    best_model = copy.deepcopy(model.state_dict())
    best_epoch = None
    best_val_kappa = -1.0

    # Move all models to device
    model = model.to(device)
    for m in model.models:
        m = m.to(device)

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        running_loss = []
        all_train_preds = []
        all_train_labels = []

        # Training phase
        model.train()
        with tqdm(total=len(train_loader), desc='Training', unit=' batch') as pbar:
            for images, labels in train_loader:
                if isinstance(images, list):
                    images = [img.to(device) for img in images]
                else:
                    images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                preds = torch.argmax(outputs, dim=1)
                all_train_preds.extend(preds.cpu().numpy())
                all_train_labels.extend(labels.cpu().numpy())
                running_loss.append(loss.item())
                
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
                pbar.update(1)

        # Fit boosting model after each epoch
        print("Fitting boosting model...")
        model.fit_boosting(train_loader, device)

        # Validation phase
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                if isinstance(images, list):
                    images = [img.to(device) for img in images]
                else:
                    images = images.to(device)
                    
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.numpy())

        # Calculate metrics
        val_metrics = compute_metrics(val_preds, val_labels)
        val_kappa = val_metrics[0]
        print(f'[Val] Kappa: {val_kappa:.4f}')

        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            best_epoch = epoch
            best_model = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model,
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'best_val_kappa': best_val_kappa
            }, checkpoint_path)

        lr_scheduler.step()

    print(f'Best validation kappa: {best_val_kappa:.4f} (epoch {best_epoch + 1})')
    return model


if __name__ == '__main__':
    # Choose between 'single image' and 'dual images' pipeline
    mode = 'single'  # forward single image to the model each time 
    assert mode in ('single', 'dual')

    # Create base models
    base_models = [MyModel(num_classes=5) for _ in range(5)]
    
    # Create the ensemble model
    ensemble_model = BoostingEnsemble(models=base_models, num_classes=5)
    
    print('Pipeline Mode:', mode)

    # Create datasets
    train_dataset = RetinopathyDataset('./DeepDRiD/train.csv', './DeepDRiD/train/', transform_train, mode)
    val_dataset = RetinopathyDataset('./DeepDRiD/val.csv', './DeepDRiD/val/', transform_test, mode)
    test_dataset = RetinopathyDataset('./DeepDRiD/test.csv', './DeepDRiD/test/', transform_test, mode, test=True)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define the weighted CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()

    # Use GPU device if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)    

    # Move ensemble model to device
    ensemble_model = ensemble_model.to(device)

    # Load pretrained weights if available
    try:
        state_dict = torch.load('./pre/pretrained/vgg16.pth', map_location=device)
        
        # Modify state dict keys to match model structure
        for model in ensemble_model.models:
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = f"backbone.{key}"  # Prefix with 'backbone.'
                new_state_dict[new_key] = value
            
            # Load state dict for each base model
            model.load_state_dict(new_state_dict, strict=False)
            print("Pretrained weights loaded successfully")
    except Exception as e:
        print(f"Could not load pretrained weights: {e}")
        print("Starting with random initialization")

    # Create optimizer for all parameters in the ensemble
    optimizer = torch.optim.Adam(ensemble_model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Train and evaluate the model
    trained_model = train_model_with_boosting(
        ensemble_model,
        train_loader, 
        val_loader,
        device,
        criterion,
        optimizer,
        lr_scheduler=lr_scheduler,
        num_epochs=num_epochs,
        checkpoint_path='./model_vgg_ensemble.pth'
    )

    # Make predictions on testing set
    evaluate_model(trained_model, test_loader, device, test_only=True)
