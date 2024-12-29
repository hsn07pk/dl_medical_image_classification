import copy
import os
import random
import sys
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
import cv2
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from visualization_vgg import visualize_and_explain
from torchvision.transforms.functional import adjust_gamma 

# Configuration dictionary for easy selection
CONFIG = {
    'models': {
        'vgg16': False,
        'resnet18': False,
        'resnet34': True
    },
    'ensemble_methods': {
        'stacking': False,
        'boosting': False,
        'weighted_average': False,
        'max_voting': True,
        'bagging': False
    },
    'preprocessing': {
        'ben_graham': True,
        'circle_crop': True,
        'clahe': True,
        'gaussian_blur': True,
        'sharpen': True
    }
}

# Hyper Parameters
batch_size = 24
num_classes = 5
learning_rate = 0.0001
num_epochs = 25

class ImagePreprocessor:
    @staticmethod
    def ben_graham_preprocessing(image):
        image = np.array(image)
        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), 10), -4, 128)
        return Image.fromarray(image)
    
    @staticmethod
    def circle_crop(image):
        image = np.array(image)
        height, width = image.shape[:2]
        mask = np.zeros((height, width), np.uint8)
        cv2.circle(mask, (width//2, height//2), min(width, height)//2, (255, 255, 255), -1)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        return Image.fromarray(masked_image)
    
    @staticmethod
    def apply_clahe(image):
        image = np.array(image)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(image)
    
    @staticmethod
    def gaussian_blur(image):
        image = np.array(image)
        blurred = cv2.GaussianBlur(image, (5,5), 0)
        return Image.fromarray(blurred)
    
    @staticmethod
    def sharpen(image):
        image = np.array(image)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return Image.fromarray(sharpened)

class PreprocessingPipeline:
    def __init__(self, config):
        self.config = config
        self.preprocessor = ImagePreprocessor()
    
    def process_image(self, image):
        if self.config['ben_graham']:
            image = self.preprocessor.ben_graham_preprocessing(image)
        if self.config['circle_crop']:
            image = self.preprocessor.circle_crop(image)
        if self.config['clahe']:
            image = self.preprocessor.apply_clahe(image)
        if self.config['gaussian_blur']:
            image = self.preprocessor.gaussian_blur(image)
        if self.config['sharpen']:
            image = self.preprocessor.sharpen(image)
        return image

class RetinopathyDataset(Dataset):
    def __init__(self, ann_file, image_dir, transform=None, mode='single', test=False, preprocessing_config=None):
        self.ann_file = ann_file
        self.image_dir = image_dir
        self.transform = transform
        self.test = test
        self.mode = mode
        self.preprocessing_pipeline = PreprocessingPipeline(preprocessing_config) if preprocessing_config else None

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
        
        if self.preprocessing_pipeline:
            img = self.preprocessing_pipeline.process_image(img)
            
        if self.transform:
            img = self.transform(img)

        if not self.test:
            label = torch.tensor(data['dr_level'], dtype=torch.int64)
            return img, label
        else:
            return img

    def load_data_dual(self):
        df = pd.read_csv(self.ann_file)
        df['prefix'] = df['image_id'].str.split('_').str[0]
        df['suffix'] = df['image_id'].str.split('_').str[1].str[0]
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

        if self.preprocessing_pipeline:
            img1 = self.preprocessing_pipeline.process_image(img1)
            img2 = self.preprocessing_pipeline.process_image(img2)

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
    
class GammaCorrection:
    def __init__(self, gamma=1.5):
        self.gamma = gamma
    
    def __call__(self, img):
        return adjust_gamma(img, gamma=self.gamma)

transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((210, 210)),
    SLORandomPad((224, 224)),
    FundRandomRotate(prob=0.5, degree=30),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=(0.1, 0.9)),
    GammaCorrection(gamma=1.5),
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
    best_model_state = None
    best_val_kappa = -1.0
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'train_kappa': [],
        'val_kappa': [],
        'learning_rates': []
    }
    
    for epoch in range(1, num_epochs + 1):
        print(f'\nEpoch {epoch}/{num_epochs}')
        model.train()
        running_loss = []
        all_preds = []
        all_labels = []

        with tqdm(total=len(train_loader), desc=f'Training', unit=' batch') as pbar:
            for batch_idx, (images, labels) in enumerate(train_loader):
                try:
                    if isinstance(images, (list, tuple)):
                        images = [img.to(device) for img in images]
                    else:
                        images = images.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    preds = torch.argmax(outputs, 1)
                    running_loss.append(loss.item())
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'lr': f'{optimizer.param_groups[0]["lr"]:.1e}'
                    })
                    pbar.update(1)

                except Exception as e:
                    print(f"Error in batch {batch_idx}: {str(e)}")
                    continue

        # Calculate training metrics
        epoch_loss = np.mean(running_loss)
        train_metrics = compute_metrics(all_preds, all_labels)
        training_history['train_loss'].append(epoch_loss)
        training_history['train_accuracy'].append(train_metrics[1])
        training_history['train_kappa'].append(train_metrics[0])
        training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        # Validation phase
        model.eval()
        val_metrics = evaluate_model(model, val_loader, device)
        val_kappa = val_metrics[0]
        
        # Update validation history
        training_history['val_loss'].append(val_metrics[1])
        training_history['val_accuracy'].append(val_metrics[2])
        training_history['val_kappa'].append(val_kappa)

        # Print epoch results
        print(f'\nEpoch {epoch} Results:')
        print(f'Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_metrics[1]:.4f}, Train Kappa: {train_metrics[0]:.4f}')
        print(f'Val Loss: {val_metrics[1]:.4f}, Val Accuracy: {val_metrics[2]:.4f}, Val Kappa: {val_kappa:.4f}')

        # Step the scheduler with validation kappa score
        lr_scheduler.step(val_kappa)

        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_kappa': best_val_kappa,
                'training_history': training_history
            }, checkpoint_path)
            print(f'Saved new best model with validation kappa: {val_kappa:.4f}')

    # Load best model
    model.load_state_dict(best_model_state)
    return model, training_history






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
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Ensure x is 4D (batch_size, channels, height, width)
        if len(x.shape) == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)  # Reshaping 2D to 4D if necessary

        batch_size, C, H, W = x.size()

        query = self.query_conv(x).view(batch_size, -1, H * W).permute(0, 2, 1)  # (B, H*W, C//8)
        key = self.key_conv(x).view(batch_size, -1, H * W)  # (B, H*W, C//8)
        value = self.value_conv(x).view(batch_size, -1, H * W)  # (B, H*W, C)

        energy = torch.bmm(query, key)  # (B, H*W, H*W)
        attention = torch.softmax(energy, dim=-1)  # (B, H*W, H*W)

        out = torch.bmm(value, attention.permute(0, 2, 1))  # (B, C, H*W)
        out = out.view(batch_size, C, H, W)  # (B, C, H, W)

        out = self.gamma * out + x
        return out

class ModelFactory:
    @staticmethod
    def create_model(model_name, num_classes=5):
        if model_name == 'vgg16':
            model = MyModel(backbone='vgg16')
        elif model_name == 'resnet18':
            model = MyModel(backbone='resnet18')
        elif model_name == 'resnet34':
            model = MyModel(backbone='resnet34')
        else:
            raise ValueError(f"Unknown model: {model_name}")
        return model

class MyModel(nn.Module):
    def __init__(self, backbone='vgg16', num_classes=5, dropout_rate=0.5):
        super().__init__()
        
        # Initialize backbone with pretrained weights
        if backbone == 'vgg16':
            base_model = models.vgg16(pretrained=True)
            self.backbone = nn.Sequential(*list(base_model.features))
            self.fc_input_features = 512 * 7 * 7
        elif backbone == 'resnet18':
            base_model = models.resnet18(pretrained=True)
            layers = list(base_model.children())[:-1]
            self.backbone = nn.Sequential(*layers)
            self.fc_input_features = 512
        elif backbone == 'resnet34':
            base_model = models.resnet34(pretrained=True)
            layers = list(base_model.children())[:-1]
            self.backbone = nn.Sequential(*layers)
            self.fc_input_features = 512
            
        # Initialize attention modules
        self.spatial_attention = SpatialAttention()
        self.self_attention = SelfAttention(in_channels=512)
        
        # Initialize classifier with track_running_stats=True for batch norm
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fc_input_features, 512),
            nn.BatchNorm1d(512, track_running_stats=True),  # Explicitly enable tracking
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256, track_running_stats=True),  # Explicitly enable tracking
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Handle both training and evaluation modes
        if self.training:
            # Training mode - regular forward pass
            x = self.backbone(x)
            sa_out = self.self_attention(x)
            spa_out = self.spatial_attention(x)
            x = x * spa_out + sa_out
            x = self.classifier(x)
        else:
            # Evaluation mode - use moving averages for batch norm
            with torch.no_grad():
                self.eval()  # Ensure eval mode
                x = self.backbone(x)
                sa_out = self.self_attention(x)
                spa_out = self.spatial_attention(x)
                x = x * spa_out + sa_out
                x = self.classifier(x)
        
        return x

    def set_gradcam_mode(self):
        """Special mode for GradCAM visualization"""
        self.eval()  # Set to evaluation mode
        # Enable gradients for feature extraction even in eval mode
        for param in self.backbone.parameters():
            param.requires_grad = True
        return self.backbone[-1]  # Return the last c

class EnsembleModel(nn.Module):
    def __init__(self, models, ensemble_methods, device, num_classes=5):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.ensemble_methods = ensemble_methods
        self.num_classes = num_classes
        self.device = device
        
        # Initialize learnable weights for weighted voting
        self.model_weights = nn.Parameter(torch.ones(len(models)) / len(models))
        
        # Initialize meta classifier for stacking
        self.meta_classifier = nn.Sequential(
            nn.Linear(len(models) * num_classes, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        ).to(device)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = [img.to(self.device) for img in x]
        else:
            x = x.to(self.device)

        all_outputs = []
        all_probs = []
        
        with torch.set_grad_enabled(self.training):
            for model in self.models:
                model.train(self.training)
                outputs = model(x)
                probs = F.softmax(outputs, dim=-1)
                all_outputs.append(outputs)
                all_probs.append(probs)
            
            stacked_outputs = torch.stack(all_outputs)
            stacked_probs = torch.stack(all_probs)

            if self.ensemble_methods.get('max_voting', False):
                # Weighted voting using softmax probabilities
                weighted_probs = stacked_probs * F.softmax(self.model_weights.view(-1, 1, 1), dim=0)
                final_probs = weighted_probs.sum(dim=0)
                return torch.log(final_probs + 1e-8)  # Add small epsilon to prevent log(0)
            
            elif self.ensemble_methods.get('stacking', False):
                meta_features = stacked_outputs.permute(1, 0, 2).reshape(
                    x[0].size(0) if isinstance(x, list) else x.size(0), -1
                )
                return self.meta_classifier(meta_features)
            
            else:  # Default to average
                return torch.mean(stacked_outputs, dim=0)
        
def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size):
    # Create DataLoaders with proper worker initialization
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def main():
    # Set device and seed for reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(42)
    random.seed(42)
    
    print(f"Using device: {device}")
    
    # Create datasets with selected preprocessing
    preprocessing_config = CONFIG['preprocessing']
    
    train_dataset = RetinopathyDataset(
        './DeepDRiD/train.csv',
        './DeepDRiD/train/',
        transform_train,
        preprocessing_config=preprocessing_config
    )
    
    val_dataset = RetinopathyDataset(
        './DeepDRiD/val.csv',
        './DeepDRiD/val/',
        transform_test,
        preprocessing_config=preprocessing_config
    )
    
    test_dataset = RetinopathyDataset(
        './DeepDRiD/test.csv',
        './DeepDRiD/test/',
        transform_test,
        preprocessing_config=preprocessing_config,
        test=True
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size
    )
    
    # Initialize models with different random seeds
    models = []
    model_names = []
    
    if CONFIG['models']['vgg16']:
        torch.manual_seed(42)  # Different seed for each model
        vgg_model = MyModel(backbone='vgg16').to(device)
        models.append(vgg_model)
        model_names.append('vgg16')
        
    if CONFIG['models']['resnet18']:
        torch.manual_seed(43)
        resnet18_model = MyModel(backbone='resnet18').to(device)
        models.append(resnet18_model)
        model_names.append('resnet18')
        
    if CONFIG['models']['resnet34']:
        torch.manual_seed(44)
        resnet34_model = MyModel(backbone='resnet34').to(device)
        models.append(resnet34_model)
        model_names.append('resnet34')
    
    # Create ensemble model
    ensemble = EnsembleModel(models, CONFIG['ensemble_methods'], device).to(device)
    
    # Training setup with weighted loss
    class_weights = torch.tensor([1.0, 2.0, 2.0, 2.0, 2.0]).to(device)  # Adjust weights based on class distribution
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer with different parameter groups
    optimizer_grouped_parameters = [
        {'params': model.parameters(), 'lr': learning_rate} for model in models
    ]
    optimizer_grouped_parameters.append({
        'params': ensemble.meta_classifier.parameters(), 
        'lr': learning_rate * 0.1  # Lower learning rate for meta classifier
    })
    
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=0.1,
        patience=5,
        verbose=True,
        min_lr=1e-7
    )
    
    # Generate unique run identifier
    model_str = '_'.join(model_names)
    ensemble_str = '_'.join([k for k, v in CONFIG['ensemble_methods'].items() if v])
    preprocess_str = '_'.join([k for k, v in preprocessing_config.items() if v])
    run_id = f"{model_str}_{ensemble_str}_{preprocess_str}"
    
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('predictions', exist_ok=True)
    visualization_dir = f'visualizations/{run_id}'
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Train model
    checkpoint_path = f'checkpoints/model_{run_id}.pth'
    
    ensemble, training_history = train_model(
        ensemble, train_loader, val_loader, device,
        criterion, optimizer, lr_scheduler,
        num_epochs=num_epochs,
        checkpoint_path=checkpoint_path
    )
    
    # Generate predictions
    prediction_path = f'predictions/pred_{run_id}.csv'
    test_metrics = evaluate_model(
        ensemble, test_loader, device,
        test_only=True,
        prediction_path=prediction_path
    )
    
    # Save visualization results
    visualize_and_explain(
        model=ensemble,
        dataloader=val_loader,
        device=device,
        num_epochs=num_epochs,
        training_history=training_history,
        save_dir=visualization_dir
    )
    
    print(f"\nTraining completed!")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Predictions saved to: {prediction_path}")
    print(f"Visualizations saved to: {visualization_dir}")

if __name__ == '__main__':
    main()