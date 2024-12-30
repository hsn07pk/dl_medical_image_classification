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
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression








class RetinopathyDataset(Dataset):
    def __init__(self, ann_file, image_dir, transform=None, mode='single', test=False):
        self.ann_file = ann_file
        self.image_dir = image_dir
        self.transform = transform
        self.test = test
        self.mode = mode
        self.labels = None  # Initialize labels attribute
        
        if self.mode == 'single':
            self.data = self.load_data()
        else:
            self.data = self.load_data_dual()
            
        # Store labels if not in test mode
        if not self.test:
            if self.mode == 'single':
                self.labels = [d['dr_level'] for d in self.data]
            else:
                self.labels = [d['dr_level'] for d in self.data]

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
    FundRandomRotate(prob=0.5, degree=30),
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










class EnsembleMethods:
    def __init__(self, models, device):
        self.models = models
        self.device = device
        for model in self.models:
            model.eval()

    def get_features(self, dataloader):
        """Extract features from all models for a given dataloader"""
        features_list = []
        labels_list = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                if isinstance(batch, (tuple, list)) and len(batch) == 2:
                    images, labels = batch
                    labels_list.extend(labels.numpy())
                else:
                    images = batch
                
                if not isinstance(images, list):
                    images = images.to(self.device)
                else:
                    images = [x.to(self.device) for x in images]
                
                batch_features = []
                for model in self.models:
                    outputs = model(images)
                    probs = F.softmax(outputs, dim=1)
                    batch_features.append(probs.cpu().numpy())
                
                # Concatenate features from all models
                combined_features = np.concatenate(batch_features, axis=1)
                features_list.append(combined_features)
        
        features = np.vstack(features_list)
        if labels_list:
            return features, np.array(labels_list)
        return features

    def weighted_average(self, dataloader, weights):
        """Weighted average ensemble prediction"""
        all_preds = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Weighted average prediction"):
                if isinstance(batch, (tuple, list)):
                    images = batch[0]
                else:
                    images = batch
                    
                if not isinstance(images, list):
                    images = images.to(self.device)
                else:
                    images = [x.to(self.device) for x in images]
                    
                outputs = [model(images) for model in self.models]
                outputs = [F.softmax(out, dim=1) for out in outputs]
                weighted_outputs = sum(w * out for w, out in zip(weights, outputs))
                preds = torch.argmax(weighted_outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                
        return np.array(all_preds)

    def max_voting(self, dataloader):
        """Max voting ensemble prediction"""
        all_preds = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Max voting prediction"):
                if isinstance(batch, (tuple, list)):
                    images = batch[0]
                else:
                    images = batch
                    
                if not isinstance(images, list):
                    images = images.to(self.device)
                else:
                    images = [x.to(self.device) for x in images]
                    
                outputs = [model(images) for model in self.models]
                preds = [torch.argmax(out, 1) for out in outputs]
                preds = torch.stack(preds, dim=1)
                final_preds = torch.mode(preds, dim=1)[0]
                all_preds.extend(final_preds.cpu().numpy())
                
        return np.array(all_preds)

    def train_stacking(self, train_loader, val_loader):
        """Train stacking ensemble"""
        print("Training stacking ensemble...")
        
        # Get features for training and validation
        X_train, y_train = self.get_features(train_loader)
        X_val, y_val = self.get_features(val_loader)
        
        # Define base models for stacking
        base_models = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ]
        
        # Define meta-classifier
        meta_classifier = LogisticRegression(random_state=42)
        
        # Create and train stacking classifier
        self.stacking_classifier = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_classifier,
            cv=5
        )
        
        self.stacking_classifier.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_preds = self.stacking_classifier.predict(X_val)
        return val_preds

    def train_boosting(self, train_loader, val_loader):
        """Train boosting ensemble"""
        print("Training boosting ensemble...")
        
        # Get features for training and validation
        X_train, y_train = self.get_features(train_loader)
        X_val, y_val = self.get_features(val_loader)
        
        # Create and train boosting classifier
        self.boosting_classifier = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        self.boosting_classifier.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_preds = self.boosting_classifier.predict(X_val)
        return val_preds

    def train_bagging(self, train_loader, val_loader):
        """Train bagging ensemble"""
        print("Training bagging ensemble...")
        
        # Get features for training and validation
        X_train, y_train = self.get_features(train_loader)
        X_val, y_val = self.get_features(val_loader)
        
        # Create and train random forest (which uses bagging)
        self.bagging_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=42
        )
        
        self.bagging_classifier.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_preds = self.bagging_classifier.predict(X_val)
        return val_preds

    def predict_stacking(self, test_loader):
        """Predict using stacking ensemble"""
        X_test = self.get_features(test_loader)
        return self.stacking_classifier.predict(X_test)

    def predict_boosting(self, test_loader):
        """Predict using boosting ensemble"""
        X_test = self.get_features(test_loader)
        return self.boosting_classifier.predict(X_test)

    def predict_bagging(self, test_loader):
        """Predict using bagging ensemble"""
        X_test = self.get_features(test_loader)
        return self.bagging_classifier.predict(X_test)

def evaluate_predictions(y_true, y_pred, method_name):
    """Evaluate predictions using multiple metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'kappa': cohen_kappa_score(y_true, y_pred, weights='quadratic'),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    print(f"\n{method_name} Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    return metrics




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
    
   

# only Last 5 layers unfrozen 
class MyVGG(nn.Module):
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


class MyResnet18(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.52):
        super().__init__()

        self.backbone = models.resnet18(pretrained=True)
        # Get the input features for the classifier dynamically
        in_features = self.backbone.fc.in_features

        for param in self.backbone.parameters():
            param.requires_grad = True

        # Self-attention layer (applied to intermediate feature maps)
        self.self_attention = SelfAttention(in_channels=512)
        self.self_attention3 = SelfAttention(in_channels=256)
        self.self_attention4 = SelfAttention(in_channels=512)

        # Replace the classifier with a custom one
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Extract intermediate feature maps from the backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.self_attention3(x)
        x = self.backbone.layer4(x)
        x = self.self_attention4(x)

        # Apply self-attention to the feature maps
        # x = self.self_attention(x)

        # Apply global average pooling
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten to (B, 512)

        # Pass through the classifier
        x = self.backbone.fc(x)
        return x





class MyResnet34(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.52):
        super().__init__()

        self.backbone = models.resnet34(pretrained=True)
        # Get the input features for the classifier dynamically
        in_features = self.backbone.fc.in_features

        for param in self.backbone.parameters():
            param.requires_grad = True

        # Self-attention layer (applied to intermediate feature maps)
        self.self_attention = SelfAttention(in_channels=512)
        self.self_attention3 = SelfAttention(in_channels=256)
        self.self_attention4 = SelfAttention(in_channels=512)



        # Replace the classifier with a custom one
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Extract intermediate feature maps from the backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        # x = self.backbone.layer3(x)
        # x = self.backbone.layer4(x)

        x = self.backbone.layer3(x)
        x = self.self_attention3(x)

        x = self.backbone.layer4(x)
        x = self.self_attention4(x)

        # Apply self-attention to the feature maps
        # x = self.self_attention(x)

        # Apply global average pooling
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten to (B, 512)

        # Pass through the classifier
        x = self.backbone.fc(x)
        return x



if __name__ == '__main__':
    # Hyper Parameters
    batch_size = 24
    num_classes = 5
    learning_rate = 0.0001
    num_epochs = 15
    mode = 'single'

    print('Pipeline Mode:', mode)

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # Initialize models
    vggModel = MyVGG(num_classes=num_classes)
    resnet18Model = MyResnet18(num_classes=num_classes)
    resnet34Model = MyResnet34(num_classes=num_classes)

    # Create datasets
    train_dataset = RetinopathyDataset('./DeepDRiD/train.csv', './DeepDRiD/train/', transform_train, mode)
    val_dataset = RetinopathyDataset('./DeepDRiD/val.csv', './DeepDRiD/val/', transform_test, mode)
    test_dataset = RetinopathyDataset('./DeepDRiD/test.csv', './DeepDRiD/test/', transform_test, mode, test=True)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load pretrained weights
    try:
        vggModel.load_state_dict(torch.load('./model_vgg.pth', map_location=device))
        resnet18Model.load_state_dict(torch.load('./model_resnet18.pth', map_location=device))
        resnet34Model.load_state_dict(torch.load('./model_resnet34.pth', map_location=device))
        print("Loaded pretrained weights successfully")
    except FileNotFoundError:
        print("No pretrained weights found. Please train the models first.")
        sys.exit(1)

    # Move models to device
    models = [vggModel.to(device), resnet18Model.to(device), resnet34Model.to(device)]
    
    try:
        # Initialize ensemble methods
        ensemble = EnsembleMethods(models, device)
        val_labels = np.array(val_dataset.labels)
        
        # 1. Weighted Average
        print("\nEvaluating Weighted Average Ensemble...")
        weights = [0.3, 0.5, 0.2]  # Adjust based on individual model performance
        weighted_preds = ensemble.weighted_average(val_loader, weights)
        weighted_metrics = evaluate_predictions(val_labels, weighted_preds, "Weighted Average")
        
        # 2. Max Voting
        print("\nEvaluating Max Voting Ensemble...")
        max_voting_preds = ensemble.max_voting(val_loader)
        max_voting_metrics = evaluate_predictions(val_labels, max_voting_preds, "Max Voting")
        
        # 3. Stacking
        print("\nEvaluating Stacking Ensemble...")
        stacking_preds = ensemble.train_stacking(train_loader, val_loader)
        stacking_metrics = evaluate_predictions(val_labels, stacking_preds, "Stacking")
        
        # 4. Boosting
        print("\nEvaluating Boosting Ensemble...")
        boosting_preds = ensemble.train_boosting(train_loader, val_loader)
        boosting_metrics = evaluate_predictions(val_labels, boosting_preds, "Boosting")
        
        # 5. Bagging
        print("\nEvaluating Bagging Ensemble...")
        bagging_preds = ensemble.train_bagging(train_loader, val_loader)
        bagging_metrics = evaluate_predictions(val_labels, bagging_preds, "Bagging")
        
        # Generate test predictions
        print("\nGenerating test predictions...")
        test_predictions = {
            'weighted': ensemble.weighted_average(test_loader, weights),
            'max_voting': ensemble.max_voting(test_loader),
            'stacking': ensemble.predict_stacking(test_loader),
            'boosting': ensemble.predict_boosting(test_loader),
            'bagging': ensemble.predict_bagging(test_loader)
        }
        
        # Save all predictions
        for method, preds in test_predictions.items():
            df = pd.DataFrame({
                'ID': [os.path.basename(test_dataset.data[i]['img_path']) for i in range(len(preds))],
                'TARGET': preds
            })
            df.to_csv(f'./{method}_predictions.csv', index=False)
            print(f"Saved {method} predictions")
        
        # Save ensemble metrics
        metrics_df = pd.DataFrame({
            'Weighted': weighted_metrics,
            'MaxVoting': max_voting_metrics,
            'Stacking': stacking_metrics,
            'Boosting': boosting_metrics,
            'Bagging': bagging_metrics
        })
        metrics_df.to_csv('./ensemble_metrics.csv')
        print("\nSaved ensemble metrics to ensemble_metrics.csv")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Traceback:")
        import traceback
        traceback.print_exc()