import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from torch.nn import functional as F
import seaborn as sns
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.features = None
        
        # Register hooks
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.features = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        # Register the hooks
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_backward_hook(backward_hook))
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
    
    def generate_cam(self, input_image, target_class=None):
        # Forward pass
        model_output = self.model(input_image)
        
        if target_class is None:
            target_class = torch.argmax(model_output)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        model_output[0, target_class].backward()
        
        # Get weights
        gradients = self.gradients.detach().cpu()
        features = self.features.detach().cpu()
        
        weights = torch.mean(gradients, dim=(2, 3))[0, :]
        
        # Generate CAM
        cam = torch.zeros(features.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * features[0, i, :, :]
        
        cam = F.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)
        
        return cam.numpy()

def visualize_and_explain(model, dataloader, device, num_epochs, training_history, save_dir='./visualizations/'):
    """
    Comprehensive function for visualization and explainable AI analysis
    
    Parameters:
    - model: trained PyTorch model
    - dataloader: DataLoader containing images to analyze
    - device: torch device (cuda/cpu)
    - num_epochs: number of epochs model was trained for
    - training_history: dict containing 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'
    - save_dir: directory to save visualizations
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Plot Training History
    def plot_training_metrics():
        plt.figure(figsize=(15, 5))
        
        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.plot(range(1, num_epochs + 1), training_history['train_loss'], label='Training Loss')
        plt.plot(range(1, num_epochs + 1), training_history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(range(1, num_epochs + 1), training_history['train_accuracy'], label='Training Accuracy')
        plt.plot(range(1, num_epochs + 1), training_history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/training_history.png')
        plt.close()
    
    # 2. Generate GradCAM Visualizations
    def generate_gradcam_visualizations():
        model.eval()
        
        # For VGG-like models, use the last conv layer
        if hasattr(model, 'backbone'):
            target_layer = model.backbone.features[-1]
        else:
            # Adjust this based on your model architecture
            target_layer = list(model.modules())[-3]
        
        gradcam = GradCAM(model, target_layer)
        
        # Get a batch of images
        images, labels = next(iter(dataloader))
        if not isinstance(images, list):
            images = images.to(device)
        else:
            images = [img.to(device) for img in images]
        
        # Process first 5 images
        for idx in range(min(5, len(images))):
            if isinstance(images, list):
                image = images[0][idx:idx+1]  # Take first image of pair
            else:
                image = images[idx:idx+1]
            
            # Generate CAM
            cam = gradcam.generate_cam(image)
            
            # Convert tensor to numpy image
            orig_img = image[0].cpu().numpy().transpose(1, 2, 0)
            orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min())
            
            # Resize CAM to match image size
            cam_resized = cv2.resize(cam, (orig_img.shape[1], orig_img.shape[0]))
            
            # Create heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            
            # Combine original image and heatmap
            cam_img = 0.7 * orig_img + 0.3 * heatmap
            
            # Save visualization
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.imshow(orig_img)
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(cam_resized, cmap='jet')
            plt.title('GradCAM Heatmap')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(cam_img)
            plt.title('Combined Visualization')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/gradcam_visualization_{idx}.png')
            plt.close()
        
        gradcam.remove_hooks()
    
    # Execute visualizations
    plot_training_metrics()
    generate_gradcam_visualizations()
    
    print(f"Visualizations saved to {save_dir}")