import os
import kagglehub

# Define the dataset path (or the directory where you want to store the dataset)
dataset_path = "./Bdataset"

# Check if the dataset already exists
if not os.path.exists(dataset_path):
    # Download the dataset
    path = kagglehub.dataset_download("tanlikesmath/diabetic-retinopathy-resized")
    print("Dataset downloaded to:", path)
else:
    print("Dataset already exists at:", dataset_path)
