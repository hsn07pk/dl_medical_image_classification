# EyeAI-Retinopathy-Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

## 🔍 Overview

EyeAI is an advanced deep learning system designed to detect and classify diabetic retinopathy from retinal images. Using state-of-the-art transfer learning and ensemble techniques, the system achieves high accuracy in identifying different stages of retinopathy, from mild to proliferative DR.

### Key Features
- Multi-model ensemble learning with ResNet and VGG architectures
- Advanced image preprocessing techniques
- Explainable AI visualizations using GradCAM
- Comprehensive performance metrics including Cohen's Kappa

## 🏗️ Project Structure

```
EyeAI-Retinopathy-Detection/
├── bagging/
│   ├── resnet18Bagging.py
│   ├── resnet34Bagging.py
│   └── vgg16Bagging.py
├── boosting/
│   ├── resnet18Boosting.py
│   ├── resnet34Boosting.py
│   └── vgg16Boosting.py
├── stacking/
│   ├── resnet18Stacking.py
│   ├── resnet34Stacking.py
│   └── vgg16Stacking.py
├── aio.py
├── visualizations/
│   ├── gradcam/
│   └── training_metrics/
├── trained_models/
│   ├── resnet18/
│   ├── resnet34/
│   └── vgg16/
├── requirements.txt
└── README.md
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/EyeAI-Retinopathy-Detection.git
cd EyeAI-Retinopathy-Detection
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Data Preparation

1. Download the DeepDRiD dataset from the provided source
2. Place the dataset in the appropriate directory:
```bash
mkdir data
mv downloaded_dataset data/deepdrid
```

## 🚀 Running the System

### Quick Start with All-in-One Script
```bash
python aio.py
```

### Running Specific Ensemble Methods

#### Bagging
```bash
cd bagging
python resnet18Bagging.py
```

#### Boosting
```bash
cd boosting
python resnet18Boosting.py
```

#### Stacking
```bash
cd stacking
python resnet18Stacking.py
```

## 🔧 Model Configuration

### Available Models
- ResNet18
- ResNet34
- VGG16

### Preprocessing Options
- Ben Graham Preprocessing
- Circle Cropping
- CLAHE
- Gaussian Blur
- Image Sharpening

### Ensemble Techniques
- Bagging
- Boosting
- Stacking

## 📈 Performance Metrics

The system evaluates performance using:
- Cohen's Kappa Score
- Accuracy
- Loss Metrics
- ROC-AUC Curves

## 🎯 Results Visualization

### Training Metrics
```bash
python visualizations/plot_metrics.py
```

### Generate GradCAM Visualizations
```bash
python visualizations/generate_gradcam.py --model resnet18 --image_path path/to/image
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 References

- [DeepDRiD Challenge](https://www.sciencedirect.com/science/article/pii/S2666389922001040)
- [Diabetic Retinopathy Information](https://www.mayoclinic.org/diseases-conditions/diabetic-retinopathy/symptoms-causes/syc-20371611)
- [GradCAM Paper](https://arxiv.org/abs/1610.02391)

## 🙏 Acknowledgments

Special thanks to the DeepDRiD dataset creators and the medical professionals who contributed to the ground truth labeling.

## 📧 Contact

For questions or collaborations, please open an issue in the repository.
