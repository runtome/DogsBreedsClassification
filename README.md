# Dogs Classification - Multi-Model Comparison

A comprehensive deep learning project for classifying 120 dog breeds using multiple state-of-the-art architectures on the Stanford Dogs Dataset.

## 📋 Overview

This project trains and compares 10 different neural network architectures on the Stanford Dogs dataset:

### Models Implemented
1. **ResNet-18** - Lightweight ResNet variant
2. **ResNet-50** - Standard ResNet architecture
3. **ResNet-101** - Deeper ResNet variant
4. **MobileNetV2** - Efficient mobile architecture
5. **EfficientNet-B0** - Compound scaling efficient model
6. **EfficientNet-B3** - Larger EfficientNet variant
7. **EfficientNet-B7** - Largest EfficientNet variant
8. **ConvNeXt-Tiny** - Modern CNN architecture
9. **ViT-Base** - Vision Transformer
10. **Swin-Tiny** - Shifted Window Transformer

## 📊 Dataset

**Stanford Dogs Dataset**
- **Classes**: 120 dog breeds
- **Images**: 20,580 total
- **Annotations**: Includes bounding boxes for localization
- **Download**: [Kaggle - Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset)

### Expected Directory Structure
```
datasets/
├── images/
│   └── Images/
│       ├── n02085620-Chihuahua/
│       │   ├── n02085620_1007.jpg
│       │   └── ...
│       ├── n02085782-Japanese_spaniel/
│       └── ... (120 breed folders)
└── annotations/
    └── Annotation/
        ├── n02085620-Chihuahua/
        │   ├── n02085620_10074
        │   └── ...
        └── ... (120 breed folders)
```

## 🚀 Installation

### 1. Clone the repository
```bash
git clone <your-repo>
cd stanford-dogs-classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset) and extract to `datasets/` directory.

## 💻 Usage

### Training All Models

Train all models with default settings:
```bash
python train_dogs_classification.py
```

The script will:
- Load and split the dataset (80% train, 20% validation)
- Train each model for 20 epochs
- Save best checkpoints to `model_checkpoints/`
- Save training history to `results/`

### Configuration

Edit the `CONFIG` dictionary in `train_dogs_classification.py`:

```python
CONFIG = {
    'images_dir': 'datasets/images/Images',
    'annotations_dir': 'datasets/annotations/Annotation',
    'batch_size': 32,              # Adjust based on GPU memory
    'num_epochs': 20,              # Number of training epochs
    'learning_rate': 0.001,        # Initial learning rate
    'num_workers': 4,              # Data loading workers
    'image_size': 224,             # Input image size
    'use_bbox': False,             # Use bounding box cropping
    'models_to_train': [...]       # List of models to train
}
```

### Training Individual Models

To train specific models, modify the `models_to_train` list:

```python
'models_to_train': ['resnet18', 'efficientnet_b0']
```

### Visualizing Results

After training, generate comparison visualizations:

```bash
python visualize_results.py
```

This creates:
- `results/comparison_metrics.png` - Bar charts comparing accuracy, F1, and training time
- `results/training_curves.png` - Training/validation curves for all models
- `results/model_efficiency.png` - Efficiency analysis
- `results/results_table.csv` - Detailed metrics table
- `results/model_comparison_report.txt` - Text summary report

### Inference on New Images

**Single Model Prediction:**
```bash
python inference.py --image path/to/dog.jpg --model resnet18 --top_k 5
```

**Compare All Models:**
```bash
python inference.py --image path/to/dog.jpg --compare
```

**Advanced Options:**
```bash
python inference.py \
    --image path/to/dog.jpg \
    --model efficientnet_b3 \
    --checkpoint model_checkpoints/efficientnet_b3_best.pth \
    --top_k 10 \
    --output_dir my_results
```

## 📁 Output Files

### Model Checkpoints
```
model_checkpoints/
├── resnet18_best.pth
├── resnet50_best.pth
├── efficientnet_b0_best.pth
└── ... (one per model)
```

Each checkpoint contains:
- Model weights
- Optimizer state
- Best accuracy
- Training history

### Results
```
results/
├── all_models_results.json          # Complete results summary
├── comparison_metrics.png           # Visual comparisons
├── training_curves.png              # Training progress
├── model_efficiency.png             # Efficiency analysis
├── results_table.csv                # Metrics table
├── model_comparison_report.txt      # Text report
└── {model_name}_history.json        # Per-model training history
```

### Inference Results
```
inference_results/
├── {model}_{image}_prediction.png   # Single predictions
└── comparison_{image}.png           # Multi-model comparison
```

## 🎯 Expected Performance

Typical accuracy ranges (20 epochs, pretrained models):

| Model | Expected Accuracy | Training Time (GPU) |
|-------|------------------|---------------------|
| ResNet-18 | 75-80% | ~30 min |
| ResNet-50 | 80-85% | ~45 min |
| ResNet-101 | 82-87% | ~60 min |
| MobileNetV2 | 72-77% | ~25 min |
| EfficientNet-B0 | 80-85% | ~35 min |
| EfficientNet-B3 | 83-88% | ~50 min |
| EfficientNet-B7 | 85-90% | ~90 min |
| ConvNeXt-Tiny | 83-88% | ~40 min |
| ViT-Base | 82-87% | ~55 min |
| Swin-Tiny | 83-88% | ~45 min |

*Note: Actual performance varies based on hardware, hyperparameters, and random initialization.*

## 🔧 Customization

### Adding New Models

1. Update `create_model()` function in `train_dogs_classification.py`:

```python
elif model_name == 'your_model':
    model = timm.create_model('your_model_name', pretrained=pretrained, num_classes=num_classes)
```

2. Add to `models_to_train` list in CONFIG

### Using Bounding Box Annotations

Enable bounding box cropping to focus on the dog:

```python
CONFIG = {
    'use_bbox': True,  # Enable bounding box cropping
    ...
}
```

### Adjusting Data Augmentation

Modify transforms in the main script:

```python
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    # Add more augmentations here
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

## 📊 Metrics Tracked

For each model:
- **Accuracy**: Overall classification accuracy
- **Precision**: Weighted precision across all classes
- **Recall**: Weighted recall across all classes
- **F1 Score**: Weighted F1 score
- **Training Time**: Total time for all epochs
- **Loss**: Cross-entropy loss

## 🐛 Troubleshooting

### Out of Memory Errors
```python
# Reduce batch size
CONFIG['batch_size'] = 16  # or 8
```

### Slow Training
```python
# Increase workers (if you have multiple CPU cores)
CONFIG['num_workers'] = 8

# Use mixed precision training (add to training loop)
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### Model Download Issues
If pretrained weights fail to download:
```python
# Set pretrained=False and train from scratch
model = create_model(model_name, num_classes=120, pretrained=False)
```

## 📝 Citation

If you use this code, please cite the Stanford Dogs dataset:

```bibtex
@inproceedings{KhoslaYaoJayadevaprakashFeiFei_FGVC2011,
  author = "Aditya Khosla and Nityananda Jayadevaprakash and Bangpeng Yao and Li Fei-Fei",
  title = "Novel Dataset for Fine-Grained Image Categorization",
  booktitle = "First Workshop on Fine-Grained Visual Categorization, IEEE Conference on Computer Vision and Pattern Recognition",
  year = "2011",
  month = "June",
  address = "Colorado Springs, CO",
}
```

## 📄 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Happy Training! 🐕🎉**
