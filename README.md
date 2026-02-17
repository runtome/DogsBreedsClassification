# Stanford Dogs Classification - Multi-Model Comparison

[![GitHub](https://img.shields.io/badge/GitHub-runtome%2FDogsBreedsClassification-blue?logo=github)](https://github.com/runtome/DogsBreedsClassification)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-orange)](https://pytorch.org/)
[![timm](https://img.shields.io/badge/timm-1.0.20-green)](https://github.com/huggingface/pytorch-image-models)

A comprehensive deep learning project for classifying **120 dog breeds** across **20,580 images** using 8 state-of-the-art architectures on the Stanford Dogs Dataset.

---

## 🏆 Benchmark Results (10 Epochs, Tesla P100 16GB)

| Rank | Model | Best Acc | Val F1 | Time (min) | Type |
|------|-------|----------|--------|------------|------|
| 🥇 1 | **ViT-Base** | **86.15%** | **83.99%** | 188.3 | Vision Transformer |
| 🥈 2 | **Swin-Tiny** | **80.76%** | **80.60%** | 25.7 | Window Transformer |
| 🥉 3 | **ConvNeXt-Tiny** | **80.52%** | **80.47%** | 64.4 | Modern CNN |
| 4 | EfficientNet-B3 | 80.39% | 80.24% | 24.0 | Compound CNN |
| 5 | ResNet-50 | 78.21% | 78.08% | 20.3 | Classic CNN |
| 6 | EfficientNet-B0 | 72.04% | 71.89% | 14.8 | Compound CNN |
| 7 | MobileNetV2 | 69.73% | 69.22% | 12.0 | Lightweight CNN |
| 8 | ResNet-18 | 69.10% | 69.06% | 11.8 | Classic CNN |

> **Best Accuracy:** ViT-Base at **86.15%**  
> **Best Efficiency:** Swin-Tiny — **80.76% in only 25.7 minutes** (best accuracy/speed trade-off)  
> **Tightest Cluster:** Swin-Tiny, ConvNeXt-Tiny, EfficientNet-B3 all within **0.4%** of each other

---

## 📋 Overview

This project benchmarks 8 neural network architectures using **model-specific optimized hyperparameters**. The most critical finding: Vision Transformers require fundamentally different training settings than CNNs. Using the same settings causes transformers to produce only ~1–6% accuracy.

### Models with Best Pre-trained Weights (timm)

| Model | Pre-trained Weights | ImageNet Acc | Optimizer | LR |
|-------|-------------------|--------------|-----------|-----|
| ResNet-18 | `resnet18.a1_in1k` | 70.6% | Adam | 0.001 |
| ResNet-50 | `resnet50.a1_in1k` | 80.4% | Adam | 0.001 |
| MobileNetV2 | `mobilenetv2_120d.ra_in1k` | 72.2% | Adam | 0.001 |
| EfficientNet-B0 | `efficientnet_b0.ra_in1k` | 77.7% | AdamW | 0.0005 |
| EfficientNet-B3 | `efficientnet_b3.ra2_in1k` | 82.1% | AdamW | 0.0003 |
| ConvNeXt-Tiny | `convnext_tiny.fb_in22k_ft_in1k` | 82.9% | AdamW | 0.0003 |
| **ViT-Base** | `vit_base_patch16_224.augreg2_in21k_ft_in1k` | 85.5% | **AdamW** | **0.00005** |
| **Swin-Tiny** | `swin_tiny_patch4_window7_224.ms_in22k_ft_in1k` | 82.2% | **AdamW** | **0.0001** |

---

## 📊 Dataset

**Stanford Dogs Dataset**
- **Classes**: 120 dog breeds
- **Total Images**: 20,580
- **Split**: 80% train (16,464 images) / 20% validation (4,116 images)
- **Annotations**: Class labels + bounding boxes
- **Download**: [Kaggle - Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset)

### Directory Structure
```
datasets/
├── images/Images/
│   ├── n02085620-Chihuahua/
│   │   ├── n02085620_1007.jpg
│   │   └── ...
│   └── ... (120 breed folders)
└── annotations/Annotation/
    ├── n02085620-Chihuahua/
    └── ... (120 breed folders)
```

---

## 🔑 Key Insight: Model-Specific Training Configs

> **The most important lesson:** CNNs and Transformers need different training strategies!

| Setting | CNNs (ResNet, EfficientNet) | Transformers (ViT, Swin) |
|---------|----------------------------|--------------------------|
| Learning Rate | `0.001` | `0.00005 – 0.0001` (10–20× lower!) |
| Optimizer | Adam | **AdamW** (required) |
| LR Warmup | None | **5–8 epochs** (critical!) |
| Scheduler | StepLR | **Cosine Annealing** |
| Grad Clipping | Not needed | **1.0** (prevents explosion) |
| Weight Decay | `1e-4` | **`0.05`** (500× higher) |

Without these transformer-specific settings, ViT-Base achieves only ~6.5% and Swin-Tiny ~1.2%.

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone <your-repo>
cd stanford-dogs-classification

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage

### Train All Models
```bash
python train_dogs_classification.py
```

The script automatically:
- Loads and splits the dataset (80/20 train/val)
- Applies model-specific optimized configurations
- Uses warmup + cosine annealing for transformers
- Saves best checkpoints to `model_checkpoints/`
- Logs training history to `results/`

### Configuration (in `train_dogs_classification.py`)
```python
CONFIG = {
    'batch_size': 16,       # Optimized for 16GB GPU
    'num_epochs': 20,
    'num_workers': 2,
    'image_size': 224,
    'models_to_train': [
        'resnet18', 'mobilenetv2', 'resnet50',
        'efficientnet_b0', 'efficientnet_b3',
        'convnext_tiny', 'vit_base', 'swin_tiny'
    ]
}
```

### Visualize Results
```bash
python visualize_results.py
```

Generates:
- `results/comparison_metrics.png` — Accuracy & F1 bar charts
- `results/training_curves.png` — Train/validation curves for all models
- `results/model_efficiency.png` — Accuracy per training minute
- `results/results_table.csv` — Full metrics table
- `results/model_comparison_report.txt` — Text summary report

### Inference on New Images
```bash
# Single model prediction
python inference.py --image dog.jpg --model vit_base --top_k 5

# Compare all trained models
python inference.py --image dog.jpg --compare
```

---

## 📁 Output Structure

```
model_checkpoints/
├── resnet18_best.pth
├── vit_base_best.pth
└── ...

results/
├── all_models_results.json
├── comparison_metrics.png
├── training_curves.png
├── model_efficiency.png
├── results_table.csv
└── model_comparison_report.txt
```

---

## 📈 Analysis & Findings

### Accuracy Ranking
ViT-Base leads at **86.15%**, with Swin-Tiny, ConvNeXt-Tiny, and EfficientNet-B3 forming a tight cluster around 80%. Classic CNNs (ResNet) trail but train much faster.

### Efficiency (Accuracy ÷ Training Time)

| Model | Efficiency Score | Notes |
|-------|-----------------|-------|
| **Swin-Tiny** | **0.0314** | 🏆 Best trade-off overall |
| EfficientNet-B3 | 0.0334 | Strong second choice |
| EfficientNet-B0 | 0.0487 | Fast, decent accuracy |
| ViT-Base | 0.0046 | Highest accuracy, lowest efficiency |

### Key Observations

**ViT-Base** converges very fast (best accuracy at epoch 2 in early tests), then stabilizes with longer warmup. It has the highest capacity and benefits most from ImageNet-21k pre-training.

**Swin-Tiny** shows the most stable, monotonically improving training curve. It achieves ViT-level performance in 7× less training time.

**EfficientNet-B3** and **ConvNeXt-Tiny** achieve similar accuracy (~80%) but ConvNeXt takes 2.7× longer. EfficientNet-B3 is the better choice for this dataset size.

**Classic CNNs** (ResNet-18/50) underperform modern architectures by 7–17%, confirming that architectural advances translate to real gains.

---

## 🔧 Customization

### Adding New Models
```python
# In create_model():
elif model_name == 'your_model':
    model = timm.create_model('timm_name', pretrained=True, num_classes=120)

# In CONFIG model_configs:
'your_model': {
    'lr': 0.0001, 'optimizer': 'adamw', 'warmup': 3,
    'wd': 0.05, 'scheduler': 'cosine', 'grad_clip': 1.0
}
```

### Adjusting Augmentation
```python
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `batch_size` to `8` |
| ViT/Swin not learning | Check `lr ≤ 0.0001`, `warmup ≥ 5`, `optimizer = adamw` |
| Loss is NaN | Enable gradient clipping: `grad_clip: 1.0` |
| Slow training | Increase `num_workers`, use `pin_memory=True` |

---

## 📊 Metrics Tracked Per Model

- **Accuracy** (train + validation per epoch)
- **Precision / Recall / F1** (weighted)
- **Cross-Entropy Loss** (train + validation)
- **Learning Rate** schedule
- **Total Training Time**

---

## 🖥️ Environment

| Component | Version |
|-----------|---------|
| GPU | NVIDIA Tesla P100-PCIE 16GB |
| CUDA | 13.0 (Driver 580.105) |
| PyTorch | 2.8.0+cu126 |
| timm | 1.0.20 |
| Batch Size | 16 |
| Epochs | 10 |

---

## 📝 Citation

If you use this code or dataset, please cite:

### Dataset — Stanford Dogs
```bibtex
@inproceedings{KhoslaYaoJayadevaprakashFeiFei_FGVC2011,
  author    = {Aditya Khosla and Nityananda Jayadevaprakash and Bangpeng Yao and Li Fei-Fei},
  title     = {Novel Dataset for Fine-Grained Image Categorization},
  booktitle = {First Workshop on Fine-Grained Visual Categorization,
               IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2011},
  month     = {June},
  address   = {Colorado Springs, CO}
}
```

### Dataset — ImageNet (source of annotations)
```bibtex
@inproceedings{imagenet_cvpr09,
  author    = {Deng, J. and Dong, W. and Socher, R. and Li, L.-J. and Li, K. and Fei-Fei, L.},
  title     = {ImageNet: A Large-Scale Hierarchical Image Database},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2009}
}
```

### Dataset Links
- **Official**: http://vision.stanford.edu/aditya86/ImageNetDogs/
- **Kaggle**: https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset

### This Repository
```
GitHub: https://github.com/runtome/DogsBreedsClassification
```

---

## 📄 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or pull requests.

---

**Happy Training! 🐕🚀**