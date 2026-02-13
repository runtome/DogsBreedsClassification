"""
Stanford Dogs Dataset Classification - Multi-Model Comparison
Train and compare different architectures on the Stanford Dogs dataset
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import timm
from PIL import Image
import xml.etree.ElementTree as ET
from pathlib import Path
import time
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

# ==================== Dataset Class ====================
class StanfordDogsDataset(Dataset):
    """Stanford Dogs Dataset with annotation support"""
    
    def __init__(self, images_dir, annotations_dir=None, transform=None, use_bbox=False):
        """
        Args:
            images_dir: Path to images directory
            annotations_dir: Path to annotations directory (optional)
            transform: torchvision transforms
            use_bbox: Whether to crop images using bounding boxes
        """
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir) if annotations_dir else None
        self.transform = transform
        self.use_bbox = use_bbox
        
        # Collect all image paths and labels
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load all images and create class mappings"""
        breed_folders = sorted([d for d in self.images_dir.iterdir() if d.is_dir()])
        
        for idx, breed_folder in enumerate(breed_folders):
            breed_name = breed_folder.name.split('-', 1)[1] if '-' in breed_folder.name else breed_folder.name
            self.class_to_idx[breed_name] = idx
            self.idx_to_class[idx] = breed_name
            
            # Get all images in this breed folder
            for img_path in breed_folder.glob('*.jpg'):
                self.samples.append((str(img_path), idx, breed_folder.name))
        
        print(f"Loaded {len(self.samples)} images from {len(self.class_to_idx)} classes")
    
    def _parse_annotation(self, img_name, breed_folder):
        """Parse XML annotation to get bounding box"""
        if not self.annotations_dir:
            return None
        
        # Remove .jpg extension
        annotation_name = img_name.replace('.jpg', '')
        annotation_path = self.annotations_dir / breed_folder / annotation_name
        
        if not annotation_path.exists():
            return None
        
        try:
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            
            # Get bounding box
            obj = root.find('object')
            if obj is not None:
                bbox = obj.find('bndbox')
                if bbox is not None:
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    return (xmin, ymin, xmax, ymax)
        except Exception as e:
            print(f"Error parsing annotation {annotation_path}: {e}")
        
        return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, breed_folder = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Crop using bounding box if available
        if self.use_bbox and self.annotations_dir:
            img_name = Path(img_path).name
            bbox = self._parse_annotation(img_name, breed_folder)
            if bbox:
                xmin, ymin, xmax, ymax = bbox
                image = image.crop((xmin, ymin, xmax, ymax))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ==================== Model Factory ====================
def create_model(model_name, num_classes=120, pretrained=True):
    """
    Create model architecture
    
    Supported models:
    - resnet18, resnet50, resnet101
    - mobilenetv2
    - efficientnet_b0, efficientnet_b3, efficientnet_b7
    - convnext_tiny
    - vit_base
    - swin_tiny
    """
    print(f"Creating model: {model_name}")
    
    if model_name == 'resnet18':
        model = timm.create_model('resnet18.a1_in1k', pretrained=pretrained, num_classes=num_classes)
    
    elif model_name == 'resnet50':
        model = timm.create_model('resnet50d.a1_in1k', pretrained=pretrained, num_classes=num_classes)
    
    elif model_name == 'resnet101':
        model = timm.create_model('resnext101_32x32d.fb_wsl_ig1b_ft_in1k', pretrained=pretrained, num_classes=num_classes)
    
    elif model_name == 'mobilenetv2':
        model = timm.create_model('mobilenetv2_100.ra_in1k', pretrained=pretrained, num_classes=num_classes)
    
    elif model_name == 'efficientnet_b0':
        model = timm.create_model('efficientnet_b0.ra4_e3600_r224_in1k', pretrained=pretrained, num_classes=num_classes)
    
    elif model_name == 'efficientnet_b3':
        model = timm.create_model('efficientnet_b3.ra2_in1k', pretrained=pretrained, num_classes=num_classes)
    
    elif model_name == 'efficientnet_b7':
        model = timm.create_model('tf_efficientnet_b7.ap_in1k', pretrained=pretrained, num_classes=num_classes)
    
    elif model_name == 'convnext_tiny':
        model = timm.create_model('convnext_tiny.fb_in22k', pretrained=pretrained, num_classes=num_classes)
    
    elif model_name == 'vit_base':
        model = timm.create_model('vit_large_patch16_224.augreg_in21k_ft_in1k', pretrained=pretrained, num_classes=num_classes)
    
    elif model_name == 'swin_tiny':
        model = timm.create_model('swin_tiny_patch4_window7_224.ms_in22k_ft_in1k', pretrained=pretrained, num_classes=num_classes)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


# ==================== Training Functions ====================
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    
    return epoch_loss, epoch_acc.item()


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="[Validation]")
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    
    # Calculate detailed metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    return epoch_loss, epoch_acc.item(), precision, recall, f1


def train_model(model, train_loader, val_loader, criterion, optimizer, 
                scheduler, device, num_epochs, model_name, save_dir):
    """Complete training loop"""
    best_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': []
    }
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Model: {model_name} | Epoch {epoch}/{num_epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(
            model, val_loader, criterion, device
        )
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_prec)
        history['val_recall'].append(val_rec)
        history['val_f1'].append(val_f1)
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Val Precision: {val_prec:.4f} | Val Recall: {val_rec:.4f} | Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'history': history
            }, os.path.join(save_dir, f'{model_name}_best.pth'))
            print(f"✓ Saved best model with accuracy: {best_acc:.4f}")
    
    return history, best_acc


# ==================== Main Training Script ====================
def main():
    # ===== Configuration =====
    CONFIG = {
        'images_dir': '/kaggle/input/stanford-dogs-dataset/images/Images',
        'annotations_dir': '/kaggle/input/stanford-dogs-dataset/annotations/Annotation',
        'save_dir': 'model_checkpoints',
        'results_dir': 'results',
        'batch_size': 32,
        'num_epochs': 10,
        'learning_rate': 0.001,
        'num_workers': 4,
        'image_size': 224,
        'use_bbox': False,  # Set to True to use bounding box cropping
        'models_to_train': [
            'resnet18',
            'mobilenetv2',
            'resnet50',
            'resnet101',
            'efficientnet_b0',
            'efficientnet_b3',
            'efficientnet_b7',
            'convnext_tiny',
            'vit_base',
            'swin_tiny'
        ]
    }
    
    # Create directories
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    os.makedirs(CONFIG['results_dir'], exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'='*60}\n")
    
    # ===== Data Transforms =====
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(CONFIG['image_size']),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # ===== Load Dataset =====
    print("Loading dataset...")
    full_dataset = StanfordDogsDataset(
        images_dir=CONFIG['images_dir'],
        annotations_dir=CONFIG['annotations_dir'],
        transform=train_transform,
        use_bbox=CONFIG['use_bbox']
    )
    
    num_classes = len(full_dataset.class_to_idx)
    print(f"Number of classes: {num_classes}")
    
    # Split dataset (80% train, 20% validation)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply validation transform to validation set
    val_dataset.dataset.transform = val_transform
    
    print(f"Train size: {train_size}, Validation size: {val_size}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    
    # ===== Train All Models =====
    results_summary = {}
    
    for model_name in CONFIG['models_to_train']:
        print(f"\n\n{'#'*60}")
        print(f"# Training Model: {model_name.upper()}")
        print(f"{'#'*60}\n")
        
        start_time = time.time()
        
        try:
            # Create model
            model = create_model(model_name, num_classes=num_classes, pretrained=True)
            model = model.to(device)
            
            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
            
            # Train model
            history, best_acc = train_model(
                model, train_loader, val_loader, criterion, optimizer,
                scheduler, device, CONFIG['num_epochs'], model_name, CONFIG['save_dir']
            )
            
            training_time = time.time() - start_time
            
            # Save results
            results_summary[model_name] = {
                'best_accuracy': float(best_acc),
                'final_train_acc': float(history['train_acc'][-1]),
                'final_val_acc': float(history['val_acc'][-1]),
                'final_val_f1': float(history['val_f1'][-1]),
                'training_time_seconds': training_time,
                'history': history
            }
            
            # Save individual model history
            with open(os.path.join(CONFIG['results_dir'], f'{model_name}_history.json'), 'w') as f:
                json.dump(history, f, indent=4)
            
            print(f"\n✓ {model_name} training completed!")
            print(f"  Best Accuracy: {best_acc:.4f}")
            print(f"  Training Time: {training_time/60:.2f} minutes")
            
        except Exception as e:
            print(f"\n✗ Error training {model_name}: {str(e)}")
            results_summary[model_name] = {
                'error': str(e),
                'status': 'failed'
            }
            continue
    
    # ===== Save Final Results =====
    with open(os.path.join(CONFIG['results_dir'], 'all_models_results.json'), 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    # ===== Print Summary =====
    print(f"\n\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}\n")
    
    successful_models = {k: v for k, v in results_summary.items() if 'error' not in v}
    
    if successful_models:
        # Sort by best accuracy
        sorted_models = sorted(
            successful_models.items(),
            key=lambda x: x[1]['best_accuracy'],
            reverse=True
        )
        
        print(f"{'Model':<20} {'Best Acc':<12} {'Val F1':<12} {'Time (min)':<12}")
        print("-" * 60)
        
        for model_name, results in sorted_models:
            print(f"{model_name:<20} {results['best_accuracy']:<12.4f} "
                  f"{results['final_val_f1']:<12.4f} "
                  f"{results['training_time_seconds']/60:<12.2f}")
        
        print(f"\n{'='*60}")
        print(f"Best Model: {sorted_models[0][0]} with accuracy {sorted_models[0][1]['best_accuracy']:.4f}")
        print(f"{'='*60}\n")
    
    print(f"Results saved to: {CONFIG['results_dir']}")
    print(f"Model checkpoints saved to: {CONFIG['save_dir']}")


if __name__ == '__main__':
    main()
