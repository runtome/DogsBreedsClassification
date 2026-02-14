"""
IMPROVED Stanford Dogs Classification with Model-Specific Training
Fixes ViT and Swin Transformer training issues
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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

# Import model configs
from model_configs import get_model_config

# Import original functions
from train_dogs_classification import StanfordDogsDataset, create_model


# ==================== Advanced Loss Functions ====================
class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        log_prob = torch.nn.functional.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prob).sum(dim=1).mean()
        return loss


def mixup_data(x, y, alpha=1.0):
    """Apply mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss calculation"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ==================== Warmup Scheduler ====================
class WarmupCosineLR:
    """Learning rate scheduler with warmup and cosine annealing"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
    
    def step(self):
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Warmup phase
            lr = self.base_lr * self.current_epoch / self.warmup_epochs
        else:
            # Cosine annealing phase
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


# ==================== Layer-wise LR Decay ====================
def get_layer_wise_lr_params(model, model_name, base_lr, layer_decay=0.65):
    """
    Apply layer-wise learning rate decay
    Early layers get lower LR, later layers get higher LR
    """
    if 'vit' in model_name or 'swin' in model_name:
        # For transformers, use layer-wise decay
        parameter_group_names = {}
        parameter_group_vars = {}
        
        num_layers = len(list(model.named_parameters()))
        layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers))
        
        for i, (name, param) in enumerate(model.named_parameters()):
            if not param.requires_grad:
                continue
            
            lr_scale = layer_scales[i]
            group_name = f"layer_{i}"
            
            if group_name not in parameter_group_names:
                parameter_group_names[group_name] = {
                    "lr": base_lr * lr_scale,
                    "weight_decay": 0.05,
                    "params": [],
                }
                parameter_group_vars[group_name] = {
                    "lr": base_lr * lr_scale,
                    "weight_decay": 0.05,
                    "params": [],
                }
            
            parameter_group_vars[group_name]["params"].append(param)
        
        return list(parameter_group_vars.values())
    else:
        # For CNNs, standard param groups
        return [{'params': model.parameters(), 'lr': base_lr}]


# ==================== Improved Training Function ====================
def train_one_epoch_improved(model, dataloader, criterion, optimizer, device, epoch, config):
    """
    Improved training loop with:
    - Gradient clipping
    - Mixup augmentation (optional)
    - Progress tracking
    """
    model.train()
    running_loss = 0.0
    running_corrects = 0
    
    use_mixup = config.get('mixup_alpha', 0) > 0
    mixup_alpha = config.get('mixup_alpha', 0)
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Apply mixup if configured
        if use_mixup and np.random.rand() < 0.5:
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, mixup_alpha)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping if configured
        if config.get('grad_clip'):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        
        optimizer.step()
        
        # Statistics
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        
        if use_mixup and np.random.rand() < 0.5:
            running_corrects += (lam * (preds == labels_a).sum().item() + 
                                (1 - lam) * (preds == labels_b).sum().item())
        else:
            running_corrects += torch.sum(preds == labels.data).item()
        
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects / len(dataloader.dataset)
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validation function (unchanged)"""
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
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects / len(dataloader.dataset)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    return epoch_loss, epoch_acc, precision, recall, f1


# ==================== Main Training with Model-Specific Configs ====================
def train_model_improved(model, train_loader, val_loader, device, num_epochs, 
                        model_name, save_dir, config):
    """
    Train model with model-specific optimizations
    """
    print(f"\n{'='*60}")
    print(f"Training Configuration for {model_name}:")
    print(f"{'='*60}")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")
    
    # Create optimizer based on config
    if config.get('layer_decay'):
        # Use layer-wise LR decay for transformers
        param_groups = get_layer_wise_lr_params(
            model, model_name, config['learning_rate'], config['layer_decay']
        )
    else:
        param_groups = model.parameters()
    
    if config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            param_groups,
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    else:
        optimizer = optim.Adam(
            param_groups,
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    
    # Create loss function
    if config.get('label_smoothing', 0) > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config['label_smoothing'])
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Create scheduler
    if config['lr_scheduler'] == 'cosine':
        scheduler = WarmupCosineLR(
            optimizer, 
            warmup_epochs=config.get('warmup_epochs', 0),
            total_epochs=num_epochs,
            base_lr=config['learning_rate']
        )
    elif config['lr_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config.get('step_size', 7), 
            gamma=config.get('gamma', 0.1)
        )
    else:
        scheduler = None
    
    best_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': [],
        'learning_rates': []
    }
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Model: {model_name} | Epoch {epoch}/{num_epochs}")
        if scheduler:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning Rate: {current_lr:.6f}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_one_epoch_improved(
            model, train_loader, criterion, optimizer, device, epoch, config
        )
        
        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        if scheduler:
            if isinstance(scheduler, WarmupCosineLR):
                current_lr = scheduler.step()
            else:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = config['learning_rate']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_prec)
        history['val_recall'].append(val_rec)
        history['val_f1'].append(val_f1)
        history['learning_rates'].append(current_lr)
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Val F1: {val_f1:.4f} | LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'history': history,
                'config': config
            }, os.path.join(save_dir, f'{model_name}_best_improved.pth'))
            print(f"✓ Saved best model with accuracy: {best_acc:.4f}")
    
    return history, best_acc


# ==================== Main ====================
def main():
    # Configuration
    CONFIG = {
        'images_dir': '/kaggle/input/stanford-dogs-dataset/images/Images',
        'annotations_dir': '/kaggle/input/stanford-dogs-dataset/annotations/Annotation',
        'save_dir': 'model_checkpoints',
        'results_dir': 'results_improved',
        'batch_size': 32,
        'num_epochs': 10,
        'num_workers': 4,
        'image_size': 224,
        'use_bbox': False,
        'models_to_train': [
            'vit_base',      # Test transformer with proper config
            'swin_tiny',     # Test swin with proper config
            'resnet50',      # Baseline for comparison
        ]
    }
    
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    os.makedirs(CONFIG['results_dir'], exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Data transforms
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
    
    # Load dataset
    print("Loading dataset...")
    full_dataset = StanfordDogsDataset(
        images_dir=CONFIG['images_dir'],
        annotations_dir=CONFIG['annotations_dir'],
        transform=train_transform,
        use_bbox=CONFIG['use_bbox']
    )
    
    num_classes = len(full_dataset.class_to_idx)
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    val_dataset.dataset.transform = val_transform
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                             shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                           shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)
    
    # Train all models
    results_summary = {}
    
    for model_name in CONFIG['models_to_train']:
        print(f"\n{'#'*60}")
        print(f"# Training: {model_name.upper()}")
        print(f"{'#'*60}\n")
        
        start_time = time.time()
        
        try:
            # Get model-specific config
            model_config = get_model_config(model_name)
            
            # Create model
            model = create_model(model_name, num_classes=num_classes, pretrained=True)
            model = model.to(device)
            
            # Train
            history, best_acc = train_model_improved(
                model, train_loader, val_loader, device, CONFIG['num_epochs'],
                model_name, CONFIG['save_dir'], model_config
            )
            
            training_time = time.time() - start_time
            
            results_summary[model_name] = {
                'best_accuracy': float(best_acc),
                'final_val_acc': float(history['val_acc'][-1]),
                'final_val_f1': float(history['val_f1'][-1]),
                'training_time_seconds': training_time,
                'config': model_config,
                'history': history
            }
            
            print(f"\n✓ {model_name} completed! Best Acc: {best_acc:.4f}")
            
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            results_summary[model_name] = {'error': str(e), 'status': 'failed'}
    
    # Save results
    with open(os.path.join(CONFIG['results_dir'], 'improved_results.json'), 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    print(f"\n{'='*60}")
    print("Results saved to:", CONFIG['results_dir'])
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
