"""
SIMPLIFIED Improved Training for Stanford Dogs
Focus on the 5 CRITICAL fixes for transformers without complex features
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

# Import from original script
from train_dogs_classification import StanfordDogsDataset, create_model, validate


# ==================== Simple Model Configs ====================
SIMPLE_CONFIGS = {
    # CNNs - Standard settings
    'resnet18': {'lr': 0.001, 'optimizer': 'adam', 'warmup': 0, 'wd': 1e-4},
    'resnet50': {'lr': 0.001, 'optimizer': 'adam', 'warmup': 0, 'wd': 1e-4},
    'resnet101': {'lr': 0.0005, 'optimizer': 'adam', 'warmup': 0, 'wd': 1e-4},
    'mobilenetv2': {'lr': 0.001, 'optimizer': 'adam', 'warmup': 0, 'wd': 4e-5},
    'efficientnet_b0': {'lr': 0.001, 'optimizer': 'adamw', 'warmup': 1, 'wd': 1e-5},
    'efficientnet_b3': {'lr': 0.0005, 'optimizer': 'adamw', 'warmup': 2, 'wd': 1e-5},
    'efficientnet_b7': {'lr': 0.0003, 'optimizer': 'adamw', 'warmup': 3, 'wd': 1e-5},
    'convnext_tiny': {'lr': 0.0005, 'optimizer': 'adamw', 'warmup': 2, 'wd': 0.05},
    
    # Transformers - CRITICAL: Much lower LR + warmup + AdamW!
    'vit_base': {'lr': 0.0001, 'optimizer': 'adamw', 'warmup': 5, 'wd': 0.05},
    'vit_large': {'lr': 0.00005, 'optimizer': 'adamw', 'warmup': 5, 'wd': 0.05},
    'swin_tiny': {'lr': 0.0001, 'optimizer': 'adamw', 'warmup': 5, 'wd': 0.05},
    'swin_small': {'lr': 0.00008, 'optimizer': 'adamw', 'warmup': 5, 'wd': 0.05},
    'swin_base': {'lr': 0.00005, 'optimizer': 'adamw', 'warmup': 5, 'wd': 0.05},
}


# ==================== Warmup Cosine Scheduler ====================
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6):
    """Simple warmup + cosine annealing"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Warmup: linear increase
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine annealing
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ==================== Simple Training Loop ====================
def train_one_epoch_simple(model, dataloader, criterion, optimizer, device, epoch):
    """Simple, stable training loop"""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward
        loss.backward()
        
        # Gradient clipping for stability (especially important for transformers)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects / len(dataloader.dataset)
    
    return epoch_loss, epoch_acc


# ==================== Main Training Function ====================
def train_model_simple(model, train_loader, val_loader, device, num_epochs, 
                       model_name, save_dir, config):
    """
    Simplified training with just the CRITICAL fixes:
    1. Correct learning rate
    2. Correct optimizer (AdamW for transformers)
    3. Warmup
    4. Cosine annealing
    5. Gradient clipping
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"LR: {config['lr']}, Optimizer: {config['optimizer']}, Warmup: {config['warmup']} epochs")
    print(f"{'='*60}\n")
    
    # Optimizer
    if config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['wd'])
    else:
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['wd'])
    
    # Scheduler with warmup
    steps_per_epoch = len(train_loader)
    num_warmup_steps = config['warmup'] * steps_per_epoch
    num_training_steps = num_epochs * steps_per_epoch
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': [],
        'learning_rates': []
    }
    
    for epoch in range(1, num_epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs} | LR: {current_lr:.6f}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_one_epoch_simple(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler (step per batch)
        # Note: This is already done per batch in the loop above
        
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
        print(f"Val F1: {val_f1:.4f}")
        
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
            }, os.path.join(save_dir, f'{model_name}_best_simple.pth'))
            print(f"✓ Saved best model: {best_acc:.4f}")
        
        # Step scheduler per epoch
        for _ in range(steps_per_epoch):
            scheduler.step()
    
    return history, best_acc


# ==================== Main ====================
def main():
    CONFIG = {
        'images_dir': '/kaggle/input/stanford-dogs-dataset/images/Images',
        'annotations_dir': '/kaggle/input/stanford-dogs-dataset/annotations/Annotation',
        'save_dir': 'model_checkpoints',
        'results_dir': 'results_simple',
        'batch_size': 16,
        'num_epochs': 5,
        'num_workers': 2,
        'image_size': 224,
        'models_to_train': [
            'resnet50',      # Baseline
            'vit_base',      # Test transformer
            'swin_tiny',     # Test swin
        ]
    }
    
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    os.makedirs(CONFIG['results_dir'], exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'='*60}\n")
    
    # Transforms
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
    
    # Dataset
    print("Loading dataset...")
    full_dataset = StanfordDogsDataset(
        images_dir=CONFIG['images_dir'],
        annotations_dir=CONFIG['annotations_dir'],
        transform=train_transform,
        use_bbox=False
    )
    
    num_classes = len(full_dataset.class_to_idx)
    print(f"Classes: {num_classes}")
    
    # Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    val_dataset.dataset.transform = val_transform
    
    print(f"Train: {train_size}, Val: {val_size}\n")
    
    # Loaders
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG['batch_size'],
        shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG['batch_size'],
        shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True
    )
    
    # Train models
    results = {}
    
    for model_name in CONFIG['models_to_train']:
        print(f"\n{'#'*60}")
        print(f"# {model_name.upper()}")
        print(f"{'#'*60}")
        
        if model_name not in SIMPLE_CONFIGS:
            print(f"No config for {model_name}, skipping...")
            continue
        
        start_time = time.time()
        
        try:
            # Get config
            config = SIMPLE_CONFIGS[model_name]
            
            # Create model
            model = create_model(model_name, num_classes=num_classes, pretrained=True)
            model = model.to(device)
            
            # Train
            history, best_acc = train_model_simple(
                model, train_loader, val_loader, device, 
                CONFIG['num_epochs'], model_name, CONFIG['save_dir'], config
            )
            
            training_time = time.time() - start_time
            
            results[model_name] = {
                'best_accuracy': float(best_acc),
                'final_val_acc': float(history['val_acc'][-1]),
                'final_val_f1': float(history['val_f1'][-1]),
                'training_time_seconds': training_time,
                'config': config,
                'history': history
            }
            
            print(f"\n✓ {model_name}: Best Acc = {best_acc:.4f}, Time = {training_time/60:.1f} min")
            
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            results[model_name] = {'error': str(e)}
    
    # Save results
    with open(os.path.join(CONFIG['results_dir'], 'simple_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, res in results.items():
        if 'best_accuracy' in res:
            print(f"{name:20s} | Acc: {res['best_accuracy']:.4f} | Time: {res['training_time_seconds']/60:.1f}min")
        else:
            print(f"{name:20s} | FAILED")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
