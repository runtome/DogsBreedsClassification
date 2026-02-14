"""
Model-Specific Training Configurations
Optimized hyperparameters for each model architecture
"""

# ==================== Model-Specific Hyperparameters ====================
# Each model gets custom settings optimized for its architecture

MODEL_CONFIGS = {
    # ===== CNN Models (ResNet, MobileNet, EfficientNet) =====
    # These work well with standard settings
    'resnet18': {
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'optimizer': 'adam',
        'lr_scheduler': 'step',
        'step_size': 7,
        'gamma': 0.1,
        'warmup_epochs': 0,
        'grad_clip': None,
        'freeze_backbone': False,
        'label_smoothing': 0.0,
    },
    
    'resnet50': {
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'optimizer': 'adam',
        'lr_scheduler': 'step',
        'step_size': 7,
        'gamma': 0.1,
        'warmup_epochs': 0,
        'grad_clip': None,
        'freeze_backbone': False,
        'label_smoothing': 0.0,
    },
    
    'resnet101': {
        'learning_rate': 0.0005,  # Lower LR for deeper model
        'weight_decay': 1e-4,
        'optimizer': 'adam',
        'lr_scheduler': 'cosine',  # Cosine annealing
        'warmup_epochs': 2,
        'grad_clip': 1.0,
        'freeze_backbone': False,
        'label_smoothing': 0.1,
    },
    
    'mobilenetv2': {
        'learning_rate': 0.001,
        'weight_decay': 4e-5,  # Less regularization for smaller model
        'optimizer': 'adam',
        'lr_scheduler': 'step',
        'step_size': 7,
        'gamma': 0.1,
        'warmup_epochs': 0,
        'grad_clip': None,
        'freeze_backbone': False,
        'label_smoothing': 0.0,
    },
    
    'efficientnet_b0': {
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'optimizer': 'adamw',  # AdamW works better for EfficientNet
        'lr_scheduler': 'cosine',
        'warmup_epochs': 1,
        'grad_clip': None,
        'freeze_backbone': False,
        'label_smoothing': 0.1,
    },
    
    'efficientnet_b3': {
        'learning_rate': 0.0005,
        'weight_decay': 1e-5,
        'optimizer': 'adamw',
        'lr_scheduler': 'cosine',
        'warmup_epochs': 2,
        'grad_clip': 1.0,
        'freeze_backbone': False,
        'label_smoothing': 0.1,
    },
    
    'efficientnet_b7': {
        'learning_rate': 0.0003,  # Very low LR for large model
        'weight_decay': 1e-5,
        'optimizer': 'adamw',
        'lr_scheduler': 'cosine',
        'warmup_epochs': 3,
        'grad_clip': 1.0,
        'freeze_backbone': False,
        'label_smoothing': 0.1,
    },
    
    'efficientnetv2_m': {
        'learning_rate': 0.0005,
        'weight_decay': 1e-5,
        'optimizer': 'adamw',
        'lr_scheduler': 'cosine',
        'warmup_epochs': 2,
        'grad_clip': 1.0,
        'freeze_backbone': False,
        'label_smoothing': 0.1,
    },
    
    'efficientnetv2_l': {
        'learning_rate': 0.0003,
        'weight_decay': 1e-5,
        'optimizer': 'adamw',
        'lr_scheduler': 'cosine',
        'warmup_epochs': 3,
        'grad_clip': 1.0,
        'freeze_backbone': False,
        'label_smoothing': 0.1,
    },
    
    # ===== ConvNeXt Models =====
    'convnext_tiny': {
        'learning_rate': 0.0005,
        'weight_decay': 0.05,  # Higher weight decay for ConvNeXt
        'optimizer': 'adamw',
        'lr_scheduler': 'cosine',
        'warmup_epochs': 2,
        'grad_clip': 1.0,
        'freeze_backbone': False,
        'label_smoothing': 0.1,
        'layer_decay': 0.8,  # Layer-wise LR decay
    },
    
    'convnext_small': {
        'learning_rate': 0.0004,
        'weight_decay': 0.05,
        'optimizer': 'adamw',
        'lr_scheduler': 'cosine',
        'warmup_epochs': 3,
        'grad_clip': 1.0,
        'freeze_backbone': False,
        'label_smoothing': 0.1,
        'layer_decay': 0.8,
    },
    
    'convnext_base': {
        'learning_rate': 0.0003,
        'weight_decay': 0.05,
        'optimizer': 'adamw',
        'lr_scheduler': 'cosine',
        'warmup_epochs': 3,
        'grad_clip': 1.0,
        'freeze_backbone': False,
        'label_smoothing': 0.1,
        'layer_decay': 0.75,
    },
    
    # ===== Vision Transformers (CRITICAL: Need special care!) =====
    'vit_base': {
        'learning_rate': 0.0001,  # MUCH LOWER LR for ViT!
        'weight_decay': 0.05,  # Higher weight decay
        'optimizer': 'adamw',  # MUST use AdamW
        'lr_scheduler': 'cosine',  # Cosine annealing
        'warmup_epochs': 5,  # LONGER warmup is critical!
        'grad_clip': 1.0,  # Gradient clipping
        'freeze_backbone': False,
        'label_smoothing': 0.1,
        'layer_decay': 0.65,  # Strong layer-wise LR decay
        'mixup_alpha': 0.8,  # Mixup augmentation
        'cutmix_alpha': 1.0,  # CutMix augmentation
    },
    
    'vit_large': {
        'learning_rate': 0.00005,  # Even lower for ViT-Large
        'weight_decay': 0.05,
        'optimizer': 'adamw',
        'lr_scheduler': 'cosine',
        'warmup_epochs': 5,
        'grad_clip': 1.0,
        'freeze_backbone': False,
        'label_smoothing': 0.1,
        'layer_decay': 0.65,
        'mixup_alpha': 0.8,
        'cutmix_alpha': 1.0,
    },
    
    # ===== Swin Transformers (Also need special care) =====
    'swin_tiny': {
        'learning_rate': 0.0001,  # Lower LR for Swin
        'weight_decay': 0.05,
        'optimizer': 'adamw',
        'lr_scheduler': 'cosine',
        'warmup_epochs': 5,  # Longer warmup
        'grad_clip': 5.0,  # Higher grad clip for Swin
        'freeze_backbone': False,
        'label_smoothing': 0.1,
        'layer_decay': 0.7,
        'mixup_alpha': 0.8,
        'cutmix_alpha': 1.0,
    },
    
    'swin_small': {
        'learning_rate': 0.00008,
        'weight_decay': 0.05,
        'optimizer': 'adamw',
        'lr_scheduler': 'cosine',
        'warmup_epochs': 5,
        'grad_clip': 5.0,
        'freeze_backbone': False,
        'label_smoothing': 0.1,
        'layer_decay': 0.7,
        'mixup_alpha': 0.8,
        'cutmix_alpha': 1.0,
    },
    
    'swin_base': {
        'learning_rate': 0.00005,
        'weight_decay': 0.05,
        'optimizer': 'adamw',
        'lr_scheduler': 'cosine',
        'warmup_epochs': 5,
        'grad_clip': 5.0,
        'freeze_backbone': False,
        'label_smoothing': 0.1,
        'layer_decay': 0.65,
        'mixup_alpha': 0.8,
        'cutmix_alpha': 1.0,
    },
}


# ==================== Get Model Config ====================
def get_model_config(model_name):
    """
    Get optimized configuration for a specific model
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary of hyperparameters
    """
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name].copy()
    else:
        # Default configuration
        print(f"Warning: No specific config for {model_name}, using defaults")
        return {
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'optimizer': 'adam',
            'lr_scheduler': 'step',
            'step_size': 7,
            'gamma': 0.1,
            'warmup_epochs': 0,
            'grad_clip': None,
            'freeze_backbone': False,
            'label_smoothing': 0.0,
        }


# ==================== Key Insights ====================
"""
WHY TRANSFORMERS FAILED:

1. **Too High Learning Rate**
   - CNNs: Can handle LR=0.001
   - Transformers: Need LR=0.0001 or lower
   - Your ViT used 0.001 → diverged immediately!

2. **No Warmup**
   - Transformers are unstable at start
   - Need 3-5 epochs of gradual LR increase
   - Without warmup → loss explodes

3. **Wrong Optimizer**
   - Adam works for CNNs
   - Transformers NEED AdamW (with decoupled weight decay)

4. **Insufficient Regularization**
   - Transformers overfit easily on small datasets
   - Need: label smoothing, mixup, cutmix, higher weight decay

5. **No Layer-wise LR Decay**
   - Early layers should learn slower
   - Later layers (head) should learn faster
   - Critical for transformers!

SOLUTIONS IN THIS CONFIG:
✅ Model-specific learning rates (10x lower for transformers)
✅ Warmup epochs for stable start
✅ AdamW optimizer for transformers
✅ Label smoothing, mixup, cutmix
✅ Gradient clipping
✅ Layer-wise learning rate decay
✅ Cosine annealing for better convergence
"""
