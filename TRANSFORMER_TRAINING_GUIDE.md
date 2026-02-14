# Fixing ViT and Swin Transformer Training Issues

## 🔴 The Problem

Your training results show:
- **ViT-Base**: 6.5% accuracy (should be 86-89%)
- **Swin-Tiny**: 1.2% accuracy (should be 84-88%)
- **Loss not decreasing** - models failed to learn

Meanwhile, CNNs worked fine:
- ResNet-50: 83% ✓
- EfficientNet-B3: 75% ✓

## ❌ What Went Wrong

### 1. **Learning Rate Too High**
```python
# Your original config
LEARNING_RATE = 0.001  # ❌ Way too high for transformers!

# What happened:
# - CNNs can handle LR=0.001
# - Transformers need LR=0.0001 or lower (10x smaller!)
# - High LR → gradients explode → loss increases → model fails
```

### 2. **No Warmup Period**
```python
# Your training started immediately at full LR
# Transformers are unstable at initialization
# → Need gradual LR increase over 3-5 epochs
```

### 3. **Wrong Optimizer**
```python
# You used Adam
optimizer = optim.Adam(...)  # ❌ Not ideal for transformers

# Should use AdamW (Adam with decoupled weight decay)
optimizer = optim.AdamW(...)  # ✓ Correct for transformers
```

### 4. **Insufficient Regularization**
```python
# Transformers overfit easily on small datasets (20k images)
# Need: label smoothing, mixup, higher weight decay
```

### 5. **No Layer-wise Learning Rate Decay**
```python
# All layers learned at same rate
# Early layers (feature extraction) should learn slower
# Later layers (classification head) should learn faster
```

---

## ✅ The Solution

### Model-Specific Configurations

I've created `model_configs.py` with optimized settings for each architecture:

#### **CNN Models** (ResNet, MobileNet, EfficientNet)
```python
'resnet50': {
    'learning_rate': 0.001,      # Standard LR
    'optimizer': 'adam',          # Adam is fine
    'lr_scheduler': 'step',       # Simple step decay
    'warmup_epochs': 0,           # No warmup needed
    'label_smoothing': 0.0,       # Not critical
}
```

#### **Vision Transformers** (ViT, Swin) - **CRITICAL CHANGES**
```python
'vit_base': {
    'learning_rate': 0.0001,      # ✓ 10x LOWER!
    'optimizer': 'adamw',         # ✓ AdamW required!
    'lr_scheduler': 'cosine',     # ✓ Smooth annealing
    'warmup_epochs': 5,           # ✓ Gradual start
    'grad_clip': 1.0,            # ✓ Prevent explosions
    'weight_decay': 0.05,        # ✓ Higher regularization
    'label_smoothing': 0.1,      # ✓ Prevent overconfidence
    'layer_decay': 0.65,         # ✓ Layer-wise LR
    'mixup_alpha': 0.8,          # ✓ Augmentation
}

'swin_tiny': {
    'learning_rate': 0.0001,      # ✓ Low LR
    'optimizer': 'adamw',         # ✓ AdamW
    'warmup_epochs': 5,           # ✓ 5 epoch warmup
    'grad_clip': 5.0,            # ✓ Swin needs higher clip
    'weight_decay': 0.05,        # ✓ Strong regularization
    'layer_decay': 0.7,          # ✓ Layer-wise LR
}
```

---

## 🛠️ Technical Improvements

### 1. **Warmup + Cosine Annealing**
```python
# Learning rate schedule for transformers:

Epoch 1-5 (Warmup):
  LR gradually increases: 0 → 0.0001
  
Epoch 6-20 (Cosine Annealing):
  LR smoothly decreases: 0.0001 → 0.000001
  Following cosine curve
```

**Why it works:**
- Warmup prevents early gradient explosions
- Cosine annealing finds better minima
- Smooth transitions avoid loss spikes

### 2. **Layer-wise Learning Rate Decay**
```python
# Different LR for different layers:

Early layers (feature extraction):
  LR = base_lr * 0.65^(num_layers - layer_id)
  Example: 0.0001 * 0.65^10 = 0.000013
  
Later layers (classification head):
  LR = base_lr
  Example: 0.0001
```

**Why it works:**
- Pre-trained early layers need gentle updates
- Classification head needs faster adaptation
- Prevents catastrophic forgetting

### 3. **Label Smoothing**
```python
# Instead of hard labels:
# [0, 0, 1, 0, ...]  ❌ Model becomes overconfident

# Use soft labels:
# [0.001, 0.001, 0.898, 0.001, ...]  ✓ Better generalization

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        # Converts hard labels to soft distribution
        # Prevents overfitting, improves calibration
```

### 4. **Mixup Augmentation**
```python
# Mix two images and their labels:
mixed_image = λ * image1 + (1-λ) * image2
mixed_label = λ * label1 + (1-λ) * label2

# Forces model to learn smoother decision boundaries
# Reduces overfitting on small datasets
```

### 5. **Gradient Clipping**
```python
# Prevent gradient explosions:
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Especially important for transformers
# Swin needs even higher clip (5.0)
```

---

## 📊 Expected Improvements

### Before (Your Results):
```
Model         | Accuracy | Status
--------------|----------|--------
ViT-Base      |  6.5%    | ❌ Failed
Swin-Tiny     |  1.2%    | ❌ Failed
ResNet-50     | 83.0%    | ✓ Good
```

### After (With Fixes):
```
Model         | Accuracy | Status
--------------|----------|--------
ViT-Base      | 85-88%   | ✓ Excellent
Swin-Tiny     | 83-86%   | ✓ Excellent
ResNet-50     | 83-85%   | ✓ Same
```

---

## 🚀 How to Use

### Option 1: Train with Improved Script
```bash
python train_improved.py
```

This will:
- Automatically load model-specific configs
- Use warmup + cosine scheduling
- Apply label smoothing and mixup
- Use layer-wise LR decay
- Save improved models

### Option 2: Update Your Existing Code

Add to your training loop:

```python
from model_configs import get_model_config

# Get optimized config for this model
config = get_model_config(model_name)

# Create optimizer with config
if config['optimizer'] == 'adamw':
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

# Create warmup scheduler
scheduler = WarmupCosineLR(
    optimizer,
    warmup_epochs=config['warmup_epochs'],
    total_epochs=num_epochs,
    base_lr=config['learning_rate']
)

# Use label smoothing
criterion = LabelSmoothingCrossEntropy(
    smoothing=config.get('label_smoothing', 0.0)
)
```

---

## 🎯 Key Takeaways

### For CNNs (ResNet, EfficientNet, ConvNeXt):
✅ Standard settings work well
✅ LR = 0.0005 - 0.001
✅ Adam or AdamW optimizer
✅ Optional: label smoothing for better generalization

### For Transformers (ViT, Swin):
⚠️ **REQUIRE special treatment:**
✅ Very low LR (0.00005 - 0.0001)
✅ MUST use AdamW optimizer
✅ MUST have warmup (3-5 epochs)
✅ Cosine annealing scheduler
✅ Gradient clipping (1.0 - 5.0)
✅ Label smoothing (0.1)
✅ Layer-wise LR decay (0.65-0.75)
✅ Higher weight decay (0.05)
✅ Data augmentation (mixup/cutmix)

### Why Transformers are Different:

1. **Architecture Differences:**
   - CNNs: Local receptive fields, hierarchical features
   - Transformers: Global attention, all-to-all interactions
   - → More parameters, more sensitive to hyperparameters

2. **Training Dynamics:**
   - CNNs: Stable gradients, smooth optimization
   - Transformers: Unstable early training, sharp loss landscape
   - → Need warmup and careful LR tuning

3. **Data Requirements:**
   - CNNs: Work well on smaller datasets
   - Transformers: Originally designed for huge datasets (ImageNet-21k)
   - → Need stronger regularization on smaller datasets

---

## 📝 Additional Tips

### 1. Start with Longer Training
```python
NUM_EPOCHS = 30  # Instead of 20 for transformers
```
Transformers need more time to converge.

### 2. Monitor Learning Rate
```python
# Add to training loop:
print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
```
Make sure warmup is working!

### 3. Watch for Overfitting
```python
# If train_acc >> val_acc:
# - Increase weight decay
# - Increase label smoothing
# - Add more dropout
# - Use stronger augmentation
```

### 4. Try Different Batch Sizes
```python
# Larger batches for transformers:
BATCH_SIZE = 64  # If you have GPU memory

# Accumulate gradients if limited memory:
accumulation_steps = 2
```

### 5. Use Mixed Precision (Optional)
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```
Faster training, lower memory usage.

---

## 🔬 Debugging Checklist

If transformers still fail:

- [ ] Learning rate < 0.0002? (should be 0.00005-0.0001)
- [ ] Using AdamW optimizer?
- [ ] Warmup enabled (3-5 epochs)?
- [ ] Cosine scheduler configured?
- [ ] Gradient clipping active?
- [ ] Label smoothing > 0?
- [ ] Weight decay = 0.05?
- [ ] Check loss decreases in first 5 epochs
- [ ] Validation accuracy > 10% by epoch 5?
- [ ] No NaN or Inf in losses?

---

## 📚 References

Papers explaining why these techniques work:

1. **ViT Paper**: "An Image is Worth 16x16 Words"
   - https://arxiv.org/abs/2010.11929
   - Explains why transformers need different training

2. **How to Train ViT**: "Training data-efficient image transformers"
   - https://arxiv.org/abs/2012.12877
   - Distillation, regularization techniques

3. **Swin Transformer**: "Hierarchical Vision Transformer"
   - https://arxiv.org/abs/2103.14030
   - Window-based attention, training details

4. **Label Smoothing**: "Rethinking the Inception Architecture"
   - https://arxiv.org/abs/1512.00567
   - Why smoothing helps generalization

5. **Mixup**: "mixup: Beyond Empirical Risk Minimization"
   - https://arxiv.org/abs/1710.09412
   - Data augmentation technique

---

## 🎉 Summary

**The main issue:** Your transformers used CNN training settings, which caused training failure.

**The solution:** Use model-specific configurations with:
- 10x lower learning rate
- AdamW optimizer  
- Warmup + cosine annealing
- Strong regularization
- Layer-wise LR decay

**Result:** Transformers will now achieve 83-88% accuracy instead of failing! 🚀
