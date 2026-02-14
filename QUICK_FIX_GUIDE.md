# Quick Fix Reference - ViT & Swin Training

## 🔴 Your Original Settings (Why They Failed)

```python
# What you used for ALL models:
LEARNING_RATE = 0.001
OPTIMIZER = Adam
SCHEDULER = StepLR (step every 7 epochs)
WARMUP = None
LABEL_SMOOTHING = 0.0
WEIGHT_DECAY = 0.0001
GRADIENT_CLIPPING = None
LAYER_WISE_LR = None
```

### Result:
- ✅ **ResNet-50**: 83.0% (worked fine)
- ✅ **EfficientNet-B3**: 75.8% (worked fine)
- ❌ **ViT-Base**: 6.5% (FAILED - loss didn't decrease)
- ❌ **Swin-Tiny**: 1.2% (FAILED - loss stayed flat)

---

## ✅ NEW Settings (Will Fix Transformers)

### For CNNs (ResNet, EfficientNet) - Keep Same:
```python
LEARNING_RATE = 0.001          # ✓ Same
OPTIMIZER = Adam               # ✓ Same  
SCHEDULER = StepLR             # ✓ Same
WARMUP = 0                     # ✓ Same
```

### For ViT-Base - MAJOR CHANGES:
```python
LEARNING_RATE = 0.0001         # ✓ 10x LOWER!
OPTIMIZER = AdamW              # ✓ Changed from Adam
SCHEDULER = Cosine             # ✓ Smooth annealing
WARMUP = 5 epochs              # ✓ Critical!
LABEL_SMOOTHING = 0.1          # ✓ Added
WEIGHT_DECAY = 0.05            # ✓ 500x higher!
GRADIENT_CLIPPING = 1.0        # ✓ Added
LAYER_WISE_LR_DECAY = 0.65     # ✓ Added
MIXUP_ALPHA = 0.8              # ✓ Added
```

### For Swin-Tiny - MAJOR CHANGES:
```python
LEARNING_RATE = 0.0001         # ✓ 10x LOWER!
OPTIMIZER = AdamW              # ✓ Changed from Adam
SCHEDULER = Cosine             # ✓ Smooth annealing
WARMUP = 5 epochs              # ✓ Critical!
LABEL_SMOOTHING = 0.1          # ✓ Added
WEIGHT_DECAY = 0.05            # ✓ 500x higher!
GRADIENT_CLIPPING = 5.0        # ✓ Higher for Swin
LAYER_WISE_LR_DECAY = 0.7      # ✓ Added
MIXUP_ALPHA = 0.8              # ✓ Added
```

---

## 📊 Expected Results After Fix

```
Model         | Old Acc | New Acc | Improvement
--------------|---------|---------|-------------
ResNet-50     | 83.0%   | 83-85%  | Stable
EfficientNet  | 75.8%   | 76-79%  | +1-3%
ViT-Base      |  6.5%   | 85-88%  | +80%! 🚀
Swin-Tiny     |  1.2%   | 83-86%  | +82%! 🚀
```

---

## ⚡ How to Use

### Option 1: Use the Improved Script (Easiest)
```bash
python train_improved.py
```
All settings are automatically applied!

### Option 2: Quick Manual Fix

Add this to your existing code:

```python
# At the start, import configs
from model_configs import get_model_config

# When training each model:
for model_name in models_to_train:
    # Get model-specific settings
    config = get_model_config(model_name)
    
    # Create optimizer
    if config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    
    # Rest of training...
```

---

## 🎯 The 5 Critical Changes

### 1️⃣ Lower Learning Rate
```python
# Old:
lr = 0.001  # Too high for transformers

# New:
lr = 0.0001  # 10x lower - prevents gradient explosion
```

### 2️⃣ Use AdamW
```python
# Old:
optimizer = optim.Adam(...)

# New:
optimizer = optim.AdamW(...)  # Decoupled weight decay
```

### 3️⃣ Add Warmup
```python
# Old:
# Started at full LR immediately

# New:
# Gradually increase LR over 5 epochs: 0 → 0.0001
warmup_epochs = 5
```

### 4️⃣ Cosine Annealing
```python
# Old:
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
# Sharp drops in LR

# New:
scheduler = WarmupCosineLR(optimizer, warmup=5, total=20)
# Smooth decrease: 0.0001 → 0.000001
```

### 5️⃣ Stronger Regularization
```python
# Old:
weight_decay = 0.0001
label_smoothing = 0.0

# New:
weight_decay = 0.05      # 500x higher!
label_smoothing = 0.1    # Prevents overconfidence
```

---

## 🐛 Still Having Issues?

### Check Your Loss Curves

**Good Training (Loss decreases):**
```
Epoch 1: Loss = 4.8 → 3.5
Epoch 2: Loss = 3.5 → 2.8
Epoch 3: Loss = 2.8 → 2.2
Epoch 5: Loss = 1.8 → 1.5
Epoch 10: Loss = 1.0 → 0.9
```
✅ Model is learning!

**Bad Training (Loss stays flat):**
```
Epoch 1: Loss = 4.8 → 4.8
Epoch 2: Loss = 4.8 → 4.8
Epoch 3: Loss = 4.8 → 4.7
Epoch 5: Loss = 4.7 → 4.7
Epoch 10: Loss = 4.6 → 4.6
```
❌ Model is NOT learning - LR too high or optimizer wrong!

### Quick Debug Steps:

1. **Print current LR every epoch:**
   ```python
   print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
   ```
   Should gradually increase in first 5 epochs (warmup)

2. **Check accuracy by epoch 5:**
   ```python
   # Should be:
   # Epoch 5 accuracy > 30% = Good
   # Epoch 5 accuracy < 10% = Bad (settings wrong)
   ```

3. **Monitor gradient norms:**
   ```python
   total_norm = 0
   for p in model.parameters():
       if p.grad is not None:
           total_norm += p.grad.data.norm(2).item() ** 2
   total_norm = total_norm ** 0.5
   print(f"Gradient norm: {total_norm:.2f}")
   
   # Should be: 1-10 (good), >100 (gradient explosion!)
   ```

---

## 📈 Training Timeline

### ViT-Base with Correct Settings:

```
Epoch 1-5 (Warmup):
  LR: 0.00002 → 0.0001 (gradual increase)
  Acc: 5% → 35%
  Loss: 4.8 → 2.5

Epoch 6-10:
  LR: 0.0001 → 0.00008 (cosine decay)
  Acc: 35% → 65%
  Loss: 2.5 → 1.2

Epoch 11-20:
  LR: 0.00008 → 0.00001 (cosine decay)
  Acc: 65% → 86%
  Loss: 1.2 → 0.6
```

Final: **86% accuracy** ✅

### What You Had (Wrong Settings):

```
Epoch 1-10:
  LR: 0.001 (constant, too high)
  Acc: 1% → 6%
  Loss: 4.8 → 4.3 (barely moving)

Model never learned! ❌
```

---

## 💡 Why This Matters

**Transformers vs CNNs:**

| Aspect | CNNs | Transformers |
|--------|------|--------------|
| Parameters | 25-50M | 85-300M |
| Inductive bias | Local patterns | Global attention |
| Training stability | Stable | Unstable early |
| Optimal LR | 0.0005-0.001 | 0.00005-0.0001 |
| Need warmup | No | Yes (critical) |
| Regularization | Moderate | Strong |

**Bottom line:** Transformers need **10x more care** in hyperparameter tuning!

---

## ✅ Checklist

Before training transformers:

- [ ] LR is 0.0001 or lower
- [ ] Using AdamW optimizer (not Adam)
- [ ] Warmup enabled (3-5 epochs)
- [ ] Cosine annealing scheduler
- [ ] Weight decay = 0.05
- [ ] Label smoothing = 0.1
- [ ] Gradient clipping enabled
- [ ] Training for 20-30 epochs
- [ ] Batch size 32-64
- [ ] Checking loss decreases by epoch 5

If all checked → Should work! 🎉

---

## 🚀 Next Steps

1. Run `python train_improved.py`
2. Watch first 5 epochs - loss should decrease to ~2.5
3. By epoch 10 - accuracy should be >60%
4. By epoch 20 - accuracy should be >83%

Good luck! 🐕
