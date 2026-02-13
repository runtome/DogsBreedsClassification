"""
Configuration file for Stanford Dogs Classification
Edit these settings to customize your training
"""

# ==================== Dataset Paths ====================
IMAGES_DIR = 'datasets/images/Images'
ANNOTATIONS_DIR = 'datasets/annotations/Annotation'

# ==================== Training Settings ====================
BATCH_SIZE = 32                    # Batch size (reduce if out of memory)
NUM_EPOCHS = 20                    # Number of training epochs
LEARNING_RATE = 0.001              # Initial learning rate
NUM_WORKERS = 4                    # Number of data loading workers
IMAGE_SIZE = 224                   # Input image size

# ==================== Data Augmentation ====================
USE_BBOX = False                   # Use bounding box cropping from annotations
RANDOM_ROTATION = 15               # Random rotation degrees
COLOR_JITTER = True                # Enable color jittering
BRIGHTNESS = 0.2                   # Brightness jitter factor
CONTRAST = 0.2                     # Contrast jitter factor
SATURATION = 0.2                   # Saturation jitter factor

# ==================== Model Selection ====================
# Comment out models you don't want to train
MODELS_TO_TRAIN = [
    'resnet18',           # Fast, good baseline
    'mobilenetv2',        # Very fast, mobile-friendly
    'resnet50',           # Standard choice, balanced
    'resnet101',          # Deeper, more capacity
    'efficientnet_b0',    # Efficient, good accuracy
    'efficientnet_b3',    # Better accuracy, slower
    'efficientnet_b7',    # Best accuracy, slowest
    'convnext_tiny',      # Modern CNN
    'vit_base',           # Transformer architecture
    'swin_tiny',          # Shifted window transformer
]

# ==================== Output Directories ====================
SAVE_DIR = 'model_checkpoints'     # Where to save model checkpoints
RESULTS_DIR = 'results'            # Where to save training results
INFERENCE_DIR = 'inference_results' # Where to save inference results

# ==================== Advanced Settings ====================
TRAIN_VAL_SPLIT = 0.8              # Train/validation split ratio
RANDOM_SEED = 42                   # Random seed for reproducibility
PRETRAINED = True                  # Use pretrained ImageNet weights

# Learning rate scheduler
USE_LR_SCHEDULER = True
LR_STEP_SIZE = 7                   # Epochs before reducing LR
LR_GAMMA = 0.1                     # LR reduction factor

# Early stopping
USE_EARLY_STOPPING = False
PATIENCE = 5                       # Epochs to wait before stopping

# ==================== GPU Settings ====================
# Leave empty to auto-detect
DEVICE = ''                        # 'cuda' or 'cpu' or '' for auto

# Mixed precision training (faster on modern GPUs)
USE_MIXED_PRECISION = False

# ==================== Preset Configurations ====================
# Uncomment one of these to quickly switch configurations

# # Quick Test (fast training for testing)
# BATCH_SIZE = 64
# NUM_EPOCHS = 5
# MODELS_TO_TRAIN = ['resnet18', 'mobilenetv2']

# # Best Accuracy (longer training)
# NUM_EPOCHS = 50
# MODELS_TO_TRAIN = ['efficientnet_b7', 'convnext_tiny', 'swin_tiny']

# # Fast Comparison (all models, fewer epochs)
# NUM_EPOCHS = 10
# MODELS_TO_TRAIN = [
#     'resnet18', 'mobilenetv2', 'resnet50', 'efficientnet_b0',
#     'efficientnet_b3', 'convnext_tiny', 'vit_base', 'swin_tiny'
# ]

# # GPU Memory Limited (smaller batch size)
# BATCH_SIZE = 16
# MODELS_TO_TRAIN = ['resnet18', 'mobilenetv2', 'efficientnet_b0']
