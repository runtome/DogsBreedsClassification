#!/bin/bash

# Stanford Dogs Classification - Quick Start Script
# This script helps you get started with the project

echo "========================================"
echo "Stanford Dogs Classification Quick Start"
echo "========================================"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python
if ! command_exists python3; then
    echo "❌ Error: Python 3 is not installed"
    exit 1
fi
echo "✓ Python 3 is installed"

# Check pip
if ! command_exists pip3; then
    echo "❌ Error: pip3 is not installed"
    exit 1
fi
echo "✓ pip3 is installed"

# Install requirements
echo ""
echo "Installing required packages..."
pip3 install -r requirements.txt
echo "✓ Requirements installed"

# Check if dataset exists
echo ""
echo "Checking dataset..."
if [ ! -d "datasets/images/Images" ]; then
    echo "❌ Dataset not found!"
    echo ""
    echo "Please download the Stanford Dogs dataset from:"
    echo "https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset"
    echo ""
    echo "Extract the files and organize them as follows:"
    echo "  datasets/"
    echo "  ├── images/Images/"
    echo "  └── annotations/Annotation/"
    echo ""
    exit 1
else
    echo "✓ Dataset directory found"
fi

# Verify dataset
echo ""
echo "Verifying dataset structure..."
python3 dataset_utils.py --verify

# Ask user what to do
echo ""
echo "========================================"
echo "What would you like to do?"
echo "========================================"
echo "1. View dataset statistics and samples"
echo "2. Train all models (may take several hours)"
echo "3. Train quick test (ResNet-18 only, 5 epochs)"
echo "4. Train selected models"
echo "5. Visualize existing results"
echo "6. Run inference on an image"
echo "7. Exit"
echo ""
read -p "Enter your choice (1-7): " choice

case $choice in
    1)
        echo ""
        echo "Generating dataset statistics..."
        python3 dataset_utils.py --stats --sample --sizes
        ;;
    2)
        echo ""
        echo "Starting full training (this may take several hours)..."
        echo "Check config.py to customize settings"
        python3 train_dogs_classification.py
        echo ""
        echo "Training complete! Generating visualizations..."
        python3 visualize_results.py
        ;;
    3)
        echo ""
        echo "Starting quick test training..."
        cat > quick_test_config.py << EOF
# Quick test configuration
from config import *
NUM_EPOCHS = 5
BATCH_SIZE = 32
MODELS_TO_TRAIN = ['resnet18']
EOF
        python3 -c "import quick_test_config as config; import train_dogs_classification; train_dogs_classification.main()"
        rm quick_test_config.py
        ;;
    4)
        echo ""
        echo "Available models:"
        echo "  1. resnet18"
        echo "  2. resnet50"
        echo "  3. resnet101"
        echo "  4. mobilenetv2"
        echo "  5. efficientnet_b0"
        echo "  6. efficientnet_b3"
        echo "  7. efficientnet_b7"
        echo "  8. convnext_tiny"
        echo "  9. vit_base"
        echo " 10. swin_tiny"
        echo ""
        echo "Edit config.py to select specific models, then run:"
        echo "  python3 train_dogs_classification.py"
        ;;
    5)
        echo ""
        if [ -f "results/all_models_results.json" ]; then
            echo "Generating visualizations..."
            python3 visualize_results.py
        else
            echo "❌ No results found. Please train models first."
        fi
        ;;
    6)
        echo ""
        read -p "Enter path to image: " image_path
        if [ ! -f "$image_path" ]; then
            echo "❌ Image not found: $image_path"
            exit 1
        fi
        
        echo ""
        echo "Available options:"
        echo "1. Single model prediction"
        echo "2. Compare all models"
        read -p "Choose (1-2): " inference_choice
        
        if [ "$inference_choice" = "1" ]; then
            read -p "Enter model name (e.g., resnet18): " model_name
            python3 inference.py --image "$image_path" --model "$model_name" --top_k 5
        else
            python3 inference.py --image "$image_path" --compare
        fi
        ;;
    7)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Done!"
echo "========================================"
