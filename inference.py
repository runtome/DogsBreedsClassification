"""
Inference script for Stanford Dogs Classification
Test trained models on new images
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Import model creation function
import sys
sys.path.append('.')
from train_dogs_classification import create_model


class DogBreedClassifier:
    """Classifier for dog breed prediction"""
    
    def __init__(self, model_name, checkpoint_path, device='cuda'):
        """
        Args:
            model_name: Name of the model architecture
            checkpoint_path: Path to model checkpoint
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        # Load model
        print(f"Loading model: {model_name}")
        self.model = create_model(model_name, num_classes=120, pretrained=False)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully! Best accuracy: {checkpoint['best_acc']:.4f}")
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load class names (you'll need to save these during training)
        self.class_names = self._load_class_names()
    
    def _load_class_names(self):
        """Load class names from dataset"""
        # This should match the order from training
        # For now, we'll create a placeholder
        # In production, save class_to_idx during training
        try:
            with open('class_names.json', 'r') as f:
                return json.load(f)
        except:
            # Fallback: return class indices
            return {str(i): f"Class_{i}" for i in range(120)}
    
    def predict(self, image_path, top_k=5):
        """
        Predict dog breed for an image
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
            
        Returns:
            predictions: List of (class_name, probability) tuples
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        original_image = image.copy()
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, top_k)
        
        # Format results
        predictions = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            class_name = self.class_names.get(str(idx.item()), f"Class_{idx.item()}")
            predictions.append((class_name, prob.item()))
        
        return predictions, original_image
    
    def predict_batch(self, image_paths, top_k=5):
        """Predict for multiple images"""
        results = []
        for img_path in image_paths:
            predictions, image = self.predict(img_path, top_k)
            results.append({
                'image_path': img_path,
                'predictions': predictions,
                'image': image
            })
        return results


def visualize_predictions(image, predictions, save_path=None):
    """Visualize image with top predictions"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Display image
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('Input Image', fontsize=14, fontweight='bold')
    
    # Display predictions
    classes = [pred[0] for pred in predictions]
    probs = [pred[1] for pred in predictions]
    
    y_pos = np.arange(len(classes))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(classes)))
    
    bars = ax2.barh(y_pos, probs, color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(classes)
    ax2.invert_yaxis()
    ax2.set_xlabel('Probability', fontsize=12)
    ax2.set_title('Top Predictions', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 1])
    
    # Add value labels
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        ax2.text(prob + 0.01, i, f'{prob:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def compare_models(image_path, model_checkpoints, output_dir='inference_results'):
    """Compare predictions from multiple models on the same image"""
    Path(output_dir).mkdir(exist_ok=True)
    
    image = Image.open(image_path).convert('RGB')
    
    fig, axes = plt.subplots(2, (len(model_checkpoints) + 1) // 2, figsize=(20, 12))
    if len(model_checkpoints) == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (model_name, checkpoint_path) in enumerate(model_checkpoints.items()):
        try:
            # Load classifier
            classifier = DogBreedClassifier(model_name, checkpoint_path)
            
            # Get predictions
            predictions, _ = classifier.predict(image_path, top_k=3)
            
            # Plot
            ax = axes[idx]
            classes = [pred[0] for pred in predictions]
            probs = [pred[1] for pred in predictions]
            
            y_pos = np.arange(len(classes))
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(classes)))
            
            ax.barh(y_pos, probs, color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(classes, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel('Probability', fontsize=10)
            ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
            ax.set_xlim([0, 1])
            
            # Add value labels
            for i, prob in enumerate(probs):
                ax.text(prob + 0.01, i, f'{prob:.3f}', va='center', fontsize=9)
            
        except Exception as e:
            print(f"Error with model {model_name}: {e}")
            ax = axes[idx]
            ax.text(0.5, 0.5, f'Error: {model_name}', 
                   ha='center', va='center', fontsize=12)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
    
    # Hide extra subplots
    for idx in range(len(model_checkpoints), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Model Comparison: {Path(image_path).name}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = Path(output_dir) / f'comparison_{Path(image_path).stem}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Dog Breed Classification Inference')
    parser.add_argument('--image', type=str, required=True, 
                       help='Path to input image')
    parser.add_argument('--model', type=str, default='resnet18',
                       help='Model name (e.g., resnet18, efficientnet_b0)')
    parser.add_argument('--checkpoint', type=str, 
                       help='Path to model checkpoint (default: model_checkpoints/{model}_best.pth)')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top predictions to show')
    parser.add_argument('--compare', action='store_true',
                       help='Compare predictions from all available models')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    if args.compare:
        # Compare all available models
        checkpoint_dir = Path('model_checkpoints')
        available_models = {}
        
        for checkpoint_file in checkpoint_dir.glob('*_best.pth'):
            model_name = checkpoint_file.stem.replace('_best', '')
            available_models[model_name] = str(checkpoint_file)
        
        if not available_models:
            print("No trained models found in model_checkpoints/")
            return
        
        print(f"Comparing {len(available_models)} models...")
        compare_models(args.image, available_models, args.output_dir)
        
    else:
        # Single model inference
        checkpoint_path = args.checkpoint or f'model_checkpoints/{args.model}_best.pth'
        
        if not Path(checkpoint_path).exists():
            print(f"Error: Checkpoint not found: {checkpoint_path}")
            return
        
        # Load classifier
        classifier = DogBreedClassifier(args.model, checkpoint_path)
        
        # Predict
        predictions, image = classifier.predict(args.image, args.top_k)
        
        # Print results
        print(f"\nPredictions for: {args.image}")
        print("="*60)
        for i, (class_name, prob) in enumerate(predictions, 1):
            print(f"{i}. {class_name}: {prob:.4f} ({prob*100:.2f}%)")
        print("="*60)
        
        # Visualize
        output_path = Path(args.output_dir) / f'{args.model}_{Path(args.image).stem}_prediction.png'
        visualize_predictions(image, predictions, str(output_path))


if __name__ == '__main__':
    main()
