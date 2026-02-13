"""
Utility script for Stanford Dogs Dataset
- Verify dataset structure
- Generate class mappings
- Display dataset statistics
"""

import os
import json
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def verify_dataset(images_dir, annotations_dir=None):
    """Verify dataset structure and integrity"""
    print("="*60)
    print("DATASET VERIFICATION")
    print("="*60 + "\n")
    
    images_path = Path(images_dir)
    
    if not images_path.exists():
        print(f"❌ Error: Images directory not found: {images_dir}")
        return False
    
    # Count breeds and images
    breed_folders = [d for d in images_path.iterdir() if d.is_dir()]
    total_images = 0
    breed_counts = {}
    
    print(f"✓ Found {len(breed_folders)} breed folders")
    
    for breed_folder in breed_folders:
        images = list(breed_folder.glob('*.jpg'))
        breed_name = breed_folder.name
        breed_counts[breed_name] = len(images)
        total_images += len(images)
    
    print(f"✓ Total images: {total_images}")
    print(f"✓ Average images per breed: {total_images/len(breed_folders):.1f}")
    
    # Check annotations if provided
    if annotations_dir:
        annotations_path = Path(annotations_dir)
        if annotations_path.exists():
            annotation_folders = [d for d in annotations_path.iterdir() if d.is_dir()]
            print(f"✓ Found {len(annotation_folders)} annotation folders")
        else:
            print(f"⚠ Annotations directory not found: {annotations_dir}")
    
    print("\n" + "="*60)
    print("Dataset structure is valid! ✓")
    print("="*60 + "\n")
    
    return True


def generate_class_mappings(images_dir, output_file='class_names.json'):
    """Generate and save class name mappings"""
    images_path = Path(images_dir)
    breed_folders = sorted([d for d in images_path.iterdir() if d.is_dir()])
    
    class_to_idx = {}
    idx_to_class = {}
    
    for idx, breed_folder in enumerate(breed_folders):
        # Extract readable breed name
        breed_name = breed_folder.name.split('-', 1)[1] if '-' in breed_folder.name else breed_folder.name
        breed_name = breed_name.replace('_', ' ')
        
        class_to_idx[breed_name] = idx
        idx_to_class[str(idx)] = breed_name
    
    # Save mappings
    mappings = {
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'num_classes': len(class_to_idx)
    }
    
    with open(output_file, 'w') as f:
        json.dump(mappings, f, indent=2)
    
    print(f"✓ Class mappings saved to: {output_file}")
    print(f"  Total classes: {len(class_to_idx)}")
    
    return mappings


def display_dataset_statistics(images_dir, save_plot='dataset_statistics.png'):
    """Display detailed dataset statistics"""
    images_path = Path(images_dir)
    breed_folders = sorted([d for d in images_path.iterdir() if d.is_dir()])
    
    breed_counts = {}
    breed_names = []
    
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60 + "\n")
    
    for breed_folder in breed_folders:
        images = list(breed_folder.glob('*.jpg'))
        breed_name = breed_folder.name.split('-', 1)[1] if '-' in breed_folder.name else breed_folder.name
        breed_name = breed_name.replace('_', ' ')
        breed_counts[breed_name] = len(images)
        breed_names.append(breed_name)
    
    # Calculate statistics
    counts = list(breed_counts.values())
    total_images = sum(counts)
    
    print(f"Total Images: {total_images}")
    print(f"Number of Breeds: {len(breed_counts)}")
    print(f"Average per Breed: {np.mean(counts):.1f}")
    print(f"Median per Breed: {np.median(counts):.1f}")
    print(f"Min per Breed: {np.min(counts)}")
    print(f"Max per Breed: {np.max(counts)}")
    print(f"Std Dev: {np.std(counts):.1f}")
    
    # Top 10 breeds
    print("\nTop 10 Breeds by Image Count:")
    sorted_breeds = sorted(breed_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (breed, count) in enumerate(sorted_breeds[:10], 1):
        print(f"  {i}. {breed}: {count} images")
    
    # Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Histogram of image counts
    ax1 = axes[0]
    ax1.hist(counts, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(counts), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(counts):.1f}')
    ax1.axvline(np.median(counts), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(counts):.1f}')
    ax1.set_xlabel('Number of Images per Breed', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Images per Breed', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Top 20 breeds bar chart
    ax2 = axes[1]
    top_20 = sorted_breeds[:20]
    names = [breed for breed, _ in top_20]
    values = [count for _, count in top_20]
    
    colors = plt.cm.viridis(np.linspace(0, 1, 20))
    ax2.barh(range(20), values, color=colors)
    ax2.set_yticks(range(20))
    ax2.set_yticklabels(names, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel('Number of Images', fontsize=12)
    ax2.set_title('Top 20 Breeds by Image Count', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_plot, dpi=300, bbox_inches='tight')
    print(f"\n✓ Statistics plot saved to: {save_plot}")
    plt.close()


def sample_images(images_dir, num_samples=20, save_plot='sample_images.png'):
    """Display random sample images from the dataset"""
    images_path = Path(images_dir)
    breed_folders = list(images_path.iterdir())
    
    # Randomly select breeds
    selected_breeds = np.random.choice(breed_folders, min(num_samples, len(breed_folders)), replace=False)
    
    fig, axes = plt.subplots(4, 5, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, breed_folder in enumerate(selected_breeds[:20]):
        images = list(breed_folder.glob('*.jpg'))
        if images:
            # Select random image
            img_path = np.random.choice(images)
            img = Image.open(img_path)
            
            # Display
            axes[idx].imshow(img)
            axes[idx].axis('off')
            
            # Title
            breed_name = breed_folder.name.split('-', 1)[1] if '-' in breed_folder.name else breed_folder.name
            breed_name = breed_name.replace('_', ' ')
            axes[idx].set_title(breed_name, fontsize=10, fontweight='bold')
    
    # Hide extra subplots
    for idx in range(len(selected_breeds), 20):
        axes[idx].axis('off')
    
    plt.suptitle('Random Sample Images from Stanford Dogs Dataset', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_plot, dpi=300, bbox_inches='tight')
    print(f"✓ Sample images saved to: {save_plot}")
    plt.close()


def check_image_sizes(images_dir, num_samples=1000):
    """Check distribution of image sizes"""
    images_path = Path(images_dir)
    breed_folders = list(images_path.iterdir())
    
    widths = []
    heights = []
    
    # Sample random images
    all_images = []
    for breed_folder in breed_folders:
        all_images.extend(list(breed_folder.glob('*.jpg')))
    
    sampled_images = np.random.choice(all_images, min(num_samples, len(all_images)), replace=False)
    
    print(f"\nAnalyzing {len(sampled_images)} random images...")
    
    for img_path in sampled_images:
        try:
            img = Image.open(img_path)
            widths.append(img.width)
            heights.append(img.height)
        except:
            continue
    
    print(f"\nImage Size Statistics:")
    print(f"  Width  - Mean: {np.mean(widths):.0f}, Min: {np.min(widths)}, Max: {np.max(widths)}")
    print(f"  Height - Mean: {np.mean(heights):.0f}, Min: {np.min(heights)}, Max: {np.max(heights)}")
    
    # Aspect ratios
    aspect_ratios = [w/h for w, h in zip(widths, heights)]
    print(f"  Aspect Ratio - Mean: {np.mean(aspect_ratios):.2f}, Median: {np.median(aspect_ratios):.2f}")


def main():
    """Main utility function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Stanford Dogs Dataset Utilities')
    parser.add_argument('--images_dir', type=str, default='datasets/images/Images',
                       help='Path to images directory')
    parser.add_argument('--annotations_dir', type=str, default='datasets/annotations/Annotation',
                       help='Path to annotations directory')
    parser.add_argument('--verify', action='store_true',
                       help='Verify dataset structure')
    parser.add_argument('--stats', action='store_true',
                       help='Display dataset statistics')
    parser.add_argument('--sample', action='store_true',
                       help='Display sample images')
    parser.add_argument('--sizes', action='store_true',
                       help='Check image sizes')
    parser.add_argument('--all', action='store_true',
                       help='Run all checks')
    
    args = parser.parse_args()
    
    if args.all:
        args.verify = args.stats = args.sample = args.sizes = True
    
    # Default: run all checks if nothing specified
    if not any([args.verify, args.stats, args.sample, args.sizes]):
        args.verify = args.stats = args.sample = args.sizes = True
    
    if args.verify:
        verify_dataset(args.images_dir, args.annotations_dir)
        generate_class_mappings(args.images_dir)
    
    if args.stats:
        display_dataset_statistics(args.images_dir)
    
    if args.sample:
        sample_images(args.images_dir)
    
    if args.sizes:
        check_image_sizes(args.images_dir)
    
    print("\n✓ All checks completed!\n")


if __name__ == '__main__':
    main()
