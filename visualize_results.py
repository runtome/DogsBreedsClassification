"""
Visualize and compare model training results
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


def load_results(results_dir='results'):
    """Load all model results"""
    results_path = Path(results_dir)
    
    # Load summary
    with open(results_path / 'all_models_results.json', 'r') as f:
        results_summary = json.load(f)
    
    return results_summary


def plot_comparison_metrics(results_summary, save_path='results/comparison_metrics.png'):
    """Plot comparison of key metrics across models"""
    # Filter successful models
    successful_models = {k: v for k, v in results_summary.items() if 'error' not in v}
    
    if not successful_models:
        print("No successful models to plot")
        return
    
    # Prepare data
    models = list(successful_models.keys())
    best_acc = [v['best_accuracy'] for v in successful_models.values()]
    val_f1 = [v['final_val_f1'] for v in successful_models.values()]
    train_time = [v['training_time_seconds']/60 for v in successful_models.values()]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Comparison - Stanford Dogs Classification', fontsize=16, fontweight='bold')
    
    # 1. Best Accuracy
    ax1 = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    bars1 = ax1.barh(models, best_acc, color=colors)
    ax1.set_xlabel('Accuracy', fontsize=12)
    ax1.set_title('Best Validation Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlim([0, 1])
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars1, best_acc)):
        ax1.text(acc + 0.01, i, f'{acc:.4f}', va='center', fontsize=10)
    
    # 2. F1 Score
    ax2 = axes[0, 1]
    bars2 = ax2.barh(models, val_f1, color=colors)
    ax2.set_xlabel('F1 Score', fontsize=12)
    ax2.set_title('Final Validation F1 Score', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 1])
    
    for i, (bar, f1) in enumerate(zip(bars2, val_f1)):
        ax2.text(f1 + 0.01, i, f'{f1:.4f}', va='center', fontsize=10)
    
    # 3. Training Time
    ax3 = axes[1, 0]
    bars3 = ax3.barh(models, train_time, color=colors)
    ax3.set_xlabel('Training Time (minutes)', fontsize=12)
    ax3.set_title('Total Training Time', fontsize=14, fontweight='bold')
    
    for i, (bar, time) in enumerate(zip(bars3, train_time)):
        ax3.text(time + max(train_time)*0.02, i, f'{time:.1f}', va='center', fontsize=10)
    
    # 4. Accuracy vs Training Time
    ax4 = axes[1, 1]
    scatter = ax4.scatter(train_time, best_acc, c=range(len(models)), 
                         s=200, cmap='viridis', alpha=0.7, edgecolors='black', linewidth=2)
    
    for i, model in enumerate(models):
        ax4.annotate(model, (train_time[i], best_acc[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax4.set_xlabel('Training Time (minutes)', fontsize=12)
    ax4.set_ylabel('Best Accuracy', fontsize=12)
    ax4.set_title('Accuracy vs Training Time', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison metrics to: {save_path}")
    plt.close()


def plot_training_curves(results_summary, save_path='results/training_curves.png'):
    """Plot training and validation curves for all models"""
    successful_models = {k: v for k, v in results_summary.items() if 'error' not in v}
    
    if not successful_models:
        print("No successful models to plot")
        return
    
    n_models = len(successful_models)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Training Progress - All Models', fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))
    
    # Plot accuracy
    ax1 = axes[0]
    for i, (model_name, results) in enumerate(successful_models.items()):
        history = results['history']
        epochs = range(1, len(history['train_acc']) + 1)
        
        ax1.plot(epochs, history['train_acc'], '--', color=colors[i], alpha=0.5, linewidth=1)
        ax1.plot(epochs, history['val_acc'], '-', color=colors[i], label=model_name, linewidth=2)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Validation Accuracy (solid) vs Training Accuracy (dashed)', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2 = axes[1]
    for i, (model_name, results) in enumerate(successful_models.items()):
        history = results['history']
        epochs = range(1, len(history['train_loss']) + 1)
        
        ax2.plot(epochs, history['train_loss'], '--', color=colors[i], alpha=0.5, linewidth=1)
        ax2.plot(epochs, history['val_loss'], '-', color=colors[i], label=model_name, linewidth=2)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Validation Loss (solid) vs Training Loss (dashed)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved training curves to: {save_path}")
    plt.close()


def create_results_table(results_summary, save_path='results/results_table.csv'):
    """Create a detailed results table"""
    successful_models = {k: v for k, v in results_summary.items() if 'error' not in v}
    
    if not successful_models:
        print("No successful models to create table")
        return
    
    # Prepare data
    data = []
    for model_name, results in successful_models.items():
        data.append({
            'Model': model_name,
            'Best Accuracy': results['best_accuracy'],
            'Final Val Accuracy': results['final_val_acc'],
            'Final Val F1': results['final_val_f1'],
            'Final Train Accuracy': results['final_train_acc'],
            'Training Time (min)': results['training_time_seconds'] / 60
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df = df.sort_values('Best Accuracy', ascending=False)
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"\nSaved results table to: {save_path}")
    
    # Print table
    print("\n" + "="*80)
    print("DETAILED RESULTS TABLE")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    return df


def plot_model_efficiency(results_summary, save_path='results/model_efficiency.png'):
    """Plot model efficiency (accuracy per training time)"""
    successful_models = {k: v for k, v in results_summary.items() if 'error' not in v}
    
    if not successful_models:
        print("No successful models to plot")
        return
    
    # Calculate efficiency
    models = []
    efficiency = []
    accuracies = []
    times = []
    
    for model_name, results in successful_models.items():
        models.append(model_name)
        acc = results['best_accuracy']
        time_min = results['training_time_seconds'] / 60
        eff = acc / time_min  # accuracy per minute
        
        accuracies.append(acc)
        times.append(time_min)
        efficiency.append(eff)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(models)))
    bars = ax.barh(models, efficiency, color=colors)
    
    ax.set_xlabel('Efficiency (Accuracy / Training Time in minutes)', fontsize=12)
    ax.set_title('Model Efficiency: Accuracy per Training Minute', fontsize=14, fontweight='bold')
    
    # Add value labels
    for i, (bar, eff, acc, time) in enumerate(zip(bars, efficiency, accuracies, times)):
        ax.text(eff + max(efficiency)*0.01, i, 
               f'{eff:.4f}\n(Acc: {acc:.3f}, {time:.1f}min)', 
               va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved efficiency plot to: {save_path}")
    plt.close()


def generate_report(results_summary):
    """Generate a text report of the results"""
    report_path = 'results/model_comparison_report.txt'
    
    successful_models = {k: v for k, v in results_summary.items() if 'error' not in v}
    failed_models = {k: v for k, v in results_summary.items() if 'error' in v}
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("STANFORD DOGS CLASSIFICATION - MODEL COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Summary
        f.write(f"Total Models Trained: {len(results_summary)}\n")
        f.write(f"Successful: {len(successful_models)}\n")
        f.write(f"Failed: {len(failed_models)}\n\n")
        
        if successful_models:
            # Best model
            best_model = max(successful_models.items(), key=lambda x: x[1]['best_accuracy'])
            f.write("="*80 + "\n")
            f.write("BEST PERFORMING MODEL\n")
            f.write("="*80 + "\n")
            f.write(f"Model: {best_model[0]}\n")
            f.write(f"Best Accuracy: {best_model[1]['best_accuracy']:.4f}\n")
            f.write(f"Final Val F1: {best_model[1]['final_val_f1']:.4f}\n")
            f.write(f"Training Time: {best_model[1]['training_time_seconds']/60:.2f} minutes\n\n")
            
            # Most efficient model
            efficiencies = {k: v['best_accuracy'] / (v['training_time_seconds']/60) 
                          for k, v in successful_models.items()}
            most_efficient = max(efficiencies.items(), key=lambda x: x[1])
            
            f.write("="*80 + "\n")
            f.write("MOST EFFICIENT MODEL\n")
            f.write("="*80 + "\n")
            f.write(f"Model: {most_efficient[0]}\n")
            f.write(f"Efficiency: {most_efficient[1]:.4f} accuracy/minute\n")
            f.write(f"Accuracy: {successful_models[most_efficient[0]]['best_accuracy']:.4f}\n")
            f.write(f"Training Time: {successful_models[most_efficient[0]]['training_time_seconds']/60:.2f} minutes\n\n")
            
            # All models ranking
            f.write("="*80 + "\n")
            f.write("ALL MODELS RANKED BY ACCURACY\n")
            f.write("="*80 + "\n")
            
            sorted_models = sorted(successful_models.items(), 
                                 key=lambda x: x[1]['best_accuracy'], 
                                 reverse=True)
            
            for rank, (model_name, results) in enumerate(sorted_models, 1):
                f.write(f"\n{rank}. {model_name}\n")
                f.write(f"   Best Accuracy: {results['best_accuracy']:.4f}\n")
                f.write(f"   Val F1 Score: {results['final_val_f1']:.4f}\n")
                f.write(f"   Training Time: {results['training_time_seconds']/60:.2f} minutes\n")
        
        if failed_models:
            f.write("\n" + "="*80 + "\n")
            f.write("FAILED MODELS\n")
            f.write("="*80 + "\n")
            for model_name, results in failed_models.items():
                f.write(f"\n{model_name}:\n")
                f.write(f"   Error: {results.get('error', 'Unknown error')}\n")
    
    print(f"Saved report to: {report_path}")


def main():
    """Main visualization function"""
    print("\nGenerating visualizations and reports...\n")
    
    try:
        # Load results
        results_summary = load_results()
        
        # Generate all visualizations
        plot_comparison_metrics(results_summary)
        plot_training_curves(results_summary)
        plot_model_efficiency(results_summary)
        
        # Create results table
        create_results_table(results_summary)
        
        # Generate text report
        generate_report(results_summary)
        
        print("\n✓ All visualizations and reports generated successfully!")
        print("Check the 'results' directory for all outputs.\n")
        
    except FileNotFoundError:
        print("Error: Could not find results file. Make sure to run training first.")
    except Exception as e:
        print(f"Error generating visualizations: {e}")


if __name__ == '__main__':
    main()
