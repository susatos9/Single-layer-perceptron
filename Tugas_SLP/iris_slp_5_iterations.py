"""
Single Layer Perceptron - 5 Iterations × 100 Epochs Implementation
Matching the format in SLP-rev.xlsx - SLP+Valid.csv
Author: [Your Name]
NIM: [Your NIM]
Date: September 7, 2025

This implementation runs 5 iterations with 100 epochs each,
using different random seed for each iteration to get different
initial weights, similar to the CSV format.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import time

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class MultiIterationSLP:
    """
    Single Layer Perceptron with multiple iterations support
    """
    
    def __init__(self, learning_rate=0.1, epochs_per_iteration=100):
        self.learning_rate = learning_rate
        self.epochs_per_iteration = epochs_per_iteration
        
        # Storage for all iterations
        self.all_iterations_history = []
        self.iteration_summaries = []
        
    def sigmoid(self, z):
        """Sigmoid activation function with clipping to prevent overflow"""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def forward(self, X):
        """Forward pass through the network"""
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def compute_loss(self, y_true, y_pred):
        """Compute binary cross-entropy loss"""
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def compute_accuracy(self, y_true, y_pred):
        """Compute accuracy"""
        predictions = (y_pred >= 0.5).astype(int)
        accuracy = np.mean(predictions == y_true) * 100
        return accuracy
    
    def train_single_iteration(self, X_train, y_train, X_val, y_val, iteration_num, random_seed):
        """Train a single iteration with specified random seed"""
        
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration_num}/5 (Random Seed: {random_seed})")
        print(f"{'='*60}")
        
        # Initialize weights with specific random seed
        np.random.seed(random_seed)
        n_features = X_train.shape[1]
        
        # Initialize weights similar to CSV (all weights = 0.5, bias = 0.5)
        if iteration_num == 1:
            # First iteration uses the exact same initialization as CSV
            self.weights = np.array([0.5, 0.5, 0.5, 0.5])
            self.bias = 0.5
        else:
            # Other iterations use different random initialization
            self.weights = np.random.normal(0.5, 0.2, n_features)
            self.bias = np.random.normal(0.5, 0.2)
        
        print(f"Initial weights: {self.weights}")
        print(f"Initial bias: {self.bias:.6f}")
        print()
        
        # History for this iteration
        iteration_history = {
            'iteration': iteration_num,
            'epoch': [],
            'train_accuracy': [],
            'train_loss': [],
            'val_accuracy': [],
            'val_loss': [],
            'initial_weights': self.weights.copy(),
            'initial_bias': self.bias,
            'final_weights': None,
            'final_bias': None
        }
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.epochs_per_iteration):
            # Forward pass on training data
            y_pred_train = self.forward(X_train)
            
            # Compute training metrics
            train_loss = self.compute_loss(y_train, y_pred_train)
            train_accuracy = self.compute_accuracy(y_train, y_pred_train)
            
            # Compute gradients
            n_samples = X_train.shape[0]
            dz = y_pred_train - y_train
            dw = (1/n_samples) * np.dot(X_train.T, dz)
            db = (1/n_samples) * np.sum(dz)
            
            # Update weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Forward pass on validation data
            y_pred_val = self.forward(X_val)
            
            # Compute validation metrics
            val_loss = self.compute_loss(y_val, y_pred_val)
            val_accuracy = self.compute_accuracy(y_val, y_pred_val)
            
            # Store history
            iteration_history['epoch'].append(epoch + 1)
            iteration_history['train_accuracy'].append(train_accuracy)
            iteration_history['train_loss'].append(train_loss)
            iteration_history['val_accuracy'].append(val_accuracy)
            iteration_history['val_loss'].append(val_loss)
            
            # Print progress every 20 epochs or at the end
            if epoch % 20 == 0 or epoch == self.epochs_per_iteration - 1:
                print(f"Epoch {epoch + 1:3d}: "
                      f"Train Acc: {train_accuracy:6.2f}% | Train Loss: {train_loss:.4f} | "
                      f"Val Acc: {val_accuracy:6.2f}% | Val Loss: {val_loss:.4f}")
        
        # Store final weights
        iteration_history['final_weights'] = self.weights.copy()
        iteration_history['final_bias'] = self.bias
        
        duration = time.time() - start_time
        
        # Create iteration summary
        summary = {
            'iteration': iteration_num,
            'random_seed': random_seed,
            'initial_weights': iteration_history['initial_weights'],
            'initial_bias': iteration_history['initial_bias'],
            'final_weights': iteration_history['final_weights'],
            'final_bias': iteration_history['final_bias'],
            'final_train_accuracy': train_accuracy,
            'final_train_loss': train_loss,
            'final_val_accuracy': val_accuracy,
            'final_val_loss': val_loss,
            'duration': duration
        }
        
        print(f"\\nIteration {iteration_num} completed in {duration:.2f} seconds")
        print(f"Final Training Accuracy: {train_accuracy:.2f}%")
        print(f"Final Validation Accuracy: {val_accuracy:.2f}%")
        
        return iteration_history, summary
    
    def train_multiple_iterations(self, X_train, y_train, X_val, y_val, n_iterations=5):
        """Train multiple iterations with different random seeds"""
        
        print(f"🚀 Starting {n_iterations} iterations training...")
        print(f"Each iteration: {self.epochs_per_iteration} epochs")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Define random seeds for each iteration (similar to CSV pattern)
        random_seeds = [42, 123, 456, 789, 999]
        
        total_start_time = time.time()
        
        for i in range(n_iterations):
            iteration_history, summary = self.train_single_iteration(
                X_train, y_train, X_val, y_val, 
                iteration_num=i+1, 
                random_seed=random_seeds[i]
            )
            
            self.all_iterations_history.append(iteration_history)
            self.iteration_summaries.append(summary)
        
        total_duration = time.time() - total_start_time
        
        print(f"\\n{'='*60}")
        print(f"🎉 ALL {n_iterations} ITERATIONS COMPLETED!")
        print(f"Total training time: {total_duration:.2f} seconds")
        print(f"{'='*60}")
        
        # Print summary table
        self.print_iterations_summary()
        
        return self.all_iterations_history, self.iteration_summaries
    
    def print_iterations_summary(self):
        """Print a summary table of all iterations"""
        
        print("\\n📊 ITERATIONS SUMMARY:")
        print("-" * 100)
        print(f"{'Iter':<5} {'Seed':<6} {'Final Train Acc':<15} {'Final Val Acc':<13} {'Train Loss':<11} {'Val Loss':<9} {'Time (s)':<8}")
        print("-" * 100)
        
        for summary in self.iteration_summaries:
            print(f"{summary['iteration']:<5} "
                  f"{summary['random_seed']:<6} "
                  f"{summary['final_train_accuracy']:<15.2f} "
                  f"{summary['final_val_accuracy']:<13.2f} "
                  f"{summary['final_train_loss']:<11.4f} "
                  f"{summary['final_val_loss']:<9.4f} "
                  f"{summary['duration']:<8.2f}")
        
        print("-" * 100)
        
        # Calculate averages
        avg_train_acc = np.mean([s['final_train_accuracy'] for s in self.iteration_summaries])
        avg_val_acc = np.mean([s['final_val_accuracy'] for s in self.iteration_summaries])
        avg_train_loss = np.mean([s['final_train_loss'] for s in self.iteration_summaries])
        avg_val_loss = np.mean([s['final_val_loss'] for s in self.iteration_summaries])
        
        print(f"{'AVG':<5} {'N/A':<6} "
              f"{avg_train_acc:<15.2f} "
              f"{avg_val_acc:<13.2f} "
              f"{avg_train_loss:<11.4f} "
              f"{avg_val_loss:<9.4f} "
              f"{'N/A':<8}")
        print()


def load_iris_data():
    """Load Iris dataset from CSV file"""
    
    data_path = "/media/nugroho-adi-susanto/Windows-SSD/Users/Nugroho Adi Susanto/Documents/UGM/Kuliah/AI/Deep Learning/SLP-rev.xlsx - Data.csv"
    
    try:
        # Try to load the data file
        df = pd.read_csv(data_path, header=None, 
                         names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
        print(f"✅ Data loaded from {data_path}")
    except FileNotFoundError:
        print("⚠️  Data file not found. Creating sample Iris dataset...")
        # Create sample data similar to the CSV
        np.random.seed(42)
        
        # Sample data matching the CSV pattern
        setosa_data = [
            [5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2],
            [4.6, 3.1, 1.5, 0.2], [5.0, 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.2],
            [4.6, 3.4, 1.4, 0.2], [5.0, 3.4, 1.5, 0.2], [4.4, 2.9, 1.4, 0.2],
            [4.9, 3.1, 1.5, 0.2], [5.4, 3.7, 1.5, 0.2], [4.8, 3.4, 1.6, 0.2],
            [4.8, 3.0, 1.4, 0.2], [4.3, 3.0, 1.1, 0.2], [5.8, 4.0, 1.2, 0.2],
        ]
        
        versicolor_data = [
            [7.0, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5], [6.9, 3.1, 4.9, 1.5],
            [5.5, 2.3, 4.0, 1.3], [6.5, 2.8, 4.6, 1.5], [5.7, 2.8, 4.5, 1.3],
            [6.3, 3.3, 4.7, 1.6], [4.9, 2.4, 3.3, 1.0], [6.6, 2.9, 4.6, 1.3],
            [5.2, 2.7, 3.9, 1.4], [5.0, 2.0, 3.5, 1.0], [5.9, 3.0, 4.2, 1.5],
            [6.0, 2.2, 4.0, 1.0], [6.1, 2.9, 4.7, 1.4], [5.6, 2.9, 3.6, 1.3],
        ]
        
        # Create DataFrame
        all_data = setosa_data + versicolor_data
        species = ['Iris-setosa'] * len(setosa_data) + ['Iris-versicolor'] * len(versicolor_data)
        
        df = pd.DataFrame(all_data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        df['species'] = species
    
    # Filter for binary classification (Setosa vs Versicolor)
    binary_df = df[df['species'].isin(['Iris-setosa', 'Iris-versicolor'])].copy()
    
    # Prepare features and target
    X = binary_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    y = (binary_df['species'] == 'Iris-versicolor').astype(int).values
    
    print(f"Dataset loaded: {len(X)} samples")
    print(f"Features: {list(binary_df.columns[:-1])}")
    print(f"Classes: Setosa (0): {np.sum(y==0)}, Versicolor (1): {np.sum(y==1)}")
    
    return X, y, binary_df


def create_comprehensive_plots(all_iterations_history, iteration_summaries):
    """Create comprehensive plots for all iterations"""
    
    # Set up plotting style
    plt.rcParams.update({
        'font.size': 10,
        'figure.titlesize': 12,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9
    })
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Single Layer Perceptron - 5 Iterations × 100 Epochs Analysis', fontsize=16, fontweight='bold')
    
    # Colors for each iteration
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot 1: Training Accuracy for all iterations
    ax1 = axes[0, 0]
    for i, history in enumerate(all_iterations_history):
        ax1.plot(history['epoch'], history['train_accuracy'], 
                color=colors[i], label=f'Iteration {i+1}', linewidth=2)
    ax1.set_title('Training Accuracy Across Iterations')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    
    # Plot 2: Validation Accuracy for all iterations
    ax2 = axes[0, 1]
    for i, history in enumerate(all_iterations_history):
        ax2.plot(history['epoch'], history['val_accuracy'], 
                color=colors[i], label=f'Iteration {i+1}', linewidth=2)
    ax2.set_title('Validation Accuracy Across Iterations')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)
    
    # Plot 3: Training Loss for all iterations
    ax3 = axes[0, 2]
    for i, history in enumerate(all_iterations_history):
        ax3.plot(history['epoch'], history['train_loss'], 
                color=colors[i], label=f'Iteration {i+1}', linewidth=2)
    ax3.set_title('Training Loss Across Iterations')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Plot 4: Validation Loss for all iterations
    ax4 = axes[1, 0]
    for i, history in enumerate(all_iterations_history):
        ax4.plot(history['epoch'], history['val_loss'], 
                color=colors[i], label=f'Iteration {i+1}', linewidth=2)
    ax4.set_title('Validation Loss Across Iterations')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # Plot 5: Final Performance Comparison
    ax5 = axes[1, 1]
    iterations = [f'Iter {i+1}' for i in range(len(iteration_summaries))]
    train_accs = [s['final_train_accuracy'] for s in iteration_summaries]
    val_accs = [s['final_val_accuracy'] for s in iteration_summaries]
    
    x = np.arange(len(iterations))
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, train_accs, width, label='Training Accuracy', 
                   color='lightblue', edgecolor='navy', alpha=0.7)
    bars2 = ax5.bar(x + width/2, val_accs, width, label='Validation Accuracy', 
                   color='lightcoral', edgecolor='darkred', alpha=0.7)
    
    ax5.set_title('Final Accuracy Comparison')
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Accuracy (%)')
    ax5.set_xticks(x)
    ax5.set_xticklabels(iterations)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim(0, 105)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Plot 6: Statistics Summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Calculate statistics
    avg_train_acc = np.mean(train_accs)
    std_train_acc = np.std(train_accs)
    avg_val_acc = np.mean(val_accs)
    std_val_acc = np.std(val_accs)
    
    stats_text = f"""
    📊 FINAL STATISTICS
    
    Training Accuracy:
    • Average: {avg_train_acc:.2f}%
    • Std Dev: {std_train_acc:.2f}%
    • Min: {min(train_accs):.2f}%
    • Max: {max(train_accs):.2f}%
    
    Validation Accuracy:
    • Average: {avg_val_acc:.2f}%
    • Std Dev: {std_val_acc:.2f}%
    • Min: {min(val_accs):.2f}%
    • Max: {max(val_accs):.2f}%
    
    Training Configuration:
    • Iterations: 5
    • Epochs per iteration: 100
    • Learning Rate: 0.1
    • Total epochs: 500
    """
    
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save the plot
    plt.savefig('/media/nugroho-adi-susanto/Windows-SSD/Users/Nugroho Adi Susanto/Documents/UGM/Kuliah/AI/Deep Learning/5_iterations_100_epochs_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Comprehensive analysis plot saved as '5_iterations_100_epochs_analysis.png'")


def save_results_to_csv(all_iterations_history, iteration_summaries):
    """Save results to CSV files for comparison with original CSV"""
    
    # Save detailed results (similar to original CSV format)
    detailed_results = []
    
    for history in all_iterations_history:
        for i, epoch in enumerate(history['epoch']):
            detailed_results.append({
                'iteration': history['iteration'],
                'epoch': epoch,
                'train_accuracy': history['train_accuracy'][i],
                'train_loss': history['train_loss'][i],
                'val_accuracy': history['val_accuracy'][i],
                'val_loss': history['val_loss'][i]
            })
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv('/media/nugroho-adi-susanto/Windows-SSD/Users/Nugroho Adi Susanto/Documents/UGM/Kuliah/AI/Deep Learning/python_5_iterations_detailed_results.csv', 
                      index=False)
    
    # Save summary results
    summary_df = pd.DataFrame(iteration_summaries)
    summary_df.to_csv('/media/nugroho-adi-susanto/Windows-SSD/Users/Nugroho Adi Susanto/Documents/UGM/Kuliah/AI/Deep Learning/python_5_iterations_summary.csv', 
                     index=False)
    
    print("✅ Results saved to CSV files:")
    print("   • python_5_iterations_detailed_results.csv")
    print("   • python_5_iterations_summary.csv")


def main():
    """Main execution function"""
    
    print("🚀 Single Layer Perceptron - 5 Iterations × 100 Epochs")
    print("=" * 60)
    
    # Load data
    print("📊 Loading Iris dataset...")
    X, y, df = load_iris_data()
    
    # Split data (using same split as CSV analysis)
    print("🔄 Splitting data into train/validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    # Initialize and train model
    print("\\n🤖 Initializing Multi-Iteration SLP...")
    slp = MultiIterationSLP(learning_rate=0.1, epochs_per_iteration=100)
    
    # Train multiple iterations
    all_iterations_history, iteration_summaries = slp.train_multiple_iterations(
        X_train, y_train, X_val, y_val, n_iterations=5
    )
    
    # Create comprehensive plots
    print("\\n📈 Creating comprehensive analysis plots...")
    create_comprehensive_plots(all_iterations_history, iteration_summaries)
    
    # Save results to CSV
    print("\\n💾 Saving results to CSV files...")
    save_results_to_csv(all_iterations_history, iteration_summaries)
    
    print("\\n🎉 Analysis completed successfully!")
    print("Files generated:")
    print("• 5_iterations_100_epochs_analysis.png")
    print("• python_5_iterations_detailed_results.csv")
    print("• python_5_iterations_summary.csv")


if __name__ == "__main__":
    main()
