"""
Single Layer Perceptron for Iris Dataset - Complete Implementation
Author: [Your Name]
NIM: [Your NIM]
Date: September 7, 2025

This implementation provides:
1. Training and validation splits
2. Accuracy and loss tracking per epoch
3. Visualization plots for presentation
4. CSV export for comparison with Google Sheets
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class IrisSLP:
    """
    Single Layer Perceptron for Iris binary classification
    """
    
    def __init__(self, learning_rate=0.1, random_seed=42):
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Model parameters
        self.weights = None
        self.bias = None
        
        # Training history
        self.history = {
            'epoch': [],
            'train_accuracy': [],
            'train_loss': [],
            'val_accuracy': [],
            'val_loss': []
        }
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def forward(self, X):
        """Forward pass"""
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def compute_loss(self, y_true, y_pred):
        """Binary cross-entropy loss"""
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def compute_accuracy(self, y_true, y_pred):
        """Compute accuracy percentage"""
        predictions = (y_pred >= 0.5).astype(int)
        return np.mean(predictions == y_true) * 100
    
    def fit(self, X_train, y_train, X_val, y_val, epochs=100):
        """
        Train the model
        """
        n_samples, n_features = X_train.shape
        
        # Initialize parameters
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0.0
        
        print("Training Single Layer Perceptron for Iris Dataset")
        print("=" * 60)
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Epochs: {epochs}")
        print()
        
        for epoch in range(epochs):
            # Forward pass on training data
            y_pred_train = self.forward(X_train)
            
            # Compute training metrics
            train_loss = self.compute_loss(y_train, y_pred_train)
            train_accuracy = self.compute_accuracy(y_train, y_pred_train)
            
            # Compute gradients
            error = y_pred_train - y_train
            dw = np.dot(X_train.T, error) / n_samples
            db = np.mean(error)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Forward pass on validation data
            y_pred_val = self.forward(X_val)
            
            # Compute validation metrics
            val_loss = self.compute_loss(y_val, y_pred_val)
            val_accuracy = self.compute_accuracy(y_val, y_pred_val)
            
            # Store history
            self.history['epoch'].append(epoch + 1)
            self.history['train_accuracy'].append(train_accuracy)
            self.history['train_loss'].append(train_loss)
            self.history['val_accuracy'].append(val_accuracy)
            self.history['val_loss'].append(val_loss)
            
            # Print progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch + 1:3d}: "
                      f"Train Acc: {train_accuracy:6.2f}% | Train Loss: {train_loss:.4f} | "
                      f"Val Acc: {val_accuracy:6.2f}% | Val Loss: {val_loss:.4f}")
        
        print("\nTraining completed!")
        return self.history
    
    def predict(self, X):
        """Make predictions"""
        return self.forward(X)
    
    def predict_classes(self, X):
        """Predict class labels"""
        probabilities = self.predict(X)
        return (probabilities >= 0.5).astype(int)


def load_and_split_data():
    """Load Iris data and create train/validation splits"""
    
    # Load data
    data_path = "/media/nugroho-adi-susanto/Windows-SSD/Users/Nugroho Adi Susanto/Documents/UGM/Kuliah/AI/Deep Learning/SLP-rev.xlsx - Data.csv"
    
    try:
        df = pd.read_csv(data_path, header=None, 
                         names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
    except FileNotFoundError:
        print("Creating sample Iris dataset...")
        # Create sample data if file not found
        np.random.seed(42)
        
        # Generate sample Iris data
        setosa_samples = 50
        versicolor_samples = 50
        
        # Setosa features (generally smaller)
        setosa_data = np.random.normal([5.0, 3.5, 1.4, 0.2], [0.3, 0.3, 0.2, 0.1], (setosa_samples, 4))
        setosa_labels = ['Iris-setosa'] * setosa_samples
        
        # Versicolor features (generally larger)
        versicolor_data = np.random.normal([6.0, 2.8, 4.3, 1.3], [0.4, 0.3, 0.4, 0.2], (versicolor_samples, 4))
        versicolor_labels = ['Iris-versicolor'] * versicolor_samples
        
        # Combine data
        all_data = np.vstack([setosa_data, versicolor_data])
        all_labels = setosa_labels + versicolor_labels
        
        df = pd.DataFrame(all_data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        df['species'] = all_labels
    
    # Filter for binary classification
    binary_df = df[df['species'].isin(['Iris-setosa', 'Iris-versicolor'])].copy()
    
    # Prepare features and labels
    X = binary_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    y = (binary_df['species'] == 'Iris-versicolor').astype(int).values
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print("Dataset Information:")
    print("-" * 30)
    print(f"Total samples: {len(X)}")
    print(f"Training samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Validation samples: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"Features: {list(binary_df.columns[:-1])}")
    print(f"Classes: Setosa (0), Versicolor (1)")
    print(f"Training class distribution: Setosa={np.sum(y_train==0)}, Versicolor={np.sum(y_train==1)}")
    print(f"Validation class distribution: Setosa={np.sum(y_val==0)}, Versicolor={np.sum(y_val==1)}")
    print()
    
    return X_train, X_val, y_train, y_val, binary_df


def create_presentation_plots(history):
    """Create plots suitable for presentation slides"""
    
    # Set up the plotting style for presentation
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 16
    })
    
    # Create accuracy plot
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['epoch'], history['train_accuracy'], 'b-', linewidth=2, label='Training', marker='o', markersize=4)
    plt.plot(history['epoch'], history['val_accuracy'], 'r-', linewidth=2, label='Validation', marker='s', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy per Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    
    # Create loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history['epoch'], history['train_loss'], 'b-', linewidth=2, label='Training', marker='o', markersize=4)
    plt.plot(history['epoch'], history['val_loss'], 'r-', linewidth=2, label='Validation', marker='s', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss per Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('slp_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create separate high-quality plots for slides
    
    # Accuracy plot only
    plt.figure(figsize=(10, 6))
    plt.plot(history['epoch'], history['train_accuracy'], 'b-', linewidth=3, label='Training Accuracy', marker='o', markersize=6)
    plt.plot(history['epoch'], history['val_accuracy'], 'r-', linewidth=3, label='Validation Accuracy', marker='s', markersize=6)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Single Layer Perceptron - Accuracy per Epoch', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig('accuracy_plot_for_slides.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Loss plot only
    plt.figure(figsize=(10, 6))
    plt.plot(history['epoch'], history['train_loss'], 'b-', linewidth=3, label='Training Loss', marker='o', markersize=6)
    plt.plot(history['epoch'], history['val_loss'], 'r-', linewidth=3, label='Validation Loss', marker='s', markersize=6)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Single Layer Perceptron - Loss per Epoch', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('loss_plot_for_slides.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_results_for_comparison(history):
    """Save results in format suitable for Google Sheets comparison"""
    
    results_df = pd.DataFrame({
        'Epoch': history['epoch'],
        'Training_Accuracy': history['train_accuracy'],
        'Training_Loss': history['train_loss'],
        'Validation_Accuracy': history['val_accuracy'],
        'Validation_Loss': history['val_loss']
    })
    
    # Round to appropriate decimal places
    results_df['Training_Accuracy'] = results_df['Training_Accuracy'].round(2)
    results_df['Training_Loss'] = results_df['Training_Loss'].round(6)
    results_df['Validation_Accuracy'] = results_df['Validation_Accuracy'].round(2)
    results_df['Validation_Loss'] = results_df['Validation_Loss'].round(6)
    
    # Save to CSV
    results_df.to_csv('python_slp_results.csv', index=False)
    
    print("Results saved to 'python_slp_results.csv'")
    print("This file can be used to compare with Google Sheets results")
    print()
    
    # Print summary statistics
    print("Summary Statistics:")
    print("-" * 40)
    print(f"Final Training Accuracy: {history['train_accuracy'][-1]:.2f}%")
    print(f"Final Validation Accuracy: {history['val_accuracy'][-1]:.2f}%")
    print(f"Final Training Loss: {history['train_loss'][-1]:.6f}")
    print(f"Final Validation Loss: {history['val_loss'][-1]:.6f}")
    print(f"Best Training Accuracy: {max(history['train_accuracy']):.2f}%")
    print(f"Best Validation Accuracy: {max(history['val_accuracy']):.2f}%")
    print(f"Lowest Training Loss: {min(history['train_loss']):.6f}")
    print(f"Lowest Validation Loss: {min(history['val_loss']):.6f}")
    
    return results_df


def create_data_visualization(df):
    """Create data distribution visualization"""
    
    plt.figure(figsize=(12, 8))
    
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    for i, feature in enumerate(features):
        plt.subplot(2, 2, i+1)
        
        setosa = df[df['species'] == 'Iris-setosa'][feature]
        versicolor = df[df['species'] == 'Iris-versicolor'][feature]
        
        plt.hist(setosa, alpha=0.7, label='Setosa', bins=15, color='skyblue', edgecolor='black')
        plt.hist(versicolor, alpha=0.7, label='Versicolor', bins=15, color='lightcoral', edgecolor='black')
        
        plt.xlabel(feature.replace('_', ' ').title())
        plt.ylabel('Frequency')
        plt.title(f'Distribution: {feature.replace("_", " ").title()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('iris_data_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function"""
    
    print("Single Layer Perceptron for Iris Dataset")
    print("=" * 60)
    print("Implementation for Deep Learning Assignment")
    print("University: UGM")
    print("Date: September 7, 2025")
    print("=" * 60)
    print()
    
    # Load and prepare data
    X_train, X_val, y_train, y_val, df = load_and_split_data()
    
    # Create data visualization
    create_data_visualization(df)
    
    # Create and train model
    model = IrisSLP(learning_rate=0.1, random_seed=42)
    history = model.fit(X_train, y_train, X_val, y_val, epochs=100)
    
    # Create plots for presentation
    create_presentation_plots(history)
    
    # Save results for comparison
    results_df = save_results_for_comparison(history)
    
    # Test the model
    print("\nModel Testing:")
    print("-" * 30)
    
    # Make predictions
    train_pred = model.predict_classes(X_train)
    val_pred = model.predict_classes(X_val)
    
    train_accuracy = np.mean(train_pred == y_train) * 100
    val_accuracy = np.mean(val_pred == y_val) * 100
    
    print(f"Final Training Accuracy: {train_accuracy:.2f}%")
    print(f"Final Validation Accuracy: {val_accuracy:.2f}%")
    
    # Show sample predictions
    print("\nSample Predictions on Validation Set:")
    print("-" * 50)
    val_probabilities = model.predict(X_val)
    
    for i in range(min(10, len(X_val))):
        prob = val_probabilities[i]
        pred_class = val_pred[i]
        true_class = y_val[i]
        
        pred_name = "Versicolor" if pred_class == 1 else "Setosa"
        true_name = "Versicolor" if true_class == 1 else "Setosa"
        status = "✓" if pred_class == true_class else "✗"
        
        print(f"Sample {i+1:2d}: Predicted: {pred_name:10s} ({prob:.3f}) | "
              f"True: {true_name:10s} | {status}")
    
    print("\n" + "=" * 60)
    print("Files Generated:")
    print("- python_slp_results.csv (for Google Sheets comparison)")
    print("- accuracy_plot_for_slides.png (for presentation)")
    print("- loss_plot_for_slides.png (for presentation)")
    print("- slp_training_results.png (combined plots)")
    print("- iris_data_distribution.png (data visualization)")
    print("=" * 60)
    print("Assignment completed successfully!")
    
    return model, history, results_df


if __name__ == "__main__":
    model, history, results = main()
