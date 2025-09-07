"""
Single Layer Perceptron (SLP) Implementation for Iris Dataset
Author: [Your Name]
NIM: [Your NIM]
Date: September 7, 2025

This implementation creates a Single Layer Perceptron from scratch using NumPy
for binary classification of Iris dataset (Setosa vs Versicolor).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns


class SingleLayerPerceptron:
    """
    Single Layer Perceptron implementation with sigmoid activation function
    """
    
    def __init__(self, learning_rate=0.1, max_epochs=100, random_seed=42):
        """
        Initialize the SLP
        
        Parameters:
        learning_rate (float): Learning rate for weight updates
        max_epochs (int): Maximum number of training epochs
        random_seed (int): Random seed for reproducibility
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.random_seed = random_seed
        
        # Initialize weights and bias
        np.random.seed(random_seed)
        self.weights = None
        self.bias = None
        
        # Training history
        self.train_accuracy_history = []
        self.train_loss_history = []
        self.val_accuracy_history = []
        self.val_loss_history = []
        self.epoch_history = []
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def forward_pass(self, X):
        """
        Forward pass through the network
        
        Parameters:
        X (numpy.ndarray): Input features
        
        Returns:
        numpy.ndarray: Output predictions
        """
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def compute_loss(self, y_true, y_pred):
        """
        Compute binary cross-entropy loss
        
        Parameters:
        y_true (numpy.ndarray): True labels
        y_pred (numpy.ndarray): Predicted probabilities
        
        Returns:
        float: Mean loss
        """
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # Binary cross-entropy loss
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def compute_accuracy(self, y_true, y_pred):
        """
        Compute accuracy
        
        Parameters:
        y_true (numpy.ndarray): True labels
        y_pred (numpy.ndarray): Predicted probabilities
        
        Returns:
        float: Accuracy percentage
        """
        predictions = (y_pred >= 0.5).astype(int)
        accuracy = np.mean(predictions == y_true) * 100
        return accuracy
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the Single Layer Perceptron
        
        Parameters:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training labels
        X_val (numpy.ndarray): Validation features (optional)
        y_val (numpy.ndarray): Validation labels (optional)
        """
        n_samples, n_features = X_train.shape
        
        # Initialize weights and bias
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0.5  # Initialize bias as 0.5 (as shown in the CSV data)
        
        print("Training Single Layer Perceptron...")
        print("=" * 50)
        
        for epoch in range(self.max_epochs):
            # Forward pass
            y_pred_train = self.forward_pass(X_train)
            
            # Compute training loss and accuracy
            train_loss = self.compute_loss(y_train, y_pred_train)
            train_accuracy = self.compute_accuracy(y_train, y_pred_train)
            
            # Compute gradients
            error = y_pred_train - y_train
            dw = np.dot(X_train.T, error) / n_samples
            db = np.mean(error)
            
            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Store training history
            self.epoch_history.append(epoch + 1)
            self.train_accuracy_history.append(train_accuracy)
            self.train_loss_history.append(train_loss)
            
            # Validation metrics (if validation data provided)
            if X_val is not None and y_val is not None:
                y_pred_val = self.forward_pass(X_val)
                val_loss = self.compute_loss(y_val, y_pred_val)
                val_accuracy = self.compute_accuracy(y_val, y_pred_val)
                
                self.val_accuracy_history.append(val_accuracy)
                self.val_loss_history.append(val_loss)
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == 0:
                if X_val is not None:
                    print(f"Epoch {epoch + 1:3d}: Train Loss: {train_loss:.6f}, "
                          f"Train Acc: {train_accuracy:.2f}%, "
                          f"Val Loss: {val_loss:.6f}, Val Acc: {val_accuracy:.2f}%")
                else:
                    print(f"Epoch {epoch + 1:3d}: Train Loss: {train_loss:.6f}, "
                          f"Train Acc: {train_accuracy:.2f}%")
        
        print("=" * 50)
        print("Training completed!")
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Parameters:
        X (numpy.ndarray): Input features
        
        Returns:
        numpy.ndarray: Predicted probabilities
        """
        return self.forward_pass(X)
    
    def predict_classes(self, X):
        """
        Predict class labels
        
        Parameters:
        X (numpy.ndarray): Input features
        
        Returns:
        numpy.ndarray: Predicted class labels
        """
        probabilities = self.predict(X)
        return (probabilities >= 0.5).astype(int)


def load_and_preprocess_data():
    """
    Load and preprocess the Iris dataset
    
    Returns:
    tuple: Processed features and labels
    """
    # Load the data
    data_path = "/media/nugroho-adi-susanto/Windows-SSD/Users/Nugroho Adi Susanto/Documents/UGM/Kuliah/AI/Deep Learning/SLP-rev.xlsx - Data.csv"
    
    # Read the CSV file
    df = pd.read_csv(data_path, header=None, 
                     names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
    
    print("Dataset Info:")
    print("=" * 40)
    print(f"Total samples: {len(df)}")
    print(f"Features: {df.columns[:-1].tolist()}")
    print(f"Classes: {df['species'].unique()}")
    print(f"Class distribution:\n{df['species'].value_counts()}")
    print()
    
    # For binary classification: Setosa (0) vs Versicolor (1)
    # Filter only Setosa and Versicolor classes
    binary_df = df[df['species'].isin(['Iris-setosa', 'Iris-versicolor'])].copy()
    
    # Prepare features (X) and labels (y)
    X = binary_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    
    # Convert labels to binary: Setosa = 0, Versicolor = 1
    y = (binary_df['species'] == 'Iris-versicolor').astype(int).values
    
    print(f"Binary classification dataset:")
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Setosa samples (0): {np.sum(y == 0)}")
    print(f"Versicolor samples (1): {np.sum(y == 1)}")
    print()
    
    return X, y, binary_df


def plot_training_history(slp):
    """
    Plot training and validation history
    
    Parameters:
    slp (SingleLayerPerceptron): Trained SLP model
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(slp.epoch_history, slp.train_accuracy_history, 'b-', label='Training Accuracy', linewidth=2)
    if slp.val_accuracy_history:
        ax1.plot(slp.epoch_history, slp.val_accuracy_history, 'r-', label='Validation Accuracy', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Training and Validation Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(slp.epoch_history, slp.train_loss_history, 'b-', label='Training Loss', linewidth=2)
    if slp.val_loss_history:
        ax2.plot(slp.epoch_history, slp.val_loss_history, 'r-', label='Validation Loss', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training and Validation Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('slp_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_data_distribution(X, y, df):
    """
    Plot data distribution
    
    Parameters:
    X (numpy.ndarray): Features
    y (numpy.ndarray): Labels
    df (pandas.DataFrame): Original dataframe
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    for i, feature in enumerate(feature_names):
        ax = axes[i//2, i%2]
        
        setosa_data = df[df['species'] == 'Iris-setosa'][feature]
        versicolor_data = df[df['species'] == 'Iris-versicolor'][feature]
        
        ax.hist(setosa_data, alpha=0.7, label='Setosa (0)', bins=15, color='blue')
        ax.hist(versicolor_data, alpha=0.7, label='Versicolor (1)', bins=15, color='red')
        
        ax.set_xlabel(feature.replace('_', ' ').title())
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {feature.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_results_to_csv(slp):
    """
    Save training results to CSV file
    
    Parameters:
    slp (SingleLayerPerceptron): Trained SLP model
    """
    results_df = pd.DataFrame({
        'Epoch': slp.epoch_history,
        'Train_Accuracy': slp.train_accuracy_history,
        'Train_Loss': slp.train_loss_history,
        'Val_Accuracy': slp.val_accuracy_history if slp.val_accuracy_history else [None] * len(slp.epoch_history),
        'Val_Loss': slp.val_loss_history if slp.val_loss_history else [None] * len(slp.epoch_history)
    })
    
    results_df.to_csv('slp_training_results.csv', index=False)
    print("Training results saved to 'slp_training_results.csv'")


def main():
    """
    Main function to run the SLP training and evaluation
    """
    print("Single Layer Perceptron for Iris Dataset")
    print("="*50)
    
    # Load and preprocess data
    X, y, df = load_and_preprocess_data()
    
    # Plot data distribution
    plot_data_distribution(X, y, df)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print()
    
    # Standardize the features (optional but recommended)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Create and train SLP
    slp = SingleLayerPerceptron(learning_rate=0.1, max_epochs=100, random_seed=42)
    slp.fit(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Make predictions on validation set
    val_predictions = slp.predict(X_val_scaled)
    val_class_predictions = slp.predict_classes(X_val_scaled)
    
    # Calculate final metrics
    final_train_accuracy = slp.compute_accuracy(y_train, slp.predict(X_train_scaled))
    final_val_accuracy = slp.compute_accuracy(y_val, val_predictions)
    
    print(f"\nFinal Results:")
    print(f"Training Accuracy: {final_train_accuracy:.2f}%")
    print(f"Validation Accuracy: {final_val_accuracy:.2f}%")
    
    # Print model parameters
    print(f"\nFinal Model Parameters:")
    print(f"Weights: {slp.weights}")
    print(f"Bias: {slp.bias}")
    
    # Plot training history
    plot_training_history(slp)
    
    # Save results to CSV
    save_results_to_csv(slp)
    
    # Print some example predictions
    print("\nSample Predictions:")
    print("="*30)
    for i in range(min(10, len(X_val))):
        prob = val_predictions[i]
        pred_class = val_class_predictions[i]
        true_class = y_val[i]
        class_name = "Versicolor" if pred_class == 1 else "Setosa"
        true_name = "Versicolor" if true_class == 1 else "Setosa"
        correct = "✓" if pred_class == true_class else "✗"
        
        print(f"Sample {i+1}: Predicted: {class_name} ({prob:.3f}) | "
              f"True: {true_name} | {correct}")


if __name__ == "__main__":
    main()
