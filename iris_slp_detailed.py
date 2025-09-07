"""
Single Layer Perceptron (SLP) - Detailed Implementation
Matching the format shown in the CSV files
Author: [Your Name]
NIM: [Your NIM]
Date: September 7, 2025

This implementation shows detailed step-by-step calculations
similar to the format in the provided CSV files.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DetailedSLP:
    """
    Detailed Single Layer Perceptron with step-by-step calculations
    """
    
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = np.array([0.5, 0.5, 0.5, 0.5])  # teta1, teta2, teta3, teta4
        self.bias = 0.5  # bias weight
        
        # History tracking
        self.history = []
        self.epoch_summary = []
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        z = np.clip(z, -500, 500)  # Prevent overflow
        return 1 / (1 + np.exp(-z))
    
    def forward_pass(self, x):
        """
        Forward pass with detailed calculations
        
        Parameters:
        x (numpy.ndarray): Input features [x1, x2, x3, x4]
        
        Returns:
        tuple: (z, sigmoid_output, prediction)
        """
        # Add bias term (x0 = 1)
        x_with_bias = np.concatenate([[1], x])
        weights_with_bias = np.concatenate([[self.bias], self.weights])
        
        # Dot product: z = bias + x1*w1 + x2*w2 + x3*w3 + x4*w4
        z = np.dot(x_with_bias, weights_with_bias)
        
        # Sigmoid activation
        sigmoid_output = self.sigmoid(z)
        
        # Binary prediction (threshold = 0.5)
        prediction = 1 if sigmoid_output >= 0.5 else 0
        
        return z, sigmoid_output, prediction
    
    def calculate_error_and_gradients(self, x, target, sigmoid_output):
        """
        Calculate error and gradients for weight updates
        
        Parameters:
        x (numpy.ndarray): Input features
        target (int): Target label
        sigmoid_output (float): Output from sigmoid
        
        Returns:
        tuple: (error, gradients)
        """
        # Error = output - target
        error = sigmoid_output - target
        
        # Square error for this sample
        square_error = error ** 2
        
        # Gradients for weight updates
        # For sigmoid: gradient = error * sigmoid_output * (1 - sigmoid_output) * input
        sigmoid_derivative = sigmoid_output * (1 - sigmoid_output)
        
        # Gradients
        dbias = error * sigmoid_derivative
        dteta1 = error * sigmoid_derivative * x[0]  # x1
        dteta2 = error * sigmoid_derivative * x[1]  # x2
        dteta3 = error * sigmoid_derivative * x[2]  # x3
        dteta4 = error * sigmoid_derivative * x[3]  # x4
        
        gradients = {
            'dbias': dbias,
            'dteta1': dteta1,
            'dteta2': dteta2,
            'dteta3': dteta3,
            'dteta4': dteta4
        }
        
        return error, square_error, gradients
    
    def update_weights(self, gradients):
        """Update weights using gradients"""
        self.bias -= self.learning_rate * gradients['dbias']
        self.weights[0] -= self.learning_rate * gradients['dteta1']
        self.weights[1] -= self.learning_rate * gradients['dteta2']
        self.weights[2] -= self.learning_rate * gradients['dteta3']
        self.weights[3] -= self.learning_rate * gradients['dteta4']
    
    def train_epoch(self, X, y, epoch_num):
        """
        Train for one epoch with detailed logging
        
        Parameters:
        X (numpy.ndarray): Training features
        y (numpy.ndarray): Training labels
        epoch_num (int): Epoch number
        
        Returns:
        tuple: (accuracy, total_loss)
        """
        epoch_data = []
        total_square_error = 0
        correct_predictions = 0
        
        for i, (x, target) in enumerate(zip(X, y)):
            # Forward pass
            z, sigmoid_output, prediction = self.forward_pass(x)
            
            # Calculate error and gradients
            error, square_error, gradients = self.calculate_error_and_gradients(x, target, sigmoid_output)
            
            # Store detailed information
            sample_data = {
                'epoch': epoch_num,
                'sample': i + 1,
                'x0': 1,  # bias input
                'x1': x[0],
                'x2': x[1],
                'x3': x[2],
                'x4': x[3],
                'target': target,
                'bias': self.bias,
                'teta1': self.weights[0],
                'teta2': self.weights[1],
                'teta3': self.weights[2],
                'teta4': self.weights[3],
                'z': z,
                'sigmoid': sigmoid_output,
                'prediction': prediction,
                'error': error,
                'square_error': square_error,
                'dbias': gradients['dbias'],
                'dteta1': gradients['dteta1'],
                'dteta2': gradients['dteta2'],
                'dteta3': gradients['dteta3'],
                'dteta4': gradients['dteta4']
            }
            
            epoch_data.append(sample_data)
            
            # Update weights
            self.update_weights(gradients)
            
            # Track performance
            total_square_error += square_error
            if prediction == target:
                correct_predictions += 1
        
        # Calculate epoch metrics
        accuracy = (correct_predictions / len(X)) * 100
        mean_square_error = total_square_error / len(X)
        
        # Store epoch data
        self.history.extend(epoch_data)
        
        # Store epoch summary
        epoch_summary = {
            'epoch': epoch_num,
            'accuracy': accuracy,
            'mse': mean_square_error,
            'final_bias': self.bias,
            'final_weights': self.weights.copy()
        }
        self.epoch_summary.append(epoch_summary)
        
        return accuracy, mean_square_error
    
    def train(self, X, y, epochs=5):
        """
        Train the SLP for specified epochs
        
        Parameters:
        X (numpy.ndarray): Training features
        y (numpy.ndarray): Training labels
        epochs (int): Number of training epochs
        """
        print("Training Single Layer Perceptron")
        print("=" * 50)
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Initial Bias: {self.bias}")
        print(f"Initial Weights: {self.weights}")
        print()
        
        for epoch in range(1, epochs + 1):
            accuracy, mse = self.train_epoch(X, y, epoch)
            
            print(f"Epoch {epoch}: Accuracy = {accuracy:.2f}%, MSE = {mse:.6f}")
            print(f"  Bias: {self.bias:.6f}")
            print(f"  Weights: {self.weights}")
            print()
    
    def save_detailed_results(self, filename="detailed_slp_results.csv"):
        """Save detailed training results to CSV"""
        df = pd.DataFrame(self.history)
        df.to_csv(filename, index=False)
        print(f"Detailed results saved to {filename}")
    
    def save_epoch_summary(self, filename="epoch_summary.csv"):
        """Save epoch summary to CSV"""
        df = pd.DataFrame(self.epoch_summary)
        df.to_csv(filename, index=False)
        print(f"Epoch summary saved to {filename}")
    
    def predict(self, X):
        """Make predictions on new data"""
        predictions = []
        probabilities = []
        
        for x in X:
            z, sigmoid_output, prediction = self.forward_pass(x)
            predictions.append(prediction)
            probabilities.append(sigmoid_output)
        
        return np.array(predictions), np.array(probabilities)


def load_iris_data():
    """Load and prepare Iris data for binary classification"""
    data_path = "/media/nugroho-adi-susanto/Windows-SSD/Users/Nugroho Adi Susanto/Documents/UGM/Kuliah/AI/Deep Learning/SLP-rev.xlsx - Data.csv"
    
    # Read the CSV file
    df = pd.read_csv(data_path, header=None, 
                     names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])
    
    # Filter for binary classification: Setosa vs Versicolor
    binary_df = df[df['species'].isin(['Iris-setosa', 'Iris-versicolor'])].copy()
    
    # Prepare features and labels
    X = binary_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    y = (binary_df['species'] == 'Iris-versicolor').astype(int).values
    
    print("Dataset Information:")
    print(f"Total samples: {len(X)}")
    print(f"Setosa (0): {np.sum(y == 0)}")
    print(f"Versicolor (1): {np.sum(y == 1)}")
    print(f"Features: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']")
    print()
    
    return X, y


def plot_training_progress(slp):
    """Plot training progress"""
    epochs = [summary['epoch'] for summary in slp.epoch_summary]
    accuracies = [summary['accuracy'] for summary in slp.epoch_summary]
    mses = [summary['mse'] for summary in slp.epoch_summary]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot accuracy
    ax1.plot(epochs, accuracies, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Training Accuracy per Epoch')
    ax1.grid(True)
    ax1.set_ylim(0, 105)
    
    # Plot MSE
    ax2.plot(epochs, mses, 'r-o', linewidth=2, markersize=8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Square Error')
    ax2.set_title('Training MSE per Epoch')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('detailed_slp_training.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function"""
    print("Detailed Single Layer Perceptron Implementation")
    print("=" * 60)
    
    # Load data
    X, y = load_iris_data()
    
    # Create and train SLP
    slp = DetailedSLP(learning_rate=0.1)
    slp.train(X, y, epochs=5)
    
    # Make predictions on training data
    predictions, probabilities = slp.predict(X)
    
    # Calculate final accuracy
    final_accuracy = np.mean(predictions == y) * 100
    print(f"Final Training Accuracy: {final_accuracy:.2f}%")
    
    # Show some sample predictions
    print("\nSample Predictions:")
    print("-" * 40)
    for i in range(min(10, len(X))):
        species = "Versicolor" if predictions[i] == 1 else "Setosa"
        true_species = "Versicolor" if y[i] == 1 else "Setosa"
        correct = "✓" if predictions[i] == y[i] else "✗"
        print(f"Sample {i+1}: {species} ({probabilities[i]:.3f}) | True: {true_species} {correct}")
    
    # Save results
    slp.save_detailed_results()
    slp.save_epoch_summary()
    
    # Plot training progress
    plot_training_progress(slp)
    
    print("\nTraining completed!")
    print("Files generated:")
    print("- detailed_slp_results.csv (step-by-step calculations)")
    print("- epoch_summary.csv (epoch-wise summary)")
    print("- detailed_slp_training.png (training plots)")


if __name__ == "__main__":
    main()
