"""
Script to analyze HuBERT layer performance
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from config import LANGUAGES, BATCH_SIZE

class SimpleClassifier(nn.Module):
    """
    Simple classifier for layer analysis
    """
    def __init__(self, input_dim, num_classes, hidden_dim=128):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Global average pooling over time dimension
        x = torch.mean(x, dim=1)  # (batch, time, dim) -> (batch, dim)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def evaluate_layer(features_dir, layer_name, device):
    """
    Evaluate performance of a specific layer
    
    Args:
        features_dir (str): Directory containing features
        layer_name (str): Name of layer to evaluate
        device (torch.device): Device to use
        
    Returns:
        float: Accuracy score
    """
    # Import here to avoid circular imports
    from train_hubert import HuBERTDataset
    
    # Create dataset for this layer
    dataset = HuBERTDataset(features_dir, LANGUAGES, layer_name)
    
    if len(dataset) == 0:
        print(f"No features found for {layer_name}")
        return 0.0
    
    # Split dataset (80% train, 20% test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Get input dimension from first sample
    sample_features, _ = dataset[0]
    input_dim = sample_features.shape[1]
    
    # Initialize model
    model = SimpleClassifier(input_dim, len(LANGUAGES))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Train for a few epochs (simplified training)
    model.train()
    for epoch in range(5):  # Only 5 epochs for quick evaluation
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    return accuracy

def analyze_all_layers(features_dir, num_layers=13):
    """
    Analyze performance of all HuBERT layers
    
    Args:
        features_dir (str): Directory containing features
        num_layers (int): Number of layers to analyze
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    layer_accuracies = []
    
    print("Analyzing HuBERT layer performance...")
    print("=" * 50)
    
    for layer_idx in range(num_layers):
        layer_name = f"layer_{layer_idx}"
        print(f"Evaluating {layer_name}...")
        
        try:
            accuracy = evaluate_layer(features_dir, layer_name, device)
            layer_accuracies.append((layer_idx, accuracy))
            print(f"  Accuracy: {accuracy:.4f}")
        except Exception as e:
            print(f"  Error evaluating {layer_name}: {e}")
            layer_accuracies.append((layer_idx, 0.0))
    
    print("\n" + "=" * 50)
    print("LAYER PERFORMANCE SUMMARY")
    print("=" * 50)
    
    # Sort by accuracy
    layer_accuracies.sort(key=lambda x: x[1], reverse=True)
    
    for layer_idx, accuracy in layer_accuracies:
        marker = " ← BEST" if layer_idx == layer_accuracies[0][0] else ""
        print(f"Layer {layer_idx:2d}: {accuracy:.4f}{marker}")
    
    best_layer, best_accuracy = layer_accuracies[0]
    print(f"\nBest layer: {best_layer} with accuracy: {best_accuracy:.4f}")
    
    return best_layer, best_accuracy

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze HuBERT layer performance")
    parser.add_argument("--features", type=str, default="data/features/hubert", 
                        help="Directory containing HuBERT features")
    parser.add_argument("--layers", type=int, default=13,
                        help="Number of layers to analyze")
    
    args = parser.parse_args()
    
    analyze_all_layers(args.features, args.layers)