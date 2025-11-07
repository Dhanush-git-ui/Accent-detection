"""
Script to train models on HuBERT features
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from config import LANGUAGES, BATCH_SIZE, LEARNING_RATE, EPOCHS

class HuBERTDataset(Dataset):
    """
    Dataset class for HuBERT features
    """
    def __init__(self, features_dir, labels, layer_name="layer_7"):
        self.features_dir = features_dir
        self.labels = labels
        self.layer_name = layer_name
        self.file_list = []
        self.label_list = []
        
        # Walk through features directory
        for root, dirs, files in os.walk(features_dir):
            for file in files:
                if file.endswith(f"_{layer_name}.npy"):
                    file_path = os.path.join(root, file)
                    # Extract label from directory structure
                    label = os.path.basename(os.path.dirname(file_path))
                    if label in LANGUAGES:
                        self.file_list.append(file_path)
                        self.label_list.append(label)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.label_list)
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # Load features
        features = np.load(self.file_list[idx])
        
        # Convert to tensor
        features = torch.FloatTensor(features)
        
        # Pad or truncate to fixed length (e.g., 1000 frames)
        target_length = 1000
        if features.shape[0] > target_length:
            features = features[:target_length, :]
        elif features.shape[0] < target_length:
            pad_length = target_length - features.shape[0]
            features = torch.nn.functional.pad(features, (0, 0, 0, pad_length))
        
        label = self.encoded_labels[idx]
        return features, label

class HuBERTClassifier(nn.Module):
    """
    Classifier for HuBERT features
    """
    def __init__(self, num_classes, input_dim=768, hidden_dim=512, dropout_rate=0.5):
        super(HuBERTClassifier, self).__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, (hidden, _) = self.lstm(x)
        
        # Use the last hidden state
        # hidden shape: (num_layers * num_directions, batch, hidden_size)
        # Concatenate forward and backward hidden states
        batch_size = hidden.shape[1]
        hidden = hidden.view(1, 2, batch_size, -1)  # (num_layers, num_directions, batch, hidden_size)
        forward_hidden = hidden[0, 0, :, :]  # (batch, hidden_size)
        backward_hidden = hidden[0, 1, :, :]  # (batch, hidden_size)
        concat_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)  # (batch, hidden_size * 2)
        
        x = self.dropout(concat_hidden)
        x = self.fc(x)
        
        return x

def train_model(model, train_loader, val_loader, device, epochs=EPOCHS):
    """
    Train the model
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        device (torch.device): Device to use for training
        epochs (int): Number of epochs to train
        
    Returns:
        tuple: (best_model, best_val_acc, train_losses, val_losses)
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_acc = 0.0
    best_model_state = None
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_val_acc, train_losses, val_losses

def evaluate_model(model, test_loader, device, class_names):
    """
    Evaluate the model on test set
    
    Args:
        model (nn.Module): Trained model
        test_loader (DataLoader): Test data loader
        device (torch.device): Device to use for evaluation
        class_names (list): List of class names
        
    Returns:
        dict: Evaluation metrics
    """
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
    report = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)
    cm = confusion_matrix(all_labels, all_predictions)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm
    }

def save_model(model, optimizer, epoch, loss, filepath):
    """
    Save model checkpoint
    
    Args:
        model (nn.Module): Model to save
        optimizer (optim.Optimizer): Optimizer
        epoch (int): Current epoch
        loss (float): Current loss
        filepath (str): Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")

def compare_with_mfcc(mfcc_accuracy, hubert_accuracy):
    """
    Compare HuBERT performance with MFCC baseline
    
    Args:
        mfcc_accuracy (float): MFCC baseline accuracy
        hubert_accuracy (float): HuBERT accuracy
    """
    print("\n" + "="*50)
    print("PERFORMANCE COMPARISON")
    print("="*50)
    print(f"MFCC Baseline Accuracy: {mfcc_accuracy:.4f}")
    print(f"HuBERT Accuracy: {hubert_accuracy:.4f}")
    print(f"Improvement: {hubert_accuracy - mfcc_accuracy:.4f}")
    print(f"Relative Improvement: {((hubert_accuracy - mfcc_accuracy) / mfcc_accuracy) * 100:.2f}%")
    
    if hubert_accuracy > mfcc_accuracy:
        print("✅ HuBERT outperforms MFCC baseline!")
    else:
        print("⚠ MFCC baseline performs better than HuBERT")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train model on HuBERT features")
    parser.add_argument("--features", type=str, default="data/features/hubert", 
                        help="Directory containing HuBERT features")
    parser.add_argument("--layer", type=str, default="layer_7",
                        help="Layer to use for features (e.g., layer_7)")
    parser.add_argument("--model-path", type=str, default="models/hubert_classifier.pth",
                        help="Path to save trained model")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--mfcc-baseline", type=float, default=0.72,
                        help="MFCC baseline accuracy for comparison")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    print("Creating dataset...")
    dataset = HuBERTDataset(args.features, LANGUAGES, args.layer)
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) == 0:
        print("No features found. Creating dummy data...")
        # Create dummy features for testing
        os.makedirs(f"data/features/hubert/dummy", exist_ok=True)
        for i in range(50):
            # Create dummy HuBERT features (1000 frames, 768 dimensions)
            dummy_features = np.random.randn(1000, 768)
            np.save(f"data/features/hubert/dummy/dummy_{i:03d}_{args.layer}.npy", dummy_features)
        
        dataset = HuBERTDataset(args.features, LANGUAGES, args.layer)
        print(f"Dummy dataset size: {len(dataset)}")
    
    # Split dataset (80% train, 10% validation, 10% test)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    model = HuBERTClassifier(num_classes=len(LANGUAGES))
    print(f"Model initialized with {len(LANGUAGES)} classes")
    
    # Train model
    print("Starting training...")
    trained_model, best_val_acc, train_losses, val_losses = train_model(
        model, train_loader, val_loader, device, args.epochs
    )
    
    print(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    results = evaluate_model(trained_model, test_loader, device, LANGUAGES)
    
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    for lang in LANGUAGES:
        if lang in results['classification_report']:
            precision = results['classification_report'][lang]['precision']
            recall = results['classification_report'][lang]['recall']
            f1 = results['classification_report'][lang]['f1-score']
            print(f"  {lang.capitalize()}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    
    # Compare with MFCC baseline
    compare_with_mfcc(args.mfcc_baseline, results['accuracy'])
    
    # Save model
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    save_model(trained_model, optimizer, args.epochs, 0.0, args.model_path)
    
    print("HuBERT model training complete!")