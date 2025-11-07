"""
Script to train baseline model on MFCC features
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

class MFCCDataset(Dataset):
    """
    Dataset class for MFCC features
    """
    def __init__(self, features_dir, labels):
        self.features_dir = features_dir
        self.labels = labels
        self.file_list = []
        self.label_list = []
        
        # Walk through features directory
        for root, dirs, files in os.walk(features_dir):
            for file in files:
                if file.endswith('.npy'):
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
        
        # Convert to tensor and ensure consistent shape
        features = torch.FloatTensor(features)
        
        # Pad or truncate to fixed length (e.g., 1000 frames)
        target_length = 1000
        if features.shape[1] > target_length:
            features = features[:, :target_length]
        elif features.shape[1] < target_length:
            pad_length = target_length - features.shape[1]
            features = torch.nn.functional.pad(features, (0, pad_length))
        
        label = self.encoded_labels[idx]
        return features, label

class MFCCBaselineModel(nn.Module):
    """
    Simple CNN model for MFCC features
    """
    def __init__(self, num_classes, input_channels=13, dropout_rate=0.5):
        super(MFCCBaselineModel, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        
        # Calculate the size after convolutions and pooling
        self.fc_input_size = 256 * 125  # Adjust based on actual calculations
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        
        x = x.view(-1, self.fc_input_size)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train baseline model on MFCC features")
    parser.add_argument("--features", type=str, default="data/features/mfcc", 
                        help="Directory containing MFCC features")
    parser.add_argument("--model-path", type=str, default="models/mfcc_baseline.pth",
                        help="Path to save trained model")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dummy dataset for demonstration
    print("Creating dummy dataset...")
    dataset = MFCCDataset(args.features, LANGUAGES)
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) == 0:
        print("No features found. Creating dummy data...")
        # Create dummy features for testing
        os.makedirs("data/features/mfcc/dummy", exist_ok=True)
        for i in range(50):
            # Create dummy MFCC features
            dummy_features = np.random.randn(13, 1000)
            np.save(f"data/features/mfcc/dummy/dummy_{i:03d}.npy", dummy_features)
        
        dataset = MFCCDataset(args.features, LANGUAGES)
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
    model = MFCCBaselineModel(num_classes=len(LANGUAGES))
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
    
    # Save model
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    save_model(trained_model, optimizer, args.epochs, 0.0, args.model_path)
    
    print("Baseline model training complete!")