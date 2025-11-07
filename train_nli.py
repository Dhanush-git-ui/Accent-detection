"""
Main training script for Native Language Identification
"""
import torch
from torch.utils.data import DataLoader
from config import BATCH_SIZE, LANGUAGES
from utils.helpers import get_language_label_encoder

# Try to import dataset module, with fallback for torchaudio issues
try:
    from data.dataset import IndianAccentDataset
    TORCHAUDIO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: torchaudio not available: {e}")
    TORCHAUDIO_AVAILABLE = False
    # Create a dummy dataset class for testing
    from torch.utils.data import Dataset
    class IndianAccentDataset(Dataset):
        def __init__(self, audio_paths, labels, feature_type='mfcc'):
            self.audio_paths = audio_paths
            self.labels = labels
            self.feature_type = feature_type
        
        def __len__(self):
            return len(self.audio_paths)
        
        def __getitem__(self, idx):
            # Return dummy data
            if self.feature_type == 'mfcc':
                return torch.randn(1, 13, 1000), self.labels[idx]
            else:  # hubert
                return torch.randn(1, 768, 1000), self.labels[idx]

# Import models
from models.classifiers import MFCCClassifier, HubertClassifier
from models.trainer import ModelTrainer

def load_dataset(data_dir, feature_type='mfcc'):
    """
    Load and prepare dataset
    
    Args:
        data_dir (str): Directory containing dataset
        feature_type (str): Type of features to use
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # This is a placeholder implementation
    # In practice, you would load the actual dataset from the Hugging Face repository
    # https://huggingface.co/datasets/DarshanaS/IndicAccentDb
    
    # Placeholder data - replace with actual data loading
    train_audio_paths = ["dummy_path"] * 100  # 100 dummy paths
    train_labels = [i % len(LANGUAGES) for i in range(100)]
    val_audio_paths = ["dummy_path"] * 20  # 20 dummy paths
    val_labels = [i % len(LANGUAGES) for i in range(20)]
    test_audio_paths = ["dummy_path"] * 30  # 30 dummy paths
    test_labels = [i % len(LANGUAGES) for i in range(30)]
    
    train_dataset = IndianAccentDataset(train_audio_paths, train_labels, feature_type=feature_type)
    val_dataset = IndianAccentDataset(val_audio_paths, val_labels, feature_type=feature_type)
    test_dataset = IndianAccentDataset(test_audio_paths, test_labels, feature_type=feature_type)
    
    return train_dataset, val_dataset, test_dataset

def train_model(feature_type='mfcc'):
    """
    Train NLI model
    
    Args:
        feature_type (str): Type of features to use ('mfcc' or 'hubert')
    """
    print(f"Training model with {feature_type} features...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    train_dataset, val_dataset, test_dataset = load_dataset('data/', feature_type=feature_type)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    num_classes = len(LANGUAGES)
    if feature_type == 'mfcc':
        model = MFCCClassifier(num_classes=num_classes)
    else:  # hubert
        model = HubertClassifier(num_classes=num_classes)
    
    # Initialize trainer
    trainer = ModelTrainer(model, device=device)
    
    # Train model (for demo, we'll just do a few epochs)
    save_path = f"models/nli_{feature_type}_best.pth"
    print("Starting training (demo mode - limited epochs)...")
    
    # For demo purposes, we'll limit to 3 epochs
    trainer.train(train_loader, val_loader, epochs=3, save_path=save_path)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    results = trainer.evaluate(test_loader, class_names=LANGUAGES)
    
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print("Classification Report:")
    for lang in LANGUAGES:
        if lang in results['classification_report']:
            precision = results['classification_report'][lang]['precision']
            recall = results['classification_report'][lang]['recall']
            f1 = results['classification_report'][lang]['f1-score']
            print(f"  {lang.capitalize()}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    
    # Note: We're not plotting in this workaround version
    # trainer.plot_training_history()
    # trainer.plot_confusion_matrix(results['confusion_matrix'], class_names=LANGUAGES)
    
    return model, trainer

def compare_features():
    """
    Compare MFCC and HuBERT features
    """
    print("Comparing MFCC and HuBERT features...")
    
    # Train with MFCC features
    print("\n--- Training with MFCC Features ---")
    mfcc_model, mfcc_trainer = train_model('mfcc')
    
    # Train with HuBERT features
    print("\n--- Training with HuBERT Features ---")
    hubert_model, hubert_trainer = train_model('hubert')
    
    print("\nFeature comparison completed!")

def main():
    """
    Main function
    """
    import argparse
    parser = argparse.ArgumentParser(description="Train Native Language Identification Model")
    parser.add_argument("--feature-type", type=str, default="hubert", 
                        choices=["mfcc", "hubert"], 
                        help="Type of features to use")
    parser.add_argument("--compare", action="store_true",
                        help="Compare MFCC and HuBERT features")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_features()
    else:
        model, trainer = train_model(args.feature_type)

if __name__ == "__main__":
    main()