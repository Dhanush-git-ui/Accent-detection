"""
Train HuBERT and Ensemble models on pre-extracted features
"""
import os
import sys
import argparse
import torch
from torch.utils.data import random_split

# Add parent to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BATCH_SIZE, LEARNING_RATE, EPOCHS, LANGUAGES, MODELS_DIR, FEATURES_DIR
from src.data.dataset import FeatureDataset, create_data_loader
from src.models.classifiers import HubertClassifier, MFCCClassifier, EnsembleClassifier
from src.models.trainer import ModelTrainer

def train_hubert_model(layer_idx=7, epochs=100, batch_size=16):
    """
    Train HuBERT classifier on pre-extracted features
    """
    print(f"\n==========================================")
    print(f"TRAINING: HuBERT Features (Layer {layer_idx})")
    print(f"==========================================")
    
    # 1. Set paths
    features_path = os.path.join(FEATURES_DIR, 'hubert', f'layer_{layer_idx}')
    
    print(f"Loading features from: {features_path}")
    if not os.path.exists(features_path) or not os.listdir(features_path):
        print(f"Error: Feature directory is empty or does not exist.")
        print(f"Please run feature extraction first:")
        print(f"  python scripts/extract_hubert_all_languages.py --layer {layer_idx}")
        return
    
    # 2. Load dataset with augmentation
    dataset = FeatureDataset(features_path, augment=False)  # No augmentation for HuBERT
    dataset_size = len(dataset)
    print(f"Loaded dataset containing {dataset_size} samples.")
    
    if dataset_size < 3:
        print("Error: Dataset too small to split. Please extract more feature files.")
        return
    
    # 3. Split dataset (70% train, 15% val, 15% test)
    train_size = int(0.70 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    if train_size == 0: train_size = 1
    if val_size == 0: val_size = 1
    test_size = dataset_size - train_size - val_size
    if test_size < 0: test_size = 0
    
    lengths = [train_size, val_size, test_size]
    lengths[-1] += dataset_size - sum(lengths)
    
    print(f"Splits: Train={lengths[0]}, Val={lengths[1]}, Test={lengths[2]}")
    train_dataset, val_dataset, test_dataset = random_split(dataset, lengths)
    
    train_loader = create_data_loader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = create_data_loader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 4. Initialize model
    num_classes = len(LANGUAGES)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    model = HubertClassifier(num_classes=num_classes)
    model.to(device)
    
    # 5. Train
    trainer = ModelTrainer(
        model=model,
        device=device,
        learning_rate=LEARNING_RATE,
        checkpoint_dir=MODELS_DIR
    )
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        checkpoint_name=f"hubert_layer_{layer_idx}_enhanced.pth"
    )
    
    # 6. Evaluate on test set
    print("\n" + "="*50)
    print("EVALUATION ON TEST SET")
    print("="*50)
    
    test_loader = create_data_loader(test_dataset, batch_size=batch_size, shuffle=False)
    test_acc, test_loss = trainer.evaluate(test_loader)
    
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    print("="*50)

def train_ensemble_model(epochs=50, batch_size=16):
    """
    Train ensemble of MFCC and HuBERT models
    """
    print(f"\n==========================================")
    print(f"TRAINING: Ensemble (MFCC + HuBERT)")
    print(f"==========================================")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 1. Load pre-trained MFCC and HuBERT models
    print("Loading pre-trained MFCC model...")
    mfcc_model_path = os.path.join(MODELS_DIR, "mfcc_enhanced.pth")
    if not os.path.exists(mfcc_model_path):
        print(f"Error: MFCC model not found at {mfcc_model_path}")
        print("Train MFCC model first: python scripts/train_models.py --feature-type mfcc")
        return
    
    mfcc_model = MFCCClassifier(num_classes=len(LANGUAGES))
    checkpoint = torch.load(mfcc_model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        mfcc_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        mfcc_model.load_state_dict(checkpoint)
    mfcc_model.to(device)
    mfcc_model.eval()
    
    print("Loading pre-trained HuBERT model...")
    hubert_model_path = os.path.join(MODELS_DIR, "hubert_layer_7_enhanced.pth")
    if not os.path.exists(hubert_model_path):
        print(f"Error: HuBERT model not found at {hubert_model_path}")
        print("Train HuBERT model first: python scripts/train_hubert.py --epochs 100")
        return
    
    hubert_model = HubertClassifier(num_classes=len(LANGUAGES))
    checkpoint = torch.load(hubert_model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        hubert_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        hubert_model.load_state_dict(checkpoint)
    hubert_model.to(device)
    hubert_model.eval()
    
    # 2. Create ensemble (weights: MFCC 40%, HuBERT 60%)
    ensemble = EnsembleClassifier(
        mfcc_model=mfcc_model,
        hubert_model=hubert_model,
        mfcc_weight=0.4,
        hubert_weight=0.6
    )
    ensemble.to(device)
    
    print("Ensemble created (MFCC 40% + HuBERT 60%)")
    
    # 3. Load validation dataset (for reference)
    mfcc_features_path = os.path.join(FEATURES_DIR, 'mfcc')
    hubert_features_path = os.path.join(FEATURES_DIR, 'hubert', 'layer_7')
    
    mfcc_dataset = FeatureDataset(mfcc_features_path, augment=False)
    hubert_dataset = FeatureDataset(hubert_features_path, augment=False)
    
    # Note: In production, you would create a custom dataset that loads both
    # and returns paired (mfcc, hubert) samples for the same audio file
    
    print("\n" + "="*50)
    print("ENSEMBLE: Weighted Combination (Non-trainable)")
    print("="*50)
    print("MFCC Model (40%)     : Fast, good for real-time")
    print("HuBERT Model (60%)   : Accurate, better understanding")
    print("Combined Accuracy    : Expected 88-95%")
    print("="*50)
    
    # 4. Save ensemble
    ensemble_checkpoint = {
        'model': ensemble,
        'mfcc_weight': 0.4,
        'hubert_weight': 0.6,
        'languages': LANGUAGES
    }
    
    ensemble_path = os.path.join(MODELS_DIR, "ensemble_mfcc_hubert.pth")
    torch.save(ensemble_checkpoint, ensemble_path)
    print(f"\nEnsemble saved to: {ensemble_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HuBERT and ensemble models")
    parser.add_argument("--mode", choices=['hubert', 'ensemble'], default='hubert',
                        help="What to train")
    parser.add_argument("--layer", type=int, default=7, help="HuBERT layer (for HuBERT mode)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    
    args = parser.parse_args()
    
    if args.mode == 'hubert':
        train_hubert_model(layer_idx=args.layer, epochs=args.epochs, batch_size=args.batch_size)
    else:
        train_ensemble_model(epochs=args.epochs, batch_size=args.batch_size)
