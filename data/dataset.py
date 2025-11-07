import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from config import HUBERT_MODEL, MAX_AUDIO_LENGTH, SAMPLE_RATE

class IndianAccentDataset(Dataset):
    """
    Dataset class for Indian Accent Database
    """
    def __init__(self, audio_paths, labels, feature_type='mfcc'):
        """
        Initialize dataset
        
        Args:
            audio_paths (list): List of audio file paths
            labels (list): List of corresponding language labels
            feature_type (str): Type of features to extract ('mfcc' or 'hubert')
        """
        self.audio_paths = audio_paths
        self.labels = labels
        self.feature_type = feature_type
        
        # Note: We're simplifying this for now due to torchaudio issues
        # In a real implementation, you would load actual audio files here
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        # This is a placeholder implementation
        # In a real implementation, you would load and process actual audio files
        try:
            # Return dummy data for now
            if self.feature_type == 'mfcc':
                # Return dummy MFCC features (batch, channels, time)
                features = torch.randn(1, 13, 1000)
            else:  # hubert
                # Return dummy HuBERT features (batch, channels, time)
                features = torch.randn(1, 768, 1000)
                
            label = self.labels[idx]
            return features, label
            
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            # Return zero tensor and dummy label in case of error
            if self.feature_type == 'mfcc':
                return torch.zeros(1, 13, 1000), 0
            else:
                return torch.zeros(1, 768, 1000), 0

def collate_fn(batch):
    """
    Collate function for DataLoader
    """
    features, labels = zip(*batch)
    features = torch.stack(features)
    labels = torch.tensor(labels)
    return features, labels

def create_data_loader(dataset, batch_size=32, shuffle=True):
    """
    Create DataLoader for dataset
    
    Args:
        dataset (Dataset): Dataset object
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle data
        
    Returns:
        DataLoader: PyTorch DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )