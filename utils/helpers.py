import os
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

# Try to import audio libraries with fallback
TORCHAUDIO_AVAILABLE = False
LIBROSA_AVAILABLE = False

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    print("Warning: torchaudio not available. Audio loading functions will not work.")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    print("Warning: librosa not available. Some audio processing functions will not work.")

def load_audio(file_path, target_sr=16000):
    """
    Load audio file and resample to target sample rate
    
    Args:
        file_path (str): Path to audio file
        target_sr (int): Target sample rate
    
    Returns:
        tuple: (audio_tensor, sample_rate)
    """
    if not TORCHAUDIO_AVAILABLE:
        print("Error: torchaudio not available for audio loading")
        return None, None
    
    try:
        audio, sr = torchaudio.load(file_path)
        
        # Resample if needed
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            audio = resampler(audio)
            
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
            
        return audio.squeeze(), target_sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def extract_mfcc_features(audio, sr, n_mfcc=13, max_length=10):
    """
    Extract MFCC features from audio
    
    Args:
        audio (tensor): Audio tensor
        sr (int): Sample rate
        n_mfcc (int): Number of MFCC coefficients
        max_length (int): Maximum length in seconds
    
    Returns:
        np.array: MFCC features
    """
    if not LIBROSA_AVAILABLE:
        print("Error: librosa not available for MFCC extraction")
        return np.random.randn(n_mfcc, max_length * sr // 512)  # Return dummy data
    
    # Convert tensor to numpy if needed
    if isinstance(audio, torch.Tensor):
        audio = audio.numpy()
    
    # Pad or truncate to max_length
    target_length = sr * max_length
    if len(audio) > target_length:
        audio = audio[:target_length]
    elif len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    # Normalize
    mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
    
    return mfccs

def get_language_label_encoder():
    """
    Get label encoder for languages
    
    Returns:
        LabelEncoder: Fitted label encoder
    """
    from config import LANGUAGES
    le = LabelEncoder()
    le.fit(LANGUAGES)
    return le

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

def load_model(model, optimizer, filepath):
    """
    Load model checkpoint
    
    Args:
        model (nn.Module): Model to load into
        optimizer (optim.Optimizer): Optimizer
        filepath (str): Path to checkpoint
    
    Returns:
        tuple: (epoch, loss)
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Model loaded from {filepath}")
    return epoch, loss