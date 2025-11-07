"""
Script to load and process the IndicAccentDb dataset
"""
import os
import numpy as np
import torch
from config import DATA_DIR, SAMPLE_RATE

def download_dataset():
    """
    Download the IndicAccentDb dataset from Hugging Face
    """
    try:
        from datasets import load_dataset
        print("Downloading IndicAccentDb dataset...")
        
        # Load the dataset
        dataset = load_dataset("DarshanaS/IndicAccentDb")
        
        # Save to local directory
        os.makedirs(DATA_DIR, exist_ok=True)
        dataset.save_to_disk(os.path.join(DATA_DIR, "indic_accent_db"))
        
        print("Dataset downloaded successfully!")
        print(f"Dataset info: {dataset}")
        return dataset
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Using dummy data for demonstration...")
        return None

def preprocess_audio(audio_array, sampling_rate):
    """
    Preprocess audio to 16 kHz mono
    
    Args:
        audio_array (np.array): Audio data
        sampling_rate (int): Original sampling rate
        
    Returns:
        np.array: Preprocessed audio
    """
    # Convert to mono if stereo
    if len(audio_array.shape) > 1:
        audio_array = np.mean(audio_array, axis=1)
    
    # Resample to 16 kHz if needed
    if sampling_rate != SAMPLE_RATE:
        try:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=SAMPLE_RATE)
        except ImportError:
            print("librosa not available for resampling, returning original audio")
    
    return audio_array

def extract_mfcc_features(audio_array, sr=SAMPLE_RATE, n_mfcc=13):
    """
    Extract MFCC features from audio
    
    Args:
        audio_array (np.array): Audio data
        sr (int): Sampling rate
        n_mfcc (int): Number of MFCC coefficients
        
    Returns:
        np.array: MFCC features
    """
    try:
        import librosa
        mfccs = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=n_mfcc)
        # Normalize
        mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
        return mfccs
    except ImportError:
        print("librosa not available, returning dummy MFCC features")
        # Return dummy features
        return np.random.randn(n_mfcc, 100)

def extract_hubert_features(audio_array, sr=SAMPLE_RATE):
    """
    Extract HuBERT features from audio
    
    Args:
        audio_array (np.array): Audio data
        sr (int): Sampling rate
        
    Returns:
        torch.Tensor: HuBERT features
    """
    try:
        from transformers import Wav2Vec2Processor, HubertModel
        from config import HUBERT_MODEL
        import torch
        
        # Initialize processor and model
        processor = Wav2Vec2Processor.from_pretrained(HUBERT_MODEL)
        model = HubertModel.from_pretrained(HUBERT_MODEL)
        model.eval()
        
        # Process audio
        inputs = processor(audio_array, sampling_rate=sr, return_tensors="pt", padding="longest")
        
        # Extract features
        with torch.no_grad():
            outputs = model(**inputs)
            features = outputs.last_hidden_state
            
        return features
    except Exception as e:
        print(f"Error extracting HuBERT features: {e}")
        print("Returning dummy HuBERT features")
        # Return dummy features
        return torch.randn(1, 100, 768)

def process_dataset():
    """
    Process the dataset: load, preprocess, extract features
    """
    print("Processing dataset...")
    
    # Try to load dataset
    try:
        from datasets import load_from_disk
        dataset_path = os.path.join(DATA_DIR, "indic_accent_db")
        if os.path.exists(dataset_path):
            dataset = load_from_disk(dataset_path)
            print("Dataset loaded from disk")
        else:
            dataset = download_dataset()
    except Exception as e:
        print(f"Could not load dataset: {e}")
        dataset = None
    
    if dataset is not None:
        # Process a sample from each split
        for split_name in dataset.keys():
            print(f"\nProcessing {split_name} split...")
            split_data = dataset[split_name]
            
            # Process first 5 samples as example
            for i in range(min(5, len(split_data))):
                try:
                    sample = split_data[i]
                    audio_array = sample["audio"]["array"]
                    sampling_rate = sample["audio"]["sampling_rate"]
                    language = sample["accent_lang"]
                    
                    print(f"  Sample {i+1}: Language={language}, SR={sampling_rate}, Shape={audio_array.shape}")
                    
                    # Preprocess audio
                    processed_audio = preprocess_audio(audio_array, sampling_rate)
                    print(f"    Preprocessed shape: {processed_audio.shape}")
                    
                    # Extract features
                    mfcc_features = extract_mfcc_features(processed_audio)
                    print(f"    MFCC features shape: {mfcc_features.shape}")
                    
                    # For HuBERT, we need to limit the length for memory efficiency
                    if len(processed_audio) > 16000 * 10:  # Limit to 10 seconds
                        processed_audio = processed_audio[:16000 * 10]
                    
                    hubert_features = extract_hubert_features(processed_audio)
                    print(f"    HuBERT features shape: {hubert_features.shape}")
                    
                except Exception as e:
                    print(f"    Error processing sample {i+1}: {e}")
    else:
        print("Using dummy data processing...")
        # Create dummy data
        for i in range(3):
            language = ["hindi", "tamil", "telugu"][i % 3]
            print(f"  Dummy sample {i+1}: Language={language}")
            
            # Dummy audio
            dummy_audio = np.random.randn(16000)  # 1 second of random audio
            mfcc_features = extract_mfcc_features(dummy_audio)
            print(f"    MFCC features shape: {mfcc_features.shape}")
            
            hubert_features = extract_hubert_features(dummy_audio)
            print(f"    HuBERT features shape: {hubert_features.shape}")

if __name__ == "__main__":
    process_dataset()