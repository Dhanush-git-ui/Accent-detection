import torch
import numpy as np
import librosa
from config import HUBERT_MODEL, SAMPLE_RATE
from transformers import Wav2Vec2FeatureExtractor, HubertModel

class FeatureExtractor:
    """
    Feature extractor for audio signals
    """
    def __init__(self, feature_type='mfcc'):
        """
        Initialize feature extractor
        
        Args:
            feature_type (str): Type of features to extract ('mfcc' or 'hubert')
        """
        self.feature_type = feature_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if feature_type == 'hubert':
            # Initialize HuBERT model and processor
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(HUBERT_MODEL)
            self.model = HubertModel.from_pretrained(HUBERT_MODEL).to(self.device)
            # Apply dynamic quantization on CPU to speed it up!
            if self.device == torch.device('cpu'):
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
    
    def extract_mfcc(self, waveform, sample_rate):
        """
        Extract MFCC features from waveform.
        Outputs 39-channel features: 13 MFCC + 13 delta + 13 delta-delta,
        matching the pre-extracted training data.
        
        Args:
            waveform (tensor): Audio waveform
            sample_rate (int): Sample rate
            
        Returns:
            tensor: MFCC features of shape (39, time)
        """
        # Convert tensor to numpy if needed
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy().flatten()
        
        # Ensure waveform is 1D
        if waveform.ndim > 1:
            waveform = waveform.flatten()
        
        # Extract 13 MFCC coefficients
        mfccs = librosa.feature.mfcc(
            y=waveform,
            sr=sample_rate,
            n_mfcc=13,
            n_fft=2048,
            hop_length=512
        )
        
        # Compute delta and delta-delta for richer features (total: 39 channels)
        delta = librosa.feature.delta(mfccs)
        delta2 = librosa.feature.delta(mfccs, order=2)
        mfccs_39 = np.concatenate([mfccs, delta, delta2], axis=0)  # (39, time)
        
        return torch.FloatTensor(mfccs_39)
    
    def extract_hubert(self, waveform, sample_rate, max_duration=None):
        """
        Extract HuBERT features from waveform
        
        Args:
            waveform (tensor): Audio waveform
            sample_rate (int): Sample rate
            max_duration (float, optional): Maximum duration to slice audio in seconds
            
        Returns:
            tensor: HuBERT features
        """
        # Convert tensor to numpy if needed
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy().flatten()
        
        # Ensure correct sample rate for HuBERT
        if sample_rate != 16000:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # Limit duration if requested (e.g. for fast inference)
        if max_duration is not None:
            max_samples = int(max_duration * sample_rate)
            if len(waveform) > max_samples:
                waveform = waveform[:max_samples]
        
        # Process audio for HuBERT
        input_values = self.processor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding="longest"
        ).input_values.to(self.device)
        
        # Extract features
        with torch.no_grad():
            hidden_states = self.model(input_values).last_hidden_state
        
        # Return features (sequence_length, hidden_size) on CPU
        return hidden_states.squeeze(0).cpu()
        
    def extract_hubert_layers(self, waveform, sample_rate, layers=None, max_duration=None):
        """
        Extract HuBERT features from multiple intermediate layers
        
        Args:
            waveform (tensor/numpy): Audio waveform
            sample_rate (int): Sample rate
            layers (list): List of layer indices to extract
            max_duration (float, optional): Maximum duration to slice audio in seconds
            
        Returns:
            dict: Mapping of layer index to feature tensor
        """
        # Convert tensor to numpy if needed
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.numpy().flatten()
            
        # Ensure correct sample rate for HuBERT
        if sample_rate != 16000:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
            
        # Limit duration if requested
        if max_duration is not None:
            max_samples = int(max_duration * sample_rate)
            if len(waveform) > max_samples:
                waveform = waveform[:max_samples]
            
        # Process audio for HuBERT
        input_values = self.processor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding="longest"
        ).input_values.to(self.device)
        
        # Extract features with hidden states
        with torch.no_grad():
            outputs = self.model(input_values, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # Tuple of tensors for each layer
            
        if layers is None:
            layers = range(len(hidden_states))
            
        layer_features = {}
        for layer_idx in layers:
            features = hidden_states[layer_idx].squeeze(0).cpu()
            layer_features[layer_idx] = features
            
        return layer_features
    
    def extract_features(self, waveform, sample_rate, max_duration=None):
        """
        Extract features from waveform
        
        Args:
            waveform (tensor): Audio waveform
            sample_rate (int): Sample rate
            max_duration (float, optional): Maximum duration in seconds
            
        Returns:
            tensor: Extracted features
        """
        if self.feature_type == 'mfcc':
            return self.extract_mfcc(waveform, sample_rate)
        elif self.feature_type == 'hubert':
            return self.extract_hubert(waveform, sample_rate, max_duration=max_duration)
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")