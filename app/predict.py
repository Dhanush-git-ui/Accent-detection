"""
CLI tool for accent prediction from an audio file.
Supports both MFCC (fast) and HuBERT (accurate) modes.

Usage:
    python app/predict.py audio.wav                     # uses MFCC (default, fast)
    python app/predict.py audio.wav --mode hubert       # uses HuBERT (accurate)
    python app/predict.py audio.wav --model path/to/model.pth
"""
import sys
import os
import torch
import numpy as np
import argparse
import librosa
import time

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.classifiers import HubertClassifier, MFCCClassifier
from src.features.extractor import FeatureExtractor
from config import LANGUAGES, MODELS_DIR


def load_model(model_path, feature_type, device):
    """Load and return the appropriate classifier."""
    num_classes = len(LANGUAGES)

    if feature_type == 'mfcc':
        model = MFCCClassifier(num_classes=num_classes)
    else:
        model = HubertClassifier(num_classes=num_classes)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            model = checkpoint

        model.to(device)
        model.eval()

        # Apply dynamic quantization on CPU for speedup
        if device.type == 'cpu':
            print("Applying dynamic quantization on CPU...")
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        print(f"Model loaded from {model_path}")
        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def predict_accent(audio_path, model_path=None, feature_type='mfcc'):
    """Predict accent from an audio file."""
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        return

    # Resolve default model paths
    if model_path is None:
        if feature_type == 'mfcc':
            model_path = os.path.join(MODELS_DIR, "mfcc_enhanced.pth")
            if not os.path.exists(model_path):
                model_path = os.path.join(MODELS_DIR, "mfcc_baseline.pth")
        else:
            model_path = os.path.join(MODELS_DIR, "hubert_enhanced.pth")
            if not os.path.exists(model_path):
                model_path = os.path.join(MODELS_DIR, "hubert_baseline.pth")

    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}.")
        if feature_type == 'mfcc':
            print("Train the MFCC model first: python scripts/train_models.py --feature-type mfcc")
        else:
            print("Train the HuBERT model first: python scripts/train_models.py --feature-type hubert")
        return

    start_time = time.perf_counter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = load_model(model_path, feature_type, device)
    if model is None:
        return

    # Load audio
    print(f"Loading audio: {audio_path}")
    waveform, sr = librosa.load(audio_path, sr=None)

    # Extract features
    print(f"Extracting {feature_type.upper()} features...")
    try:
        extractor = FeatureExtractor(feature_type=feature_type)

        if feature_type == 'mfcc':
            features = extractor.extract_mfcc(waveform, sr)  # (39, time)
            # Pad/truncate to 1000 frames
            target_length = 1000
            if features.shape[1] > target_length:
                features = features[:, :target_length]
            elif features.shape[1] < target_length:
                features = torch.nn.functional.pad(features, (0, target_length - features.shape[1]))
            feature_tensor = features.unsqueeze(0).to(device)  # (1, 39, 1000)
        else:
            # HuBERT: slice to max 5s for fast prediction
            hubert_layers = extractor.extract_hubert_layers(waveform, sr, layers=[7], max_duration=5)
            feature_tensor = hubert_layers[7].unsqueeze(0).to(device)  # (1, seq_len, 768)

    except Exception as e:
        print(f"Failed to extract features: {e}")
        return

    # Predict
    with torch.no_grad():
        outputs = model(feature_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence_scores = probabilities.squeeze().tolist()
        predicted_idx = torch.argmax(probabilities).item()

    if not isinstance(confidence_scores, list):
        confidence_scores = [confidence_scores]

    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000

    # Print results
    print("\n" + "=" * 50)
    print(f"Mode: {'MFCC CNN (Speed)' if feature_type == 'mfcc' else 'HuBERT Transformer (Accuracy)'}")
    
    # Check confidence threshold
    confidence_threshold = 0.45
    max_confidence = max(confidence_scores)
    
    if max_confidence < confidence_threshold:
        print(f"⚠️  WARNING: Low Confidence ({max_confidence*100:.1f}% < {confidence_threshold*100:.0f}%)")
        print("Recommendation: Try a longer audio clip (5+ seconds) for better accuracy")
    else:
        if predicted_idx < len(LANGUAGES):
            predicted_lang = LANGUAGES[predicted_idx]
            confidence = confidence_scores[predicted_idx] * 100
            print(f"Predicted Accent : {predicted_lang.replace('_', ' ').title()}")
            print(f"Confidence       : {confidence:.2f}%")
    
    print("\nAll Language Probabilities:")
    for i, lang in enumerate(LANGUAGES):
        if i < len(confidence_scores):
            bar = "█" * int(confidence_scores[i] * 20)
            print(f"  {lang:<20} {confidence_scores[i]*100:5.1f}% {bar}")
    print(f"\nTotal Latency: {elapsed_ms:.1f} ms")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Indian accent from audio file")
    parser.add_argument("audio_path", type=str, help="Path to audio file (.wav/.mp3/.flac)")
    parser.add_argument("--mode", type=str, default="mfcc", choices=["mfcc", "hubert"],
                        help="Feature type: 'mfcc' (fast) or 'hubert' (accurate). Default: mfcc")
    parser.add_argument("--model", type=str, default=None,
                        help="Optional: path to a specific model checkpoint (.pth)")

    args = parser.parse_args()
    predict_accent(args.audio_path, args.model, args.mode)
