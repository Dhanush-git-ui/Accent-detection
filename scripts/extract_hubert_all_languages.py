"""
HuBERT Feature Extractor for all languages
Extracts embeddings from specified layer of HuBERT model
"""
import os
import sys
import torch
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LANGUAGES, SAMPLE_RATE
from transformers import Wav2Vec2FeatureExtractor, HubertModel

def extract_hubert_embedding(audio_path, model, processor, layer_idx=7, device='cpu'):
    """
    Extract HuBERT embedding from a single audio file
    Returns: (T, 768) array where T = time frames, 768 = embedding dimension
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        
        # Process for HuBERT
        inputs = processor(y, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        input_values = inputs["input_values"].to(device)
        
        # Extract embeddings with attention to specific layer
        with torch.no_grad():
            outputs = model(
                input_values,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Get hidden state from specified layer
        hidden_state = outputs.hidden_states[layer_idx]  # (batch, time, 768)
        embedding = hidden_state[0].cpu().numpy()  # (time, 768)
        
        return embedding
    except Exception as e:
        print(f"    ERROR extracting {audio_path}: {e}")
        return None

def extract_all_languages_hubert(
    data_dir="data_norm",
    output_dir=None,
    layer_idx=7,
    batch_process=False
):
    """Extract HuBERT features for all languages"""
    
    if output_dir is None:
        output_dir = Path("data") / "features" / "hubert" / f"layer_{layer_idx}"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_path = Path(data_dir) / "IndicAccentDB_audio_norm"
    
    print("\n" + "="*70)
    print(f"HuBERT FEATURE EXTRACTION - Layer {layer_idx}")
    print("="*70)
    print(f"Input:  {data_path}")
    print(f"Output: {output_dir}")
    print(f"Languages: {', '.join(LANGUAGES)}")
    print("="*70 + "\n")
    
    # Initialize HuBERT
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("Loading HuBERT model (first run may take a few minutes)...")
    
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)
    model.eval()
    
    # Quantize on CPU for faster inference
    if device == torch.device('cpu'):
        print("Applying quantization to HuBERT on CPU...")
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    
    print()
    
    total_extracted = 0
    total_skipped = 0
    lang_counts = {}
    
    for lang in LANGUAGES:
        lang_dir = data_path / lang
        
        if not lang_dir.exists():
            print(f"[{lang}] Directory not found: {lang_dir}")
            continue
        
        # Get all WAV files
        wav_files = sorted(lang_dir.glob("*.wav"))
        print(f"[{lang}] Found {len(wav_files)} files")
        
        if len(wav_files) == 0:
            print(f"[{lang}] SKIP: No WAV files found")
            continue
        
        lang_extracted = 0
        lang_skipped = 0
        
        for wav_file in tqdm(wav_files, desc=f"  {lang}"):
            # Output file path
            output_file = output_dir / f"{lang}_{wav_file.stem}.npy"
            
            # Skip if already extracted
            if output_file.exists():
                lang_extracted += 1
                total_extracted += 1
                continue
            
            # Extract features
            embedding = extract_hubert_embedding(
                str(wav_file),
                model,
                processor,
                layer_idx=layer_idx,
                device=device
            )
            
            if embedding is not None:
                np.save(str(output_file), embedding)
                lang_extracted += 1
                total_extracted += 1
            else:
                lang_skipped += 1
                total_skipped += 1
        
        lang_counts[lang] = lang_extracted
        print(f"[{lang}] Extracted: {lang_extracted}, Skipped: {lang_skipped}\n")
    
    print("\n" + "="*70)
    print("EXTRACTION SUMMARY")
    print("="*70)
    for lang in LANGUAGES:
        print(f"  {lang}: {lang_counts.get(lang, 0)} features")
    print(f"  TOTAL: {total_extracted} features extracted, {total_skipped} skipped")
    print("="*70)
    
    # Verify output
    output_files = list(output_dir.glob("*.npy"))
    print(f"\nOutput directory contains {len(output_files)} .npy files")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract HuBERT features for all languages")
    parser.add_argument("--layer", type=int, default=7, help="HuBERT layer to extract (0-11)")
    parser.add_argument("--data-dir", default="data_norm", help="Input data directory")
    parser.add_argument("--output-dir", default=None, help="Output directory for features")
    
    args = parser.parse_args()
    
    extract_all_languages_hubert(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        layer_idx=args.layer
    )
