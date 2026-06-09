"""
Extract MFCC features for all languages from the dataset
Optimized for batch processing and Windows OneDrive handling
"""
import os
import sys
import ctypes
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LANGUAGES, SAMPLE_RATE, MFCC_FEATURES

def is_file_offline(file_path):
    """Check if file is offline in Windows OneDrive without recalling it."""
    if os.name != 'nt':
        return False
    try:
        attrs = ctypes.windll.kernel32.GetFileAttributesW(file_path)
        if attrs == -1:
            return False
        # 0x1000 = FILE_ATTRIBUTE_OFFLINE, 0x400000 = FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS
        return bool(attrs & 0x1000) or bool(attrs & 0x400000)
    except Exception:
        return False

def extract_mfcc_features(audio_path, n_mfcc=13):
    """
    Extract MFCC + Delta + Delta-Delta features
    Returns: (39, T) array where 39 = 13*3 channels, T = time frames
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        
        # Extract MFCC (13, T)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
        
        # Delta (13, T)
        mfcc_delta = librosa.feature.delta(mfcc)
        
        # Delta-Delta (13, T)
        mfcc_delta_delta = librosa.feature.delta(mfcc, order=2)
        
        # Concatenate: (39, T)
        features = np.vstack([mfcc, mfcc_delta, mfcc_delta_delta])
        
        return features
    except Exception as e:
        print(f"    ERROR extracting {audio_path}: {e}")
        return None

def extract_all_languages(data_dir="data_norm", output_dir=None):
    """Extract MFCC features for all languages"""
    
    if output_dir is None:
        output_dir = Path("data") / "features" / "mfcc"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_path = Path(data_dir) / "IndicAccentDB_audio_norm"
    
    print("\n" + "="*70)
    print("MFCC FEATURE EXTRACTION FOR ALL LANGUAGES")
    print("="*70)
    print(f"Input:  {data_path}")
    print(f"Output: {output_dir}")
    print(f"Languages: {', '.join(LANGUAGES)}")
    print("="*70 + "\n")
    
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
            # Check if offline (OneDrive optimization)
            if is_file_offline(str(wav_file)):
                lang_skipped += 1
                total_skipped += 1
                continue
            
            # Output file path
            output_file = output_dir / f"{lang}_{wav_file.stem}.npy"
            
            # Skip if already extracted
            if output_file.exists():
                lang_extracted += 1
                total_extracted += 1
                continue
            
            # Extract features
            features = extract_mfcc_features(str(wav_file))
            if features is not None:
                np.save(str(output_file), features)
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
    
    if total_extracted == 0:
        print("\n⚠️  WARNING: No features were extracted!")
        print("Possible causes:")
        print("  1. Audio files are still offline in OneDrive")
        print("  2. Audio files directory doesn't exist")
        print("  3. File paths have special characters")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract MFCC features for all languages")
    parser.add_argument("--data-dir", default="data_norm", help="Input data directory")
    parser.add_argument("--output-dir", default=None, help="Output directory for features")
    
    args = parser.parse_args()
    
    extract_all_languages(data_dir=args.data_dir, output_dir=args.output_dir)
