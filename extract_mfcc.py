"""
Script to extract MFCC features from audio files
"""
import os
import numpy as np
import librosa
from config import SAMPLE_RATE, MFCC_FEATURES

def extract_mfcc_features(audio_path, n_mfcc=MFCC_FEATURES):
    """
    Extract MFCC features from audio file
    
    Args:
        audio_path (str): Path to audio file
        n_mfcc (int): Number of MFCC coefficients to extract
        
    Returns:
        np.array: MFCC features (n_mfcc, time_frames)
    """
    try:
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample if needed
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=n_mfcc)
        
        # Normalize features
        mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
        
        return mfccs
    except Exception as e:
        print(f"Error extracting MFCC features from {audio_path}: {e}")
        return None

def process_audio_directory(input_dir, output_dir, n_mfcc=MFCC_FEATURES):
    """
    Process all audio files in directory and extract MFCC features
    
    Args:
        input_dir (str): Input directory containing audio files
        output_dir (str): Output directory for features
        n_mfcc (int): Number of MFCC coefficients to extract
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    error_count = 0
    
    # Walk through input directory
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                input_path = os.path.join(root, file)
                
                # Create relative path structure in output directory
                rel_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, rel_path)
                os.makedirs(output_subdir, exist_ok=True)
                
                # Output file path
                output_file = os.path.join(output_subdir, os.path.splitext(file)[0] + '.npy')
                
                # Extract MFCC features
                mfcc_features = extract_mfcc_features(input_path, n_mfcc)
                
                if mfcc_features is not None:
                    # Save features
                    np.save(output_file, mfcc_features)
                    print(f"Processed: {input_path} -> {output_file} (shape: {mfcc_features.shape})")
                    processed_count += 1
                else:
                    error_count += 1
    
    print(f"MFCC extraction complete: {processed_count} files processed, {error_count} errors")

def test_mfcc_features(feature_file):
    """
    Test MFCC features file
    
    Args:
        feature_file (str): Path to .npy file containing MFCC features
    """
    try:
        features = np.load(feature_file)
        print(f"Feature shape: {features.shape}")
        print(f"Feature range: [{np.min(features):.3f}, {np.max(features):.3f}]")
        print(f"Feature mean: {np.mean(features):.3f}")
        print(f"Feature std: {np.std(features):.3f}")
        return True
    except Exception as e:
        print(f"Error testing features file {feature_file}: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract MFCC features from audio files")
    parser.add_argument("--input", type=str, help="Input directory containing audio files")
    parser.add_argument("--output", type=str, default="data/features/mfcc", 
                        help="Output directory for features")
    parser.add_argument("--n-mfcc", type=int, default=MFCC_FEATURES, help="Number of MFCC coefficients")
    parser.add_argument("--test", type=str, help="Test a specific .npy file")
    
    args = parser.parse_args()
    
    if args.test:
        print(f"Testing features file: {args.test}")
        test_mfcc_features(args.test)
    elif args.input:
        print(f"Extracting MFCC features from: {args.input}")
        process_audio_directory(args.input, args.output, args.n_mfcc)
    else:
        print("Please specify either --input directory or --test feature file")
        print("Example usage:")
        print("  python extract_mfcc.py --input data_norm/IndicAccentDB_audio_norm --output data/features/mfcc")
        print("  python extract_mfcc.py --test data/features/mfcc/some_file.npy")