"""
Script to convert all audio files to 16 kHz mono format
"""
import os
import librosa
import soundfile as sf
from config import SAMPLE_RATE
import numpy as np

def convert_audio_file(input_path, output_path, target_sr=SAMPLE_RATE):
    """
    Convert audio file to target sample rate and mono channel
    
    Args:
        input_path (str): Path to input audio file
        output_path (str): Path to output audio file
        target_sr (int): Target sample rate
    """
    try:
        # Load audio file
        audio, sr = librosa.load(input_path, sr=None)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample if needed
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        # Save converted audio
        sf.write(output_path, audio, target_sr)
        print(f"Converted: {input_path} -> {output_path}")
        return True
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        return False

def process_dataset_audio(dataset_path, output_path):
    """
    Process all audio files in dataset
    
    Args:
        dataset_path (str): Path to dataset directory
        output_path (str): Path to output directory
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Process audio files
    processed_count = 0
    error_count = 0
    
    # Walk through dataset directory
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                input_path = os.path.join(root, file)
                
                # Create relative path structure in output directory
                rel_path = os.path.relpath(root, dataset_path)
                output_dir = os.path.join(output_path, rel_path)
                os.makedirs(output_dir, exist_ok=True)
                
                # Output file path
                output_file = os.path.join(output_dir, os.path.splitext(file)[0] + '.wav')
                
                # Convert audio file
                if convert_audio_file(input_path, output_file):
                    processed_count += 1
                else:
                    error_count += 1
    
    print(f"Processing complete: {processed_count} files processed, {error_count} errors")

def create_dummy_data(output_path, num_files=100):
    """
    Create dummy audio files for testing
    
    Args:
        output_path (str): Path to output directory
        num_files (int): Number of dummy files to create
    """
    os.makedirs(output_path, exist_ok=True)
    
    languages = ['hindi', 'tamil', 'telugu', 'malayalam', 'kannada', 'punjabi', 'bengali', 'gujarati']
    
    for i in range(num_files):
        # Create dummy audio (1 second of random noise)
        duration = 1.0  # seconds
        sample_rate = SAMPLE_RATE
        samples = int(duration * sample_rate)
        audio = np.random.randn(samples)
        
        # Create language-specific subdirectory
        language = languages[i % len(languages)]
        lang_dir = os.path.join(output_path, language)
        os.makedirs(lang_dir, exist_ok=True)
        
        # Save dummy audio file
        file_path = os.path.join(lang_dir, f"dummy_{i:03d}.wav")
        sf.write(file_path, audio, sample_rate)
        print(f"Created dummy file: {file_path}")
    
    print(f"Created {num_files} dummy audio files")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert audio files to 16 kHz mono")
    parser.add_argument("--input", type=str, help="Input dataset path")
    parser.add_argument("--output", type=str, default="data_norm/IndicAccentDB_audio_norm", 
                        help="Output directory path")
    parser.add_argument("--dummy", action="store_true", help="Create dummy data instead")
    parser.add_argument("--num-dummy", type=int, default=100, help="Number of dummy files to create")
    
    args = parser.parse_args()
    
    if args.dummy:
        print("Creating dummy data...")
        create_dummy_data(args.output, args.num_dummy)
    elif args.input:
        print(f"Processing dataset: {args.input}")
        process_dataset_audio(args.input, args.output)
    else:
        print("Please specify either --input dataset path or --dummy to create dummy data")
        print("Example usage:")
        print("  python resample_all.py --input data/IndicAccentDB --output data_norm/IndicAccentDB_audio_norm")
        print("  python resample_all.py --dummy --num-dummy 200")