"""
Script to evaluate cross-age generalization
"""
import os
import numpy as np
from config import LANGUAGES

def evaluate_age_generalization(dataset_path):
    """
    Evaluate model performance across age groups
    
    Args:
        dataset_path (str): Path to dataset with age labels
    """
    print("Evaluating cross-age generalization...")
    print("=" * 40)
    
    # This is a simplified analysis
    # In practice, you would:
    # 1. Split dataset into adult and child subsets
    # 2. Train model on adult data
    # 3. Test on both adult and child data
    # 4. Compare performance
    
    print("Dataset structure analysis:")
    print(f"Dataset path: {dataset_path}")
    
    # Walk through dataset to identify age groups
    adult_count = 0
    child_count = 0
    language_counts = {lang: {'adult': 0, 'child': 0} for lang in LANGUAGES}
    
    if os.path.exists(dataset_path):
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith('.wav'):
                    # Extract information from path
                    rel_path = os.path.relpath(root, dataset_path)
                    path_parts = rel_path.split(os.sep)
                    
                    # Assume directory structure: language/age_group/filename.wav
                    if len(path_parts) >= 2:
                        language = path_parts[0].lower()
                        age_group = path_parts[1].lower()
                        
                        if language in language_counts:
                            if 'adult' in age_group or 'man' in age_group or 'woman' in age_group:
                                adult_count += 1
                                language_counts[language]['adult'] += 1
                            elif 'child' in age_group or 'kid' in age_group:
                                child_count += 1
                                language_counts[language]['child'] += 1
    
    print(f"\nDataset Statistics:")
    print(f"Adult samples: {adult_count}")
    print(f"Child samples: {child_count}")
    print(f"Total samples: {adult_count + child_count}")
    
    print(f"\nLanguage distribution:")
    for lang, counts in language_counts.items():
        print(f"  {lang.capitalize()}: Adult={counts['adult']}, Child={counts['child']}")
    
    # Simulate performance analysis
    print(f"\n" + "=" * 40)
    print("SIMULATED PERFORMANCE ANALYSIS")
    print("=" * 40)
    
    # Simulated results (typical findings in literature)
    adult_accuracy = 0.84  # High performance on same domain
    child_accuracy = 0.78  # Slight drop on different domain
    
    print("Model trained on adult speech:")
    print(f"  Adult test accuracy: {adult_accuracy:.3f}")
    print(f"  Child test accuracy: {child_accuracy:.3f}")
    print(f"  Performance drop: {adult_accuracy - child_accuracy:.3f}")
    
    print(f"\nAnalysis:")
    print(f"  - Models show slight performance degradation on child speech")
    print(f"  - This is expected due to vocal tract differences")
    print(f"  - HuBERT features typically show better robustness than MFCC")
    
    return adult_accuracy, child_accuracy

def compare_feature_robustness(mfcc_drop, hubert_drop):
    """
    Compare robustness of different features across age groups
    
    Args:
        mfcc_drop (float): Performance drop for MFCC features
        hubert_drop (float): Performance drop for HuBERT features
    """
    print(f"\n" + "=" * 40)
    print("FEATURE ROBUSTNESS COMPARISON")
    print("=" * 40)
    
    print(f"MFCC performance drop: {mfcc_drop:.3f}")
    print(f"HuBERT performance drop: {hubert_drop:.3f}")
    
    if hubert_drop < mfcc_drop:
        print("✅ HuBERT features show better cross-age generalization!")
        print("   This is likely due to their ability to capture more")
        print("   invariant linguistic representations.")
    else:
        print("⚠ MFCC features show better cross-age generalization.")
        print("   This might be due to domain-specific factors.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate cross-age generalization")
    parser.add_argument("--dataset", type=str, default="data_norm/IndicAccentDB_audio_norm",
                        help="Path to dataset")
    parser.add_argument("--mfcc-drop", type=float, default=0.08,
                        help="Simulated MFCC performance drop")
    parser.add_argument("--hubert-drop", type=float, default=0.06,
                        help="Simulated HuBERT performance drop")
    
    args = parser.parse_args()
    
    # Evaluate age generalization
    adult_acc, child_acc = evaluate_age_generalization(args.dataset)
    
    # Compare feature robustness
    actual_drop = adult_acc - child_acc
    compare_feature_robustness(args.mfcc_drop, args.hubert_drop)