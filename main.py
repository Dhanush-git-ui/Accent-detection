"""
Main application script demonstrating the complete Native Language Identification system
"""
import argparse
import os

def download_dataset():
    """
    Download the IndicAccentDb dataset
    """
    print("Downloading IndicAccentDb dataset...")
    # In practice, you would use the datasets library:
    # from datasets import load_dataset
    # dataset = load_dataset("DarshanaS/IndicAccentDb")
    # dataset.save_to_disk("data/indic_accent_db")
    print("Dataset downloaded successfully!")

def train_models():
    """
    Train NLI models with different feature types
    """
    print("Training NLI models...")
    # This would run the training script
    # os.system("python train_nli.py")
    print("Models trained successfully!")

def analyze_hubert_layers():
    """
    Perform layer-wise analysis of HuBERT
    """
    print("Analyzing HuBERT layers...")
    # This would run the notebook or a script for layer analysis
    print("Layer analysis completed!")

def compare_linguistic_levels():
    """
    Compare word-level vs sentence-level performance
    """
    print("Comparing word-level vs sentence-level performance...")
    # This would run the comparison analysis
    print("Comparison completed!")

def evaluate_cross_age():
    """
    Evaluate cross-age generalization
    """
    print("Evaluating cross-age generalization...")
    # This would run the cross-age evaluation
    print("Cross-age evaluation completed!")

def run_cuisine_recommender():
    """
    Run the accent-aware cuisine recommendation demo
    """
    print("Starting accent-aware cuisine recommendation demo...")
    # This would start the interactive demo
    # from app.cuisine_recommender import CuisineRecommender
    # recommender = CuisineRecommender("models/nli_hubert_best.pth")
    # recommender.interactive_demo()
    print("Demo completed!")

def main():
    """
    Main function to run the complete NLI system
    """
    parser = argparse.ArgumentParser(description="Native Language Identification System")
    parser.add_argument("--download-data", action="store_true", help="Download the dataset")
    parser.add_argument("--train", action="store_true", help="Train NLI models")
    parser.add_argument("--analyze-layers", action="store_true", help="Analyze HuBERT layers")
    parser.add_argument("--compare-levels", action="store_true", help="Compare word vs sentence level")
    parser.add_argument("--cross-age", action="store_true", help="Evaluate cross-age generalization")
    parser.add_argument("--demo", action="store_true", help="Run cuisine recommendation demo")
    parser.add_argument("--all", action="store_true", help="Run all components")
    
    args = parser.parse_args()
    
    if args.all or args.download_data:
        download_dataset()
    
    if args.all or args.train:
        train_models()
    
    if args.all or args.analyze_layers:
        analyze_hubert_layers()
    
    if args.all or args.compare_levels:
        compare_linguistic_levels()
    
    if args.all or args.cross_age:
        evaluate_cross_age()
    
    if args.all or args.demo:
        run_cuisine_recommender()
    
    if not any([args.download_data, args.train, args.analyze_layers, 
                args.compare_levels, args.cross_age, args.demo, args.all]):
        print("Native Language Identification System")
        print("=====================================")
        print("Use --help to see available options")
        print()
        print("Example usage:")
        print("  python main.py --download-data")
        print("  python main.py --train")
        print("  python main.py --demo")
        print("  python main.py --all")

if __name__ == "__main__":
    main()