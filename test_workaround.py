"""
Test script to verify our workaround for torchaudio issues
"""
import sys
import os

def test_imports():
    """
    Test that our modules can be imported without torchaudio issues
    """
    try:
        # Test config import
        from config import LANGUAGES, HUBERT_MODEL
        print("✓ Config imported successfully")
        print(f"  Languages: {', '.join(LANGUAGES)}")
        
        # Test data module
        from data.dataset import IndianAccentDataset
        print("✓ Dataset module imported successfully")
        
        # Test features module
        from features.extractor import FeatureExtractor
        print("✓ Feature extractor module imported successfully")
        
        # Test models module
        from models.classifiers import MFCCClassifier, HubertClassifier
        print("✓ Classifiers module imported successfully")
        
        # Test trainer module
        from models.trainer import ModelTrainer
        print("✓ Trainer module imported successfully")
        
        # Test app module
        from app.cuisine_recommender import CuisineRecommender
        print("✓ Cuisine recommender module imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

def test_dataset_creation():
    """
    Test that we can create dataset objects
    """
    try:
        from data.dataset import IndianAccentDataset
        
        # Create dummy data
        dummy_paths = ["path1.wav", "path2.wav", "path3.wav"]
        dummy_labels = [0, 1, 2]
        
        # Test MFCC dataset
        mfcc_dataset = IndianAccentDataset(dummy_paths, dummy_labels, feature_type='mfcc')
        print(f"✓ MFCC dataset created with {len(mfcc_dataset)} items")
        
        # Test HuBERT dataset
        hubert_dataset = IndianAccentDataset(dummy_paths, dummy_labels, feature_type='hubert')
        print(f"✓ HuBERT dataset created with {len(hubert_dataset)} items")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset creation test failed: {e}")
        return False

def test_model_creation():
    """
    Test that we can create model objects
    """
    try:
        from config import LANGUAGES
        from models.classifiers import MFCCClassifier, HubertClassifier
        
        num_classes = len(LANGUAGES)
        
        # Test MFCC classifier
        mfcc_model = MFCCClassifier(num_classes=num_classes)
        print("✓ MFCC classifier created successfully")
        
        # Test HuBERT classifier
        hubert_model = HubertClassifier(num_classes=num_classes)
        print("✓ HuBERT classifier created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        return False

def test_feature_extraction():
    """
    Test that we can create feature extractor objects
    """
    try:
        from features.extractor import FeatureExtractor
        
        # Test MFCC feature extractor
        mfcc_extractor = FeatureExtractor(feature_type='mfcc')
        print("✓ MFCC feature extractor created successfully")
        
        # Test HuBERT feature extractor
        hubert_extractor = FeatureExtractor(feature_type='hubert')
        print("✓ HuBERT feature extractor created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Feature extractor test failed: {e}")
        return False

def main():
    """
    Main test function
    """
    print("Testing Native Language Identification Workaround")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed.")
        return
    
    print()
    
    # Test dataset creation
    if not test_dataset_creation():
        print("\n❌ Dataset creation tests failed.")
        return
    
    print()
    
    # Test model creation
    if not test_model_creation():
        print("\n❌ Model creation tests failed.")
        return
    
    print()
    
    # Test feature extraction
    if not test_feature_extraction():
        print("\n❌ Feature extraction tests failed.")
        return
    
    print("\n" + "=" * 50)
    print("✅ All workaround tests completed successfully!")
    print("The system is ready to use with the workaround in place.")
    print("\nNote: Audio processing functionality is currently disabled due to torchaudio issues.")
    print("To enable full functionality, resolve the torchaudio installation issues.")

if __name__ == "__main__":
    main()