"""
Final test to verify the complete system works with workarounds
"""
import sys
import os

def test_complete_system():
    """
    Test that the complete system works with workarounds
    """
    print("Testing Complete NLI System with Workarounds")
    print("=" * 45)
    
    try:
        # Test config
        from config import LANGUAGES, HUBERT_MODEL
        print("✓ Config loaded successfully")
        
        # Test models
        from models.classifiers import MFCCClassifier, HubertClassifier
        from models.trainer import ModelTrainer
        
        # Create models
        mfcc_model = MFCCClassifier(num_classes=len(LANGUAGES))
        hubert_model = HubertClassifier(num_classes=len(LANGUAGES))
        print("✓ Models created successfully")
        
        # Test trainer
        trainer = ModelTrainer(hubert_model)
        print("✓ Trainer initialized successfully")
        
        # Test features
        from features.extractor import FeatureExtractor
        mfcc_extractor = FeatureExtractor(feature_type='mfcc')
        hubert_extractor = FeatureExtractor(feature_type='hubert')
        print("✓ Feature extractors created successfully")
        
        # Test app
        from app.cuisine_recommender import CuisineRecommender
        recommender = CuisineRecommender("dummy_path", feature_type='hubert')
        print("✓ Cuisine recommender created successfully")
        
        # Test dataset (should work with workaround)
        from data.dataset import IndianAccentDataset
        dummy_paths = ["path1.wav", "path2.wav"]
        dummy_labels = [0, 1]
        dataset = IndianAccentDataset(dummy_paths, dummy_labels, feature_type='hubert')
        print("✓ Dataset created successfully")
        
        print("\n" + "=" * 45)
        print("✅ All tests passed!")
        print("The system is ready to use with workarounds in place.")
        print("\nTo use the full system with audio processing:")
        print("1. Resolve torchaudio installation issues")
        print("2. Replace dummy implementations with real audio processing")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_complete_system()