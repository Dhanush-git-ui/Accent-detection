"""
System test to verify all components work together
"""
import os
import sys

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from config import LANGUAGES, CUISINE_MAPPING
        print("✓ Config imported successfully")
    except Exception as e:
        print(f"✗ Config import failed: {e}")
        return False
    
    try:
        import flask
        print("✓ Flask imported successfully")
    except Exception as e:
        print(f"✗ Flask import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ NumPy imported successfully")
    except Exception as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        from datasets import load_dataset
        print("✓ Datasets library imported successfully")
    except Exception as e:
        print(f"⚠ Datasets library import failed (optional): {e}")
    
    try:
        import librosa
        print("✓ Librosa imported successfully")
    except Exception as e:
        print(f"⚠ Librosa import failed (optional): {e}")
    
    try:
        import torch
        print("✓ PyTorch imported successfully")
    except Exception as e:
        print(f"⚠ PyTorch import failed (optional): {e}")
    
    return True

def test_project_structure():
    """Test that project structure is correct"""
    print("\nTesting project structure...")
    
    required_files = [
        'config.py',
        'requirements.txt',
        'app.py',
        'train_nli.py',
        'dataset_loader.py'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")
            return False
    
    required_dirs = [
        'data',
        'models',
        'features',
        'utils',
        'app',
        'notebooks',
        'templates',
        'uploads',
        'results'
    ]
    
    for directory in required_dirs:
        if os.path.exists(directory) and os.path.isdir(directory):
            print(f"✓ {directory}/ directory exists")
        else:
            print(f"✗ {directory}/ directory missing")
            return False
    
    return True

def test_templates():
    """Test that HTML templates exist"""
    print("\nTesting HTML templates...")
    
    required_templates = [
        'templates/index.html',
        'templates/demo.html'
    ]
    
    for template in required_templates:
        if os.path.exists(template):
            print(f"✓ {template} exists")
        else:
            print(f"✗ {template} missing")
            return False
    
    return True

def test_config():
    """Test that configuration is correct"""
    print("\nTesting configuration...")
    
    try:
        from config import LANGUAGES, CUISINE_MAPPING, HUBERT_MODEL
        
        print(f"✓ Supported languages: {len(LANGUAGES)}")
        print(f"✓ Cuisine mappings: {len(CUISINE_MAPPING)} languages")
        print(f"✓ HuBERT model: {HUBERT_MODEL}")
        
        # Check that all languages have cuisine mappings
        missing_cuisines = []
        for lang in LANGUAGES:
            if lang not in CUISINE_MAPPING:
                missing_cuisines.append(lang)
        
        if missing_cuisines:
            print(f"⚠ Missing cuisine mappings for: {missing_cuisines}")
        else:
            print("✓ All languages have cuisine mappings")
        
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def main():
    """Main test function"""
    print("Native Language Identification System Test")
    print("=" * 45)
    
    all_passed = True
    
    # Run all tests
    all_passed &= test_imports()
    all_passed &= test_project_structure()
    all_passed &= test_templates()
    all_passed &= test_config()
    
    print("\n" + "=" * 45)
    if all_passed:
        print("✅ All tests passed! The system is ready to use.")
        print("\nTo run the web application:")
        print("  python app.py")
        print("\nTo train models:")
        print("  python train_nli.py --feature-type mfcc")
        print("  python train_nli.py --feature-type hubert")
        print("\nTo process the dataset:")
        print("  python dataset_loader.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()