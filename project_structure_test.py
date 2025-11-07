"""
Test script to verify project structure
"""
import os
import sys

def test_project_structure():
    """
    Test that all required files and directories exist
    """
    # Check main directories
    required_dirs = ['data', 'models', 'features', 'utils', 'app', 'notebooks', 'results']
    for directory in required_dirs:
        if not os.path.exists(directory):
            print(f"✗ Missing directory: {directory}")
            return False
        print(f"✓ Directory exists: {directory}")
    
    # Check main files
    required_files = [
        'config.py',
        'requirements.txt',
        'README.md',
        'train_nli.py',
        'main.py',
        'setup.py'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"✗ Missing file: {file}")
            return False
        print(f"✓ File exists: {file}")
    
    # Check module files
    module_files = {
        'data': ['__init__.py', 'dataset.py'],
        'models': ['__init__.py', 'classifiers.py', 'trainer.py'],
        'features': ['__init__.py', 'extractor.py'],
        'utils': ['__init__.py', 'helpers.py'],
        'app': ['__init__.py', 'cuisine_recommender.py'],
        'results': ['analysis.py']
    }
    
    for directory, files in module_files.items():
        for file in files:
            filepath = os.path.join(directory, file)
            if not os.path.exists(filepath):
                print(f"✗ Missing file: {filepath}")
                return False
            print(f"✓ File exists: {filepath}")
    
    # Check notebook files
    notebook_files = [
        'notebooks/hubert_layer_analysis.ipynb',
        'notebooks/word_vs_sentence_analysis.ipynb',
        'notebooks/cross_age_generalization.ipynb'
    ]
    
    for file in notebook_files:
        if not os.path.exists(file):
            print(f"✗ Missing file: {file}")
            return False
        print(f"✓ File exists: {file}")
    
    return True

def test_config_import():
    """
    Test config import
    """
    try:
        from config import LANGUAGES, HUBERT_MODEL
        print(f"✓ Config imported successfully")
        print(f"  Languages: {LANGUAGES}")
        print(f"  HuBERT Model: {HUBERT_MODEL}")
        return True
    except Exception as e:
        print(f"✗ Config import failed: {e}")
        return False

def main():
    """
    Main test function
    """
    print("Testing Native Language Identification Project Structure")
    print("=" * 60)
    
    # Test project structure
    if not test_project_structure():
        print("\n❌ Project structure test failed.")
        return
    
    print()
    
    # Test config import
    if not test_config_import():
        print("\n❌ Config import test failed.")
        return
    
    print("\n" + "=" * 60)
    print("✅ Project structure test completed successfully!")
    print("All required files and directories are in place.")

if __name__ == "__main__":
    main()