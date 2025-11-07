"""
Test script to verify project installation and dependencies
"""
def test_imports():
    """
    Test that all required packages can be imported
    """
    try:
        import torch
        print("✓ PyTorch imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import PyTorch: {e}")
        return False
    
    try:
        import torchaudio
        print("✓ Torchaudio imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Torchaudio: {e}")
        return False
    
    try:
        import transformers
        print("✓ Transformers imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Transformers: {e}")
        return False
    
    try:
        import librosa
        print("✓ Librosa imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Librosa: {e}")
        return False
    
    try:
        import sklearn
        print("✓ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Scikit-learn: {e}")
        return False
    
    try:
        import numpy
        print("✓ NumPy imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import NumPy: {e}")
        return False
    
    try:
        import pandas
        print("✓ Pandas imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Pandas: {e}")
        return False
    
    try:
        # Test our own modules
        from config import LANGUAGES
        print("✓ Project config imported successfully")
        
        from utils.helpers import get_language_label_encoder
        print("✓ Utils module imported successfully")
        
        from data.dataset import IndianAccentDataset
        print("✓ Data module imported successfully")
        
        from features.extractor import FeatureExtractor
        print("✓ Features module imported successfully")
        
        from models.classifiers import MFCCClassifier, HubertClassifier
        print("✓ Models module imported successfully")
        
        print(f"✓ All project modules imported successfully")
        print(f"  Supported languages: {', '.join(LANGUAGES)}")
        
    except ImportError as e:
        print(f"✗ Failed to import project modules: {e}")
        return False
    
    return True

def test_pytorch_cuda():
    """
    Test PyTorch CUDA availability
    """
    import torch
    
    if torch.cuda.is_available():
        print(f"✓ CUDA is available with {torch.cuda.device_count()} device(s)")
        print(f"  Current device: {torch.cuda.get_device_name()}")
    else:
        print("⚠ CUDA is not available, training will use CPU")

def test_transformers_model():
    """
    Test loading a pre-trained transformer model
    """
    try:
        from transformers import Wav2Vec2Processor, HubertModel
        from config import HUBERT_MODEL
        
        print("Testing HuBERT model loading...")
        processor = Wav2Vec2Processor.from_pretrained(HUBERT_MODEL)
        model = HubertModel.from_pretrained(HUBERT_MODEL)
        print("✓ HuBERT model loaded successfully")
        print(f"  Model: {HUBERT_MODEL}")
        print(f"  Hidden size: {model.config.hidden_size}")
        print(f"  Number of layers: {model.config.num_hidden_layers}")
        
        # Clean up
        del processor, model
        
    except Exception as e:
        print(f"✗ Failed to load HuBERT model: {e}")
        return False
    
    return True

def main():
    """
    Main test function
    """
    print("Testing Native Language Identification Project Installation")
    print("=" * 60)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please check your installation.")
        return
    
    print()
    
    # Test PyTorch CUDA
    test_pytorch_cuda()
    print()
    
    # Test transformer model
    if not test_transformers_model():
        print("\n⚠ Model loading test failed, but this might be due to network connectivity.")
    
    print("\n" + "=" * 60)
    print("✅ Installation test completed successfully!")
    print("You're ready to start working on the project.")

if __name__ == "__main__":
    main()