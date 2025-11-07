"""
Setup script for Native Language Identification project
"""
import os
import subprocess
import sys

def install_dependencies():
    """
    Install project dependencies
    """
    print("Installing project dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False
    return True

def create_model_directories():
    """
    Create directories for saving models
    """
    directories = ['models', 'data', 'results']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def download_pretrained_models():
    """
    Download pretrained models if needed
    """
    print("Downloading pretrained models...")
    # This would download any additional pretrained models needed
    # For now, HuBERT will be automatically downloaded by the transformers library
    print("Pretrained models ready!")

def setup_project():
    """
    Set up the complete project environment
    """
    print("Setting up Native Language Identification project...")
    print("=" * 50)
    
    # Install dependencies
    if not install_dependencies():
        print("Failed to install dependencies. Please check your internet connection and try again.")
        return False
    
    # Create directories
    create_model_directories()
    
    # Download pretrained models
    download_pretrained_models()
    
    print("=" * 50)
    print("Project setup completed successfully!")
    print("\nNext steps:")
    print("1. Download the IndicAccentDb dataset from https://huggingface.co/datasets/DarshanaS/IndicAccentDb")
    print("2. Run training: python train_nli.py")
    print("3. For complete pipeline: python main.py --all")
    
    return True

if __name__ == "__main__":
    setup_project()