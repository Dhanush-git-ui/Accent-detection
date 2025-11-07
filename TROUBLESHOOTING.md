# Troubleshooting Guide for Native Language Identification System

This guide helps resolve common issues encountered when setting up and running the Native Language Identification system.

## Common Issues and Solutions

### 1. torchaudio Installation Issues

**Problem**: 
```
FileNotFoundError: Could not find module '...\torchaudio\lib\libtorchaudio.pyd'
```

**Solution**:
1. Uninstall current torchaudio:
   ```bash
   pip uninstall torchaudio -y
   ```

2. Install a compatible version:
   ```bash
   pip install torchaudio==2.2.0
   ```

3. If that fails, try installing via conda:
   ```bash
   conda install torchaudio -c pytorch
   ```

### 2. NumPy Compatibility Issues

**Problem**:
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.2
```

**Solution**:
Downgrade NumPy to a compatible version:
```bash
pip install numpy==1.24.3
```

### 3. Missing Dependencies

**Problem**: 
Import errors for librosa, transformers, or other packages

**Solution**:
Install all dependencies:
```bash
pip install -r requirements.txt
```

### 4. CUDA Issues

**Problem**: 
CUDA-related errors or warnings

**Solution**:
1. Check CUDA installation:
   ```bash
   nvidia-smi
   ```

2. Install CPU-only version of PyTorch:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

## Workaround Mode

The system includes workaround implementations that allow you to:

1. Test the system architecture without audio processing
2. Run the cuisine recommendation demo
3. Verify model training pipelines with dummy data

### Using Workaround Mode

The workaround mode is automatically enabled when torchaudio is not available. It provides:

- Dummy dataset implementations
- Mock audio processing functions
- Simulated feature extraction
- Working model architectures

### Enabling Full Functionality

To enable full audio processing functionality:

1. **Fix torchaudio installation**:
   ```bash
   pip uninstall torchaudio
   pip install torchaudio==2.2.0
   ```

2. **Verify installation**:
   ```bash
   python -c "import torchaudio; print('torchaudio imported successfully')"
   ```

3. **Replace dummy implementations**:
   - Uncomment actual audio processing code in [data/dataset.py](file:///c%3A/Users/Hp/OneDrive/Desktop/NLP/data/dataset.py)
   - Uncomment actual feature extraction code in [features/extractor.py](file:///c%3A/Users/Hp/OneDrive/Desktop/NLP/features/extractor.py)
   - Uncomment actual audio loading in [utils/helpers.py](file:///c%3A/Users/Hp/OneDrive/Desktop/NLP/utils/helpers.py)

## Platform-Specific Issues

### Windows

1. **Visual Studio Redistributables**:
   Ensure you have the latest Visual Studio Redistributables installed.

2. **Path Issues**:
   Make sure Python and pip are in your system PATH.

### macOS

1. **Xcode Command Line Tools**:
   ```bash
   xcode-select --install
   ```

2. **Homebrew Dependencies**:
   ```bash
   brew install libsndfile
   ```

### Linux

1. **System Dependencies**:
   ```bash
   sudo apt-get install libsndfile1-dev
   ```

## Testing Your Installation

Run the installation test:
```bash
python test_installation.py
```

Run the workaround test:
```bash
python test_workaround.py
```

Run the final system test:
```bash
python final_test.py
```

## Running the System

### With Workarounds (No Audio Processing):
```bash
python main.py --demo
```

### With Full Functionality (After Fixing Dependencies):
```bash
# Train model
python train_nli.py --feature-type hubert

# Run demo
python main.py --demo

# Run complete pipeline
python main.py --all
```

## Dataset Preparation

1. Download the IndicAccentDb dataset from:
   https://huggingface.co/datasets/DarshanaS/IndicAccentDb

2. Extract to the [data/](file:///c%3A/Users/Hp/OneDrive/Desktop/NLP/data) directory

3. Update the dataset loading code in [data/dataset.py](file:///c%3A/Users/Hp/OneDrive/Desktop/NLP/data/dataset.py) to point to your dataset location

## Need Help?

If you continue to experience issues:

1. Check the error messages carefully
2. Verify all dependencies are installed
3. Ensure compatible versions of packages
4. Consult the PyTorch and torchaudio documentation
5. Check system requirements for audio processing libraries

For further assistance, please provide:
- Operating system and version
- Python version
- Error messages
- Steps to reproduce the issue