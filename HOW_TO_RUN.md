# How to Run the Native Language Identification Project

This guide explains how to set up and run the complete Native Language Identification system for Indian English speakers.

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation Steps

### 1. Install Required Dependencies

```bash
# Navigate to the project directory
cd NLP

# Install all required packages
pip install -r requirements.txt

# If you encounter issues with torchaudio, install it separately
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 2. Install Flask for Web Application

```bash
pip install flask
```

### 3. Create Necessary Directories

```bash
mkdir uploads
mkdir data
```

## Dataset Setup

### Option 1: Download the IndicAccentDb Dataset

```bash
# Run the dataset loader script
python dataset_loader.py
```

This will automatically download the dataset from Hugging Face:
https://huggingface.co/datasets/DarshanaS/IndicAccentDb

### Option 2: Manual Download

1. Visit the dataset page: https://huggingface.co/datasets/DarshanaS/IndicAccentDb
2. Download the dataset files
3. Extract to the `data/indic_accent_db` directory

## Audio Preprocessing

The system automatically preprocesses audio to ensure consistent quality:
- Converts to mono channel
- Resamples to 16 kHz sampling rate
- Normalizes audio levels

## Feature Extraction

The system implements two types of feature extraction:

### 1. MFCC Features (Baseline)
- Extracts 13 Mel-Frequency Cepstral Coefficients
- Traditional approach for speech analysis

### 2. HuBERT Embeddings (Advanced)
- Uses pre-trained HuBERT model for self-supervised representation
- Captures deeper linguistic information

## Model Training

To train the models:

```bash
# Train with MFCC features
python train_nli.py --feature-type mfcc

# Train with HuBERT features
python train_nli.py --feature-type hubert

# Compare both approaches
python train_nli.py --compare
```

## Evaluation Metrics

The system evaluates models using:
- Accuracy
- F1-Score
- Unweighted Average Recall (UAR)

## Analysis Components

### 1. HuBERT Layer Analysis
Determines which layer of the HuBERT model provides the best accent discrimination.

### 2. Cross-age Generalization
Evaluates how well models trained on adult speech generalize to children's speech.

### 3. Linguistic Level Comparison
Compares performance between word-level and sentence-level classification.

## Running the Demo Web Application

### Start the Flask Server

```bash
python app.py
```

### Access the Application

Open your web browser and navigate to:
```
http://localhost:5000
```

### Using the Web Interface

1. **Upload Page**: Upload an English speech sample audio file
2. **Analysis**: The system will detect the native language and show confidence scores
3. **Recommendation**: Get cuisine recommendations based on the detected language
4. **Demo Page**: View sample predictions without uploading files

### Supported Audio Formats
- WAV
- MP3
- M4A
- FLAC

## Project Structure

```
NLP/
├── app.py                 # Flask web application
├── train_nli.py           # Model training script
├── dataset_loader.py      # Dataset handling
├── config.py              # Configuration settings
├── requirements.txt       # Dependencies
├── data/                  # Dataset storage
├── models/                # Model definitions
├── features/              # Feature extraction
├── utils/                 # Utility functions
├── app/                   # Application components
├── notebooks/             # Analysis notebooks
├── templates/             # Flask HTML templates
├── uploads/               # Uploaded audio files
└── results/               # Evaluation results
```

## Troubleshooting

### torchaudio Issues

If you encounter torchaudio installation problems:

```bash
# Uninstall existing versions
pip uninstall torch torchaudio -y

# Reinstall with CPU version
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### NumPy Compatibility

If you see NumPy compatibility warnings:

```bash
# Downgrade NumPy to compatible version
pip install numpy==1.24.3
```

### Dataset Loading Issues

If the dataset fails to load:
1. Ensure you have internet connection
2. Check Hugging Face access
3. Try manual download from the dataset page

## Expected Results

With the complete system:
- **MFCC Accuracy**: ~72%
- **HuBERT Accuracy**: ~84%
- **Best HuBERT Layer**: Layer 7
- **Cross-age Performance Drop**: ~6% for HuBERT vs ~8% for MFCC

## Extending the System

### Adding New Languages
1. Update `LANGUAGES` list in `config.py`
2. Add cuisine mappings in `CUISINE_MAPPING`
3. Retrain models with extended dataset

### Adding New Features
1. Implement feature extractor in `features/extractor.py`
2. Update dataset class in `data/dataset.py`
3. Modify training script to support new features

## Need Help?

If you encounter issues:
1. Check error messages carefully
2. Verify all dependencies are installed
3. Ensure compatible package versions
4. Consult the PyTorch and Hugging Face documentation