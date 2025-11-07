# Getting Started with Native Language Identification

This guide will help you set up and run the Native Language Identification (NLI) system for Indian English speakers.

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Git (for cloning the repository, if needed)

## Installation

### 1. Clone the Repository (if needed)
```bash
git clone <repository-url>
cd NLI_System
```

### 2. Set Up Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv nli_env

# Activate virtual environment
# On Windows:
nli_env\Scripts\activate
# On macOS/Linux:
source nli_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Setup Script
```bash
python setup.py
```

## Project Structure

```
NLI_System/
├── data/              # Dataset handling
├── models/            # Neural network architectures
├── features/          # Feature extraction modules
├── utils/             # Utility functions
├── app/               # Application code
├── notebooks/         # Jupyter notebooks for analysis
├── results/           # Analysis results and visualizations
├── config.py          # Project configuration
├── train_nli.py       # Main training script
├── main.py            # Application entry point
└── requirements.txt   # Dependencies
```

## Quick Start

### Run the Demo Application
```bash
python main.py --demo
```

### Run the Complete Pipeline
```bash
python main.py --all
```

### Train Models
```bash
# Train with MFCC features
python train_nli.py --feature-type mfcc

# Train with HuBERT features
python train_nli.py --feature-type hubert
```

## Detailed Usage

### 1. Dataset Preparation
The system uses the IndicAccentDb dataset:
- Download from: https://huggingface.co/datasets/DarshanaS/IndicAccentDb
- Supported languages: Hindi, Tamil, Telugu, Malayalam, Kannada, Punjabi, Bengali, Gujarati

### 2. Feature Extraction
Two types of features are supported:
- **MFCC**: Traditional Mel-Frequency Cepstral Coefficients
- **HuBERT**: Self-supervised speech representations

### 3. Model Training
Three model architectures are implemented:
- **CNN Classifier**: Convolutional neural network
- **BiLSTM Classifier**: Bidirectional LSTM
- **Transformer Classifier**: Self-attention model

### 4. Experimental Analysis
The system includes comprehensive analysis:
- HuBERT layer-wise performance evaluation
- Cross-age generalization (adults vs children)
- Linguistic level comparison (word vs sentence)

### 5. Application
The accent-aware cuisine recommendation system demonstrates real-world usage:
- Detects native language from English speech
- Recommends region-specific cuisines based on accent

## Running Analysis Notebooks

The project includes Jupyter notebooks for detailed analysis:
```bash
# Start Jupyter notebook server
jupyter notebook

# Then open notebooks in the 'notebooks/' directory:
# - hubert_layer_analysis.ipynb
# - word_vs_sentence_analysis.ipynb
# - cross_age_generalization.ipynb
```

## Configuration

Key parameters can be adjusted in [config.py](file:///c%3A/Users/Hp/OneDrive/Desktop/NLP/config.py):
- Audio processing parameters
- Model hyperparameters
- Language mappings
- Cuisine recommendations

## Results

Expected performance:
- **MFCC Accuracy**: ~72%
- **HuBERT Accuracy**: ~84%
- **Best HuBERT Layer**: Layer 7
- **Cross-age Robustness**: HuBERT shows better generalization

## Troubleshooting

### Common Issues

1. **Dependency Installation Errors**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **CUDA Issues**:
   - The system will automatically use CPU if CUDA is not available
   - For GPU training, ensure compatible CUDA drivers are installed

3. **Dataset Loading Issues**:
   - Ensure the dataset is properly downloaded and placed in the [data/](file:///c%3A/Users/Hp/OneDrive/Desktop/NLP/data) directory
   - Check dataset format matches expected structure

### Testing Installation
```bash
python test_installation.py
```

## Extending the System

### Adding New Languages
1. Update `LANGUAGES` list in [config.py](file:///c%3A/Users/Hp/OneDrive/Desktop/NLP/config.py)
2. Add cuisine mappings in `CUISINE_MAPPING`
3. Retrain models with extended dataset

### Adding New Features
1. Implement feature extractor in [features/extractor.py](file:///c%3A/Users/Hp/OneDrive/Desktop/NLP/features/extractor.py)
2. Update dataset class in [data/dataset.py](file:///c%3A/Users/Hp/OneDrive/Desktop/NLP/data/dataset.py)
3. Modify training script to support new features

### Adding New Models
1. Implement new classifier in [models/classifiers.py](file:///c%3A/Users/Hp/OneDrive/Desktop/NLP/models/classifiers.py)
2. Update training script to support new model architecture

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](file:///c%3A/Users/Hp/OneDrive/Desktop/NLP/LICENSE) file for details.

## Acknowledgments

- Facebook AI for the HuBERT model
- Hugging Face for the Transformers library
- The creators of the IndicAccentDb dataset