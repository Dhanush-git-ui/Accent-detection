# Native Language Identification Project - Summary

## Project Overview

This project implements a Native Language Identification (NLI) system for Indian English speakers using both traditional acoustic features (MFCCs) and self-supervised representations (HuBERT embeddings).

## Project Structure

```
NLI_System/
├── data/
│   ├── dataset.py           # Dataset handling and preprocessing
│   └── __init__.py
├── models/
│   ├── classifiers.py       # Model architectures (CNN, BiLSTM, Transformer)
│   ├── trainer.py           # Training and evaluation utilities
│   └── __init__.py
├── features/
│   ├── extractor.py         # Feature extraction (MFCC, HuBERT)
│   └── __init__.py
├── utils/
│   ├── helpers.py           # Utility functions
│   └── __init__.py
├── app/
│   ├── cuisine_recommender.py  # Accent-aware cuisine recommendation
│   └── __init__.py
├── notebooks/
│   ├── hubert_layer_analysis.ipynb      # HuBERT layer-wise analysis
│   ├── word_vs_sentence_analysis.ipynb  # Linguistic level comparison
│   └── cross_age_generalization.ipynb   # Cross-age evaluation
├── results/
│   └── analysis.py          # Results visualization and analysis
├── config.py                # Project configuration
├── train_nli.py             # Main training script
├── main.py                  # Application entry point
├── setup.py                 # Environment setup
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
```

## Key Components

### 1. Feature Extraction
- **MFCC Features**: Traditional Mel-Frequency Cepstral Coefficients
- **HuBERT Features**: Self-supervised speech representations

### 2. Model Architectures
- **CNN Classifier**: Convolutional neural network for temporal feature extraction
- **BiLSTM Classifier**: Bidirectional LSTM for sequence modeling
- **Transformer Classifier**: Self-attention mechanisms for global context

### 3. Experimental Analysis
- **HuBERT Layer Analysis**: Identification of optimal layers for accent information
- **Cross-age Generalization**: Performance on adults vs children
- **Linguistic Level Comparison**: Word-level vs sentence-level classification

### 4. Application
- **Accent-Aware Cuisine Recommendation**: Real-world application demonstrating personalization

## How to Use

1. **Setup Environment**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Dataset**:
   - Dataset: https://huggingface.co/datasets/DarshanaS/IndicAccentDb

3. **Train Models**:
   ```bash
   python train_nli.py --feature-type hubert
   ```

4. **Run Analysis**:
   - Jupyter notebooks in the [notebooks/](file:///c%3A/Users/Hp/OneDrive/Desktop/NLP/notebooks) directory

5. **Run Application**:
   ```bash
   python main.py --demo
   ```

## Expected Results

- **Overall Accuracy**: ~84% with HuBERT features vs ~72% with MFCCs
- **Best HuBERT Layer**: Layer 7 for accent identification
- **Cross-age Performance**: Better generalization with HuBERT features
- **Linguistic Level**: Sentence-level outperforms word-level

## Technologies Used

- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face library for pre-trained models
- **Librosa**: Audio processing
- **Scikit-learn**: Machine learning utilities
- **Jupyter**: Interactive analysis notebooks

## Future Improvements

1. Fine-tuning HuBERT on Indian English data
2. Expanding to more Indian languages
3. Real-time inference implementation
4. Speaker-independent vs speaker-adaptive approaches
5. Integration with speech recognition systems