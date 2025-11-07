# Native Language Identification of Indian English Speakers Using HuBERT

## Project Overview

This project develops a system to identify the native language (L1) of Indian speakers by analyzing accent patterns in their English speech. The system uses both traditional acoustic features (MFCCs) and self-supervised representations (HuBERT embeddings) for classification.

## System Architecture

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

## Methodology

### 1. Feature Extraction

#### Traditional Features: MFCCs
- Extract 13 Mel-Frequency Cepstral Coefficients from speech signals
- Standard preprocessing: pre-emphasis, framing, windowing
- Normalization to zero mean and unit variance

#### Self-supervised Features: HuBERT
- Utilize pre-trained HuBERT Base model (960 hours of LibriSpeech)
- Extract embeddings from different layers for analysis
- Compare performance across layers to identify optimal representations

### 2. Model Architectures

#### CNN-based Classifier
- 1D Convolutional layers for temporal feature extraction
- Batch normalization and max pooling
- Fully connected layers for classification

#### BiLSTM Classifier
- Bidirectional LSTM for sequence modeling
- Captures long-term dependencies in speech
- Final hidden states for classification

#### Transformer Classifier
- Self-attention mechanisms for global context
- Multi-head attention for parallel feature processing
- Positional encoding for temporal information

### 3. Experimental Evaluation

#### Cross-age Generalization
- Train on adult speech samples
- Test on children's speech samples
- Analyze robustness of feature representations

#### Linguistic Level Analysis
- Word-level classification using segmented speech
- Sentence-level classification using full utterances
- Compare accuracy, robustness, and interpretability

## Implementation Details

### Dataset
The Indian Accent Database (IndicAccentDb) contains English speech recordings from Indian speakers of various native language backgrounds:
- Hindi
- Tamil
- Telugu
- Malayalam
- Kannada
- Punjabi
- Bengali
- Gujarati

The dataset is divided into adult and child subsets for cross-age generalization studies.

### Training Process
1. Data preprocessing and feature extraction
2. Model initialization with appropriate architecture
3. Training with Adam optimizer (learning rate = 0.001)
4. Validation every epoch with early stopping
5. Evaluation on test set with comprehensive metrics

### Evaluation Metrics
- Overall accuracy
- Per-class precision, recall, and F1-score
- Confusion matrices
- Cross-age performance drops

## Results and Analysis

### Feature Comparison
HuBERT embeddings consistently outperform traditional MFCC features:
- MFCC accuracy: ~72%
- HuBERT accuracy: ~84%
- Improvement: ~12 percentage points

### HuBERT Layer Analysis
- Middle layers (6-9) show optimal performance
- Lower layers capture acoustic information
- Higher layers capture linguistic information
- Layer 7 provides the best balance for accent identification

### Cross-age Generalization
- Performance drop when testing on children's speech
- MFCC drop: ~8%
- HuBERT drop: ~6%
- HuBERT shows better robustness to age variations

### Linguistic Level Comparison
- Sentence-level outperforms word-level
- Sentence accuracy: ~84%
- Word accuracy: ~76%
- Longer context provides better accent cues

## Application: Accent-Aware Cuisine Recommendation

A real-world application demonstrates how accent detection can enhance personalization:
1. Customer speaks English phrase
2. System detects native language from accent
3. Infers regional background
4. Recommends region-specific cuisines

Example mappings:
- Malayalam accent → Kerala cuisine (Appam, Puttu, Avial)
- Punjabi accent → North Indian cuisine (Butter Chicken, Kulcha)
- Tamil accent → South Indian cuisine (Dosa, Idli, Sambar)

## Conclusion

This project successfully demonstrates:
1. Superior performance of self-supervised HuBERT embeddings over traditional MFCCs
2. Optimal layer identification for accent-related information in HuBERT
3. Better cross-age generalization with HuBERT features
4. Improved performance with sentence-level analysis
5. Practical application in personalized recommendation systems

The system achieves over 84% accuracy in native language identification and demonstrates strong potential for real-world applications in personalized services.

## Future Work

1. Extend to more Indian languages
2. Investigate other self-supervised models (Wav2Vec 2.0, Data2Vec)
3. Explore fine-tuning HuBERT on Indian English data
4. Implement real-time inference for practical deployment
5. Study speaker-independent vs speaker-adaptive approaches