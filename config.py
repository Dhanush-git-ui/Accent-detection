import os

# Project configuration
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'results', 'models')
FEATURES_DIR = os.path.join(PROJECT_ROOT, 'data', 'features')
APP_DIR = os.path.join(PROJECT_ROOT, 'app')
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, 'notebooks')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Dataset configuration
DATASET_NAME = "DarshanaS/IndicAccentDb"
# 6 Indian languages from HuggingFace dataset (8,116 samples total)
LANGUAGES = ['gujarati', 'hindi', 'kannada', 'malayalam', 'tamil', 'telugu']
NUM_LANGUAGES = len(LANGUAGES)

# Audio configuration
SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 10  # seconds

# Feature configuration
# MFCC: 39 channels (13 MFCC + 13 delta + 13 delta-delta)
MFCC_FEATURES = 39
HUBERT_MODEL = "facebook/hubert-base-ls960"
HUBERT_LAYER = 7  # Layer to extract embeddings from
HUBERT_HIDDEN_DIM = 768

# Enhanced model configuration for 90-95% accuracy
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
EPOCHS = 100
DROPOUT_RATE = 0.3
LABEL_SMOOTHING = 0.1

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.45  # Minimum confidence (45%) to accept prediction

# Cuisine recommendations by language
CUISINE_MAPPING = {
    "gujarati": [
        "Dhokla", "Undhiyu", "Fafda", "Gujarati Thali",
        "Khandvi", "Khichu", "Locho", "Ghari"
    ],
    "hindi": [
        "Chole Bhature", "Butter Chicken", "Rogan Josh", "Samosa",
        "Paratha", "Dal Makhani", "Paneer Tikka", "Rasgulla"
    ],
    "kannada": [
        "Rava Idli", "Mysore Pak", "Bisi Bele Bath", "Mangalorean Fish Curry",
        "Neer Dosa", "Jolada Roti", "Enne Badnekai", "Kori Rotti"
    ],
    "malayalam": [
        "Appam", "Puttu", "Avial", "Kerala Sadya",
        "Karimeen Pollichathu", "Malabar Parotta", "Kappa Biryani", "Erachi Varutharacha Curry"
    ],
    "tamil": [
        "Dosa", "Idli", "Sambar", "Chettinad Chicken",
        "Pongal", "Kothu Parotta", "Fish Moilee", "Arisi Payasam"
    ],
    "telugu": [
        "Pesarattu", "Pulihora", "Gongura Pachadi", "Hyderabadi Biryani",
        "Bobbatlu", "Royyala Iguru", "Gutti Vankaya Kura", "Poothareku"
    ]
}