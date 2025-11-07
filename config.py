import os

# Project configuration
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
FEATURES_DIR = os.path.join(PROJECT_ROOT, 'features')
UTILS_DIR = os.path.join(PROJECT_ROOT, 'utils')
APP_DIR = os.path.join(PROJECT_ROOT, 'app')
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, 'notebooks')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# Dataset configuration
DATASET_NAME = "DarshanaS/IndicAccentDb"
LANGUAGES = ['hindi', 'tamil', 'telugu', 'malayalam', 'kannada', 'punjabi', 'bengali', 'gujarati']
NUM_LANGUAGES = len(LANGUAGES)

# Audio configuration
SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 10  # seconds

# Feature configuration
MFCC_FEATURES = 13
HUBERT_MODEL = "facebook/hubert-base-ls960"

# Model configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
DROPOUT_RATE = 0.5

# Application configuration
CUISINE_MAPPING = {
    'hindi': ['Butter Chicken', 'Naan', 'Palak Paneer'],
    'tamil': ['Dosa', 'Idli', 'Sambar'],
    'telugu': ['Hyderabadi Biryani', 'Gongura Pachadi', 'Pesarattu'],
    'malayalam': ['Appam', 'Puttu', 'Avial'],
    'kannada': ['Rava Idli', 'Mysore Pak', 'Bisi Bele Bath'],
    'punjabi': ['Butter Chicken', 'Amritsari Kulcha', 'Sarson da Saag'],
    'bengali': ['Fish Curry', 'Rasgulla', 'Mishti Doi'],
    'gujarati': ['Dhokla', 'Thepla', 'Undhiyu']
}