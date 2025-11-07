"""
Flask web application for accent detection and cuisine recommendation
"""
from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import librosa
import torch
import torch.nn as nn
from config import LANGUAGES, CUISINE_MAPPING, SAMPLE_RATE, MFCC_FEATURES
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac'}

# Simple MFCC model for demo (matching the baseline model)
class MFCCDemoModel(nn.Module):
    def __init__(self, num_classes, input_channels=13, dropout_rate=0.5):
        super(MFCCDemoModel, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        
        self.fc_input_size = 256 * 125
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        
        x = x.view(-1, self.fc_input_size)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_audio_file(filepath):
    """
    Preprocess uploaded audio file
    
    Args:
        filepath (str): Path to audio file
        
    Returns:
        np.array: Preprocessed audio array
    """
    try:
        # Load audio file
        audio_array, sr = librosa.load(filepath, sr=None)
        
        # Convert to mono if stereo
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)
        
        # Resample to 16 kHz
        if sr != SAMPLE_RATE:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=SAMPLE_RATE)
            
        return audio_array
    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        # Return dummy audio for demo
        return np.random.randn(SAMPLE_RATE)  # 1 second of random audio

def extract_mfcc_features(audio_array, n_mfcc=MFCC_FEATURES):
    """
    Extract MFCC features from audio array
    
    Args:
        audio_array (np.array): Audio data
        n_mfcc (int): Number of MFCC coefficients
        
    Returns:
        torch.Tensor: MFCC features tensor
    """
    try:
        mfccs = librosa.feature.mfcc(y=audio_array, sr=SAMPLE_RATE, n_mfcc=n_mfcc)
        # Normalize
        mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
        
        # Convert to tensor and ensure consistent shape
        features = torch.FloatTensor(mfccs)
        
        # Pad or truncate to fixed length (e.g., 1000 frames)
        target_length = 1000
        if features.shape[1] > target_length:
            features = features[:, :target_length]
        elif features.shape[1] < target_length:
            pad_length = target_length - features.shape[1]
            features = torch.nn.functional.pad(features, (0, pad_length))
        
        return features.unsqueeze(0)  # Add batch dimension
    except Exception as e:
        print(f"Error extracting MFCC features: {e}")
        # Return dummy features
        return torch.randn(1, n_mfcc, 1000)

def load_trained_model(model_path="models/mfcc_baseline.pth"):
    """
    Load trained model
    
    Args:
        model_path (str): Path to trained model
        
    Returns:
        nn.Module: Loaded model or None if failed
    """
    try:
        model = MFCCDemoModel(num_classes=len(LANGUAGES))
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print("Model loaded successfully")
        else:
            # Initialize with random weights for demo
            print("Model file not found, using random weights for demo")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_language(features, model):
    """
    Predict native language from features
    
    Args:
        features (torch.Tensor): Input features
        model (nn.Module): Trained model
        
    Returns:
        tuple: (predicted_language, confidence_scores)
    """
    try:
        if model is not None:
            with torch.no_grad():
                outputs = model(features)
                probabilities = torch.softmax(outputs, dim=1)
                confidence_scores = probabilities.squeeze().tolist()
                predicted_idx = torch.argmax(probabilities).item()
                predicted_language = LANGUAGES[predicted_idx]
        else:
            # Random prediction for demo
            import random
            predicted_idx = random.randint(0, len(LANGUAGES) - 1)
            predicted_language = LANGUAGES[predicted_idx]
            
            # Generate random confidence scores
            confidence_scores = [random.random() for _ in LANGUAGES]
            # Normalize to sum to 1
            total = sum(confidence_scores)
            confidence_scores = [score/total for score in confidence_scores]
        
        return predicted_language, confidence_scores
    except Exception as e:
        print(f"Error predicting language: {e}")
        # Default prediction
        return LANGUAGES[0], [1.0] + [0.0] * (len(LANGUAGES) - 1)

def get_cuisine_recommendations(language):
    """
    Get cuisine recommendations based on language
    
    Args:
        language (str): Predicted language
        
    Returns:
        list: Recommended cuisines
    """
    return CUISINE_MAPPING.get(language, ["Unknown cuisine"])

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['audio_file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Secure the filename
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the audio file
        audio_array = preprocess_audio_file(filepath)
        
        # Extract features
        features = extract_mfcc_features(audio_array)
        
        # Load model
        model = load_trained_model()
        
        # Predict language
        predicted_language, confidence_scores = predict_language(features, model)
        
        # Get cuisine recommendations
        cuisines = get_cuisine_recommendations(predicted_language)
        
        # Prepare response
        language_confidences = {}
        for i, lang in enumerate(LANGUAGES):
            language_confidences[lang] = float(confidence_scores[i]) if isinstance(confidence_scores, list) else 0.0
        
        response = {
            'success': True,
            'predicted_language': predicted_language.capitalize(),
            'confidence_scores': language_confidences,
            'recommended_cuisines': cuisines
        }
        
        return jsonify(response)
    else:
        return jsonify({'error': 'Invalid file format. Please upload a WAV, MP3, M4A, or FLAC file.'}), 400

@app.route('/demo')
def demo():
    """Demo page with sample predictions"""
    # Sample predictions for demo
    samples = [
        {
            'input': 'I would like to order some delicious food',
            'language': 'Malayalam',
            'cuisines': ['Appam', 'Puttu', 'Avial']
        },
        {
            'input': 'Can I get some butter chicken and naan?',
            'language': 'Hindi',
            'cuisines': ['Butter Chicken', 'Naan', 'Palak Paneer']
        },
        {
            'input': 'Where can I find the best dosa in town?',
            'language': 'Tamil',
            'cuisines': ['Dosa', 'Idli', 'Sambar']
        }
    ]
    
    return render_template('demo.html', samples=samples)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)