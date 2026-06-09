"""
Flask web application for accent detection and cuisine recommendation
"""
import sys
import os
import time

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
import torch
import torch.nn as nn
from config import LANGUAGES, CUISINE_MAPPING, SAMPLE_RATE, MFCC_FEATURES, MODELS_DIR
from werkzeug.utils import secure_filename

# Import models and extractor from src package
from src.models.classifiers import MFCCClassifier, HubertClassifier
from src.features.extractor import FeatureExtractor

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac'}

# Device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"App running on device: {device}")

# Global cache for extractors and models to avoid reloading on each request
global_extractors = {}
global_models = {}

def get_extractor(feature_type):
    """Get or load feature extractor cached globally"""
    if feature_type not in global_extractors:
        print(f"Initializing global FeatureExtractor for: {feature_type}...")
        global_extractors[feature_type] = FeatureExtractor(feature_type=feature_type)
    return global_extractors[feature_type]

def get_model(feature_type):
    """Get or load classifier model cached globally, applying optimizations"""
    if feature_type not in global_models:
        print(f"Initializing global Classifier Model for: {feature_type}...")
        num_classes = len(LANGUAGES)
        
        if feature_type == 'mfcc':
            model = MFCCClassifier(num_classes=num_classes)
            model_path = os.path.join(MODELS_DIR, "mfcc_enhanced.pth")
            if not os.path.exists(model_path):
                model_path = os.path.join(MODELS_DIR, "mfcc_baseline.pth")
        else:
            model = HubertClassifier(num_classes=num_classes)
            model_path = os.path.join(MODELS_DIR, "hubert_enhanced.pth")
            if not os.path.exists(model_path):
                model_path = os.path.join(MODELS_DIR, "hubert_baseline.pth")
                
        # Load weights if checkpoint exists
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif isinstance(checkpoint, dict):
                    model.load_state_dict(checkpoint)
                else:
                    model = checkpoint
                print(f"Successfully loaded trained {feature_type} model from {model_path}")
            except Exception as e:
                print(f"Error loading model weights from {model_path}: {e}. Using random weights.")
        else:
            print(f"Model checkpoint not found at {model_path}. Running with initialized weights.")
            
        model.to(device)
        model.eval()
        
        # CPU Optimization: Apply dynamic quantization to linear projections
        if device == torch.device('cpu'):
            print(f"Applying dynamic quantization to {feature_type} classifier on CPU...")
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            
        global_models[feature_type] = model
        
    return global_models[feature_type]

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_audio_file(filepath):
    """Preprocess uploaded audio file"""
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

def extract_mfcc_features(audio_array):
    """Extract 39-channel MFCC features (13 MFCC + 13 delta + 13 delta-delta) matching training format"""
    try:
        import numpy as np
        mfccs = librosa.feature.mfcc(y=audio_array, sr=SAMPLE_RATE, n_mfcc=13,
                                     n_fft=2048, hop_length=512)
        delta = librosa.feature.delta(mfccs)
        delta2 = librosa.feature.delta(mfccs, order=2)
        mfccs_39 = np.concatenate([mfccs, delta, delta2], axis=0)  # (39, time)
        
        # Convert to tensor
        features = torch.FloatTensor(mfccs_39)
        
        # Pad or truncate to fixed length (1000 frames)
        target_length = 1000
        if features.shape[1] > target_length:
            features = features[:, :target_length]
        elif features.shape[1] < target_length:
            pad_length = target_length - features.shape[1]
            features = torch.nn.functional.pad(features, (0, pad_length))
        
        return features.unsqueeze(0)  # Add batch dimension -> (1, 39, 1000)
    except Exception as e:
        print(f"Error extracting MFCC features: {e}")
        return torch.randn(1, 39, 1000)

@app.route('/health')
def health():
    """Health check endpoint - returns status of loaded models"""
    return jsonify({
        'status': 'ok',
        'device': str(device),
        'models_loaded': list(global_models.keys()),
        'available_languages': LANGUAGES
    }), 200

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and run selected model predictions"""
    try:
        print("Upload endpoint called")
        
        if 'audio_file' not in request.files:
            print("No file in request")
            return jsonify({'error': 'No file selected'}), 400
        
        file = request.files['audio_file']
        model_mode = request.form.get('model_mode', 'mfcc').lower()  # 'mfcc' or 'hubert'
        
        if file.filename == '':
            print("Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        print(f"Received file: {file.filename} (Model Mode: {model_mode})")
        
        if file and allowed_file(file.filename):
            # Secure the filename
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            print(f"Saving file to: {filepath}")
            file.save(filepath)
            
            # Start latency timer
            start_time = time.perf_counter()
            
            # 1. Process the audio file
            print("Processing audio file...")
            audio_array = preprocess_audio_file(filepath)
            
            # 2. Retrieve global model and extractor
            extractor = get_extractor(model_mode)
            model = get_model(model_mode)
            
            # 3. Extract features
            if model_mode == 'mfcc':
                print("Extracting MFCC features...")
                features = extract_mfcc_features(audio_array)
                features = features.to(device)
            else:  # hubert
                print("Extracting HuBERT features (sliced to max 5s)...")
                hubert_layers = extractor.extract_hubert_layers(
                    audio_array, SAMPLE_RATE, layers=[7], max_duration=5
                )
                features = hubert_layers[7].unsqueeze(0).to(device)
                
            print("Feature extraction completed. Running classifier inference...")
            
            # 4. Predict language
            with torch.no_grad():
                outputs = model(features)
                probabilities = torch.softmax(outputs, dim=1)
                confidence_scores = probabilities.squeeze().tolist()
                predicted_idx = torch.argmax(probabilities).item()
                
            if not isinstance(confidence_scores, list):
                confidence_scores = [confidence_scores]
                
            predicted_language = LANGUAGES[predicted_idx] if predicted_idx < len(LANGUAGES) else LANGUAGES[0]
            max_confidence = max(confidence_scores)
            confidence_threshold = 0.45
            
            # Check if confidence is too low
            if max_confidence < confidence_threshold:
                print(f"Low confidence prediction: {max_confidence*100:.1f}%")
                language_confidences = {}
                for i, lang in enumerate(LANGUAGES):
                    if i < len(confidence_scores):
                        language_confidences[lang] = float(confidence_scores[i])
                    else:
                        language_confidences[lang] = 0.0
                
                return jsonify({
                    'success': False,
                    'error': f'Low confidence prediction ({max_confidence*100:.1f}% < {confidence_threshold*100:.0f}%)',
                    'message': 'Please try a longer audio clip (5+ seconds) for better accuracy',
                    'confidence_scores': language_confidences,
                    'max_confidence': float(max_confidence)
                }), 400
            
            print(f"Prediction completed: {predicted_language}")
            
            # Stop latency timer
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            
            # Get cuisine recommendations
            cuisines = get_cuisine_recommendations(predicted_language)
            
            # Prepare response
            language_confidences = {}
            for i, lang in enumerate(LANGUAGES):
                if i < len(confidence_scores):
                    language_confidences[lang] = float(confidence_scores[i])
                else:
                    language_confidences[lang] = 0.0
            
            response = {
                'success': True,
                'predicted_language': predicted_language.replace('_', ' ').title(),
                'confidence_scores': language_confidences,
                'max_confidence': float(max_confidence),
                'recommended_cuisines': cuisines,
                'latency_ms': round(latency_ms, 2),
                'model_used': 'HuBERT Transformer' if model_mode == 'hubert' else 'MFCC CNN'
            }
            
            print(f"Response prepared successfully. Latency: {latency_ms:.2f} ms")
            return jsonify(response)
        else:
            print(f"Invalid file format: {file.filename}")
            return jsonify({'error': 'Invalid file format. Please upload a WAV, MP3, M4A, or FLAC file.'}), 400
            
    except Exception as e:
        print(f"Upload error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/demo')
def demo():
    """Demo page with sample predictions"""
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

# Pre-warm extractors and models at startup to avoid delay on first user upload
try:
    print("Pre-warming models at startup...")
    get_extractor('mfcc')
    get_model('mfcc')
    print("Models pre-warmed successfully!")
except Exception as e:
    print(f"Warning warming up models: {e}")

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000, threaded=True)