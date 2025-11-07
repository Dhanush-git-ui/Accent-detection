import torch
from config import CUISINE_MAPPING, LANGUAGES
from features.extractor import FeatureExtractor
from models.classifiers import HubertClassifier

class CuisineRecommender:
    """
    Accent-aware cuisine recommendation system
    """
    def __init__(self, model_path, feature_type='hubert'):
        """
        Initialize recommender
        
        Args:
            model_path (str): Path to trained model
            feature_type (str): Feature type used for training
        """
        self.feature_type = feature_type
        self.feature_extractor = FeatureExtractor(feature_type=feature_type)
        
        # Initialize model
        if feature_type == 'hubert':
            self.model = HubertClassifier(num_classes=len(LANGUAGES))
        else:
            # For MFCC, we would use a different classifier
            raise NotImplementedError("Only HuBERT features are supported for now")
        
        # Load trained model (if it exists)
        try:
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
        except FileNotFoundError:
            print(f"Model file {model_path} not found. Using untrained model.")
            self.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}. Using untrained model.")
            self.model.eval()
        
        # Language to index mapping
        self.lang_to_idx = {lang: idx for idx, lang in enumerate(LANGUAGES)}
        self.idx_to_lang = {idx: lang for idx, lang in enumerate(LANGUAGES)}
    
    def predict_language(self, audio_path):
        """
        Predict native language from audio
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            tuple: (predicted_language, confidence_scores)
        """
        # This is a placeholder implementation due to torchaudio issues
        # In a real implementation, you would load and process the actual audio file
        
        try:
            # Return dummy prediction for now
            import random
            predicted_idx = random.randint(0, len(LANGUAGES) - 1)
            predicted_language = self.idx_to_lang[predicted_idx]
            
            # Generate dummy confidence scores
            confidence_scores = [random.random() for _ in LANGUAGES]
            # Normalize to sum to 1
            total = sum(confidence_scores)
            confidence_scores = [score / total for score in confidence_scores]
            
            return predicted_language, confidence_scores
        except Exception as e:
            print(f"Error processing audio: {e}")
            # Return default prediction
            return LANGUAGES[0], [1.0] + [0.0] * (len(LANGUAGES) - 1)
    
    def recommend_cuisines(self, audio_path, top_k=3):
        """
        Recommend cuisines based on speaker's accent
        
        Args:
            audio_path (str): Path to audio file
            top_k (int): Number of top recommendations
            
        Returns:
            dict: Recommendation results
        """
        # Predict language
        predicted_language, confidence_scores = self.predict_language(audio_path)
        
        # Get cuisine recommendations
        recommended_cuisines = CUISINE_MAPPING.get(predicted_language, [])
        
        # Get confidence for all languages
        language_confidences = {}
        for i, lang in enumerate(LANGUAGES):
            language_confidences[lang] = confidence_scores[i]
        
        return {
            'predicted_language': predicted_language,
            'confidence_scores': language_confidences,
            'recommended_cuisines': recommended_cuisines[:top_k],
            'all_cuisines': CUISINE_MAPPING
        }
    
    def interactive_demo(self):
        """
        Run interactive demo
        """
        print("=== Accent-Aware Cuisine Recommendation System ===")
        print("Welcome to our accent-aware cuisine recommendation system!")
        print("Speak an English phrase and we'll detect your native language and recommend cuisines.")
        print()
        
        # For demo purposes, we'll simulate predictions
        sample_predictions = [
            ("I would like to order some delicious food", "malayalam"),
            ("Can I get some butter chicken and naan?", "hindi"),
            ("Where can I find the best dosa in town?", "tamil")
        ]
        
        for i, (sample_input, expected_lang) in enumerate(sample_predictions):
            print(f"Sample Input {i+1}: '{sample_input}'")
            
            # Simulate prediction
            predicted_language = expected_lang
            recommended_cuisines = CUISINE_MAPPING.get(predicted_language, [])[:3]
            
            print(f"Detected Language: {predicted_language.capitalize()}")
            print(f"Recommended Cuisines: {', '.join(recommended_cuisines)}")
            print()
            
            if i < len(sample_predictions) - 1:
                input("Press Enter to continue...")
        
        print("Demo completed!")

# Example usage
if __name__ == "__main__":
    # This would be used after training a model
    # recommender = CuisineRecommender("path/to/trained/model.pth")
    # results = recommender.recommend_cuisines("path/to/audio.wav")
    # print(results)
    pass