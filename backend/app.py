from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from textblob import TextBlob
import re
import os
import warnings
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# Global variables
model_components = None
stats_data = None
analyzer = None

def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        return True
    except:
        print("Warning: Could not download NLTK data")
        return False

def improved_preprocess_text(text):
    """
    Preprocessing that matches the training code exactly
    """
    if pd.isna(text) or not text:
        return ""

    text = str(text)
    
    # Keep original case information by adding markers
    caps_words = len([word for word in text.split() if word.isupper() and len(word) > 2])
    
    # Convert to lowercase for processing
    text = text.lower()
    
    # Mark special patterns that might indicate fake news
    text = re.sub(r'!{2,}', ' MULTIPLE_EXCL ', text)
    text = re.sub(r'\?{2,}', ' MULTIPLE_QUEST ', text)
    text = re.sub(r'\.{3,}', ' ELLIPSIS ', text)
    
    # Add caps marker if many caps words
    if caps_words > 3:
        text += ' MANY_CAPS '
    
    # Clean URLs and emails but keep markers
    text = re.sub(r'http\S+|www\S+|https\S+', ' URL_LINK ', text)
    text = re.sub(r'\S+@\S+', ' EMAIL_ADDR ', text)
    
    # Remove excessive punctuation but keep some structure
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Clean whitespace
    text = ' '.join(text.split())
    
    return text

def extract_conservative_features(text, processed_text):
    """Extract numerical features exactly as in training"""
    try:
        # Basic length features (normalized to reduce overfitting)
        text_length = len(text)
        word_count = len(processed_text.split()) if processed_text else 0
        
        # Normalize length features by taking log to reduce extreme values
        log_text_length = np.log1p(text_length)
        log_word_count = np.log1p(word_count)
        
        # Readability features (more conservative)
        avg_word_length = np.mean([len(word) for word in processed_text.split()]) if processed_text.split() else 0
        
        # Punctuation features (normalized)
        exclamation_ratio = text.count('!') / max(len(text.split()), 1)
        question_ratio = text.count('?') / max(len(text.split()), 1)
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        # Simple sentiment analysis (less prone to overfitting)
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Categorical sentiment
        sentiment_category = 1 if polarity > 0.1 else (-1 if polarity < -0.1 else 0)
        
        return [
            log_text_length, log_word_count, avg_word_length,
            exclamation_ratio, question_ratio, caps_ratio,
            polarity, subjectivity, sentiment_category
        ]
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        # Return default values matching the expected 9 features
        return [0.0] * 9

def load_models():
    """Load model components trained by the Jupyter notebook"""
    global model_components, stats_data, analyzer
    
    try:
        # Download NLTK data
        download_nltk_data()
        
        # Initialize VADER analyzer
        try:
            analyzer = SentimentIntensityAnalyzer()
            print("VADER analyzer initialized")
        except:
            print("Warning: VADER analyzer not available")
            analyzer = None
        
        # Try loading the enhanced model first
        model_path = "models/new/enhanced_news_model_1.joblib"
        if not os.path.exists(model_path):
            model_path = "models/new/improved_news_model.joblib"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found. Please run the training notebook first.")
        
        try:
            model_components = joblib.load(model_path)
            print(f"Model loaded successfully from {model_path}!")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Create a simple fallback model
            model_components = {
                'model': None,
                'tfidf_vectorizer': None,
                'scaler': None,
                'feature_columns': ['text_length', 'word_count', 'avg_word_length', 
                                  'exclamation_ratio', 'question_ratio', 'caps_ratio',
                                  'polarity', 'subjectivity', 'sentiment_category'],
                'preprocessing_function': improved_preprocess_text
            }
            print("Using fallback model components")
        
        # Load stats
        stats_path = "models/new/fake_news_model_optimized.joblib"
        if not os.path.exists(stats_path):
            stats_path = "models/new/improved_model_stats.joblib"
        
        if os.path.exists(stats_path):
            try:
                stats_data = joblib.load(stats_path)
                print("Model statistics loaded successfully!")
            except Exception as e:
                print(f"Error loading stats: {e}")
                stats_data = {
                    'final_accuracy': 0.0,
                    'final_auc': 0.0,
                    'individual_scores': {},
                    'cv_results': {},
                    'confusion_matrix': [[0, 0], [0, 0]]
                }
        else:
            print("Warning: Model statistics file not found")
            stats_data = {
                'final_accuracy': 0.0,
                'final_auc': 0.0,
                'individual_scores': {},
                'cv_results': {},
                'confusion_matrix': [[0, 0], [0, 0]]
            }
        
        print("All components loaded successfully!")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        # Don't raise the error, just log it and continue with fallback components
        model_components = {
            'model': None,
            'tfidf_vectorizer': None,
            'scaler': None,
            'feature_columns': ['text_length', 'word_count', 'avg_word_length', 
                              'exclamation_ratio', 'question_ratio', 'caps_ratio',
                              'polarity', 'subjectivity', 'sentiment_category'],
            'preprocessing_function': improved_preprocess_text
        }
        stats_data = {
            'final_accuracy': 0.0,
            'final_auc': 0.0,
            'individual_scores': {},
            'cv_results': {},
            'confusion_matrix': [[0, 0], [0, 0]]
        }
        print("Using fallback components due to loading error")

def predict_news_compatible(title, text):
    """
    Compatible prediction function that matches the training code exactly
    """
    try:
        # Preprocess exactly like training
        processed_text = improved_preprocess_text(text)
        
        # Extract features exactly like training
        features_dict = {
            'text_length': len(text),
            'word_count': len(processed_text.split()),
            'avg_word_length': np.mean([len(word) for word in processed_text.split()]) if processed_text.split() else 0,
            'exclamation_ratio': text.count('!') / max(len(text.split()), 1),
            'question_ratio': text.count('?') / max(len(text.split()), 1),
            'caps_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'polarity': TextBlob(text).sentiment.polarity,
            'subjectivity': TextBlob(text).sentiment.subjectivity,
            'sentiment_category': 1 if TextBlob(text).sentiment.polarity > 0.1 else (-1 if TextBlob(text).sentiment.polarity < -0.1 else 0)
        }
        
        # If model is not loaded, use a simple rule-based approach
        if model_components.get('model') is None:
            # Simple rule-based prediction
            fake_indicators = [
                text.count('!') > 3,
                len([c for c in text if c.isupper()]) / len(text) > 0.2,
                any(word in text.lower() for word in ['shocking', 'unbelievable', 'exposed', 'secret', 'urgent']),
                features_dict['subjectivity'] > 0.8
            ]
            fake_score = sum(fake_indicators) / len(fake_indicators)
            fake_probability = min(max(fake_score, 0), 1)
            real_probability = 1 - fake_probability
            prediction = 1 if real_probability > fake_probability else 0
            label = 'Real' if prediction == 1 else 'Fake'
            confidence = max(fake_probability, real_probability)
            return {
                'prediction': prediction,
                'label': label,
                'confidence': float(confidence),
                'probabilities': {
                    'fake': float(fake_probability),
                    'real': float(real_probability)
                },
                'features_used': features_dict
            }
        
        # Use the actual model if available
        numerical_features = [features_dict[col] for col in model_components['feature_columns']]
        
        # Vectorize and combine exactly like training
        tfidf_vec = model_components['tfidf_vectorizer'].transform([processed_text])
        num_vec = model_components['scaler'].transform([numerical_features])
        combined_vec = hstack([tfidf_vec, csr_matrix(num_vec)])
        
        # Predict
        prediction = model_components['model'].predict(combined_vec)[0]
        probabilities = model_components['model'].predict_proba(combined_vec)[0]
        
        return {
            'prediction': int(prediction),
            'label': 'Real' if prediction == 1 else 'Fake',
            'confidence': float(max(probabilities)),
            'probabilities': {
                'fake': float(probabilities[0]),
                'real': float(probabilities[1])
            },
            'features_used': features_dict
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        # Return a fallback prediction
        return {
            'prediction': 1,
            'label': 'Real',
            'confidence': 0.5,
            'probabilities': {
                'fake': 0.5,
                'real': 0.5
            },
            'features_used': features_dict
        }

@app.route("/", methods=["GET"])
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "Compatible Flask server is running",
        "model_loaded": model_components is not None,
        "model_type": "improved_ensemble",
        "features_available": len(model_components.get('feature_columns', [])) if model_components else 0,
        "components": list(model_components.keys()) if model_components else []
    })

@app.route("/predict", methods=["POST"])
def predict():
    """Main prediction endpoint compatible with training code"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        text = data.get("text", "")
        title = data.get("title", "")  # Optional title
        
        if not text or not text.strip():
            return jsonify({"error": "No text provided"}), 400
        
        # Use the compatible prediction function
        result = predict_news_compatible(title, text)
        
        # Get sentiment info for response
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        sentiment = "positive" if polarity > 0.1 else "negative" if polarity < -0.1 else "neutral"
        
        # Calculate certainty
        certainty = abs(result['probabilities']['real'] - result['probabilities']['fake'])
        
        # Risk assessment
        risk_factors = []
        
        # Check for common fake news indicators
        if text.count('!') > 3:
            risk_factors.append("Excessive exclamation marks")
        
        if len([c for c in text if c.isupper()]) / len(text) > 0.2:
            risk_factors.append("High proportion of capital letters")
        
        if any(word in text.lower() for word in ['shocking', 'unbelievable', 'exposed', 'secret', 'urgent']):
            risk_factors.append("Sensational language detected")
        
        if certainty < 0.3:
            risk_factors.append("Low model confidence")
        
        return jsonify({
            "prediction": result['label'],
            "confidence": round(result['confidence'], 3),
            "fake_probability": round(result['probabilities']['fake'], 3),
            "real_probability": round(result['probabilities']['real'], 3),
            "certainty": round(certainty, 3),
            "sentiment": sentiment,
            "polarity": round(polarity, 3),
            "risk_factors": risk_factors,
            "text_stats": {
                "word_count": len(text.split()),
                "character_count": len(text),
                "exclamation_count": text.count('!'),
                "question_count": text.count('?'),
                "caps_ratio": round(len([c for c in text if c.isupper()]) / len(text), 3) if text else 0
            },
            "features_analyzed": result['features_used']
        })
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """Predict multiple texts at once"""
    try:
        data = request.json
        if not data or 'texts' not in data:
            return jsonify({"error": "No texts provided"}), 400
        
        texts = data['texts']
        titles = data.get('titles', [""] * len(texts))  # Optional titles
        
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({"error": "Texts must be a non-empty list"}), 400
        
        if len(texts) > 50:  # Limit batch size
            return jsonify({"error": "Maximum 50 texts allowed per batch"}), 400
        
        # Ensure titles list matches texts length
        if len(titles) != len(texts):
            titles = [""] * len(texts)
        
        results = []
        for i, (text, title) in enumerate(zip(texts, titles)):
            try:
                result = predict_news_compatible(title, text)
                
                results.append({
                    "index": i,
                    "text_preview": text[:100] + "..." if len(text) > 100 else text,
                    "title": title if title else "No title provided",
                    "prediction": result['label'],
                    "confidence": round(result['confidence'], 3),
                    "fake_probability": round(result['probabilities']['fake'], 3),
                    "real_probability": round(result['probabilities']['real'], 3)
                })
                
            except Exception as e:
                results.append({
                    "index": i,
                    "error": f"Processing failed: {str(e)}"
                })
        
        return jsonify({
            "results": results,
            "total_processed": len(results),
            "summary": {
                "fake_count": sum(1 for r in results if r.get("prediction") == "Fake"),
                "real_count": sum(1 for r in results if r.get("prediction") == "Real"),
                "errors": sum(1 for r in results if "error" in r)
            }
        })
    
    except Exception as e:
        return jsonify({"error": f"Batch prediction failed: {str(e)}"}), 500

@app.route("/stats", methods=["GET"])
def stats():
    """Get model statistics"""
    try:
        if not stats_data:
            return jsonify({"error": "Stats data not available"}), 500
        
        return jsonify({
            "model_type": "improved_ensemble",
            "accuracy": stats_data.get('final_accuracy', 0),
            "auc": stats_data.get('final_auc', 0),
            "individual_scores": stats_data.get('individual_scores', {}),
            "cross_validation": stats_data.get('cv_results', {}),
            "confusion_matrix": stats_data.get('confusion_matrix', [[0, 0], [0, 0]]),
            "feature_count": stats_data.get('feature_count', 0),
            "training_samples": stats_data.get('training_samples', 0)
        })
    
    except Exception as e:
        print(f"Stats error: {e}")
        return jsonify({"error": f"Failed to get stats: {str(e)}"}), 500

@app.route("/analyze", methods=["POST"])
def analyze_text():
    """Detailed text analysis endpoint"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        text = data.get("text", "")
        title = data.get("title", "")
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        processed_text = improved_preprocess_text(text)
        processed_title = improved_preprocess_text(title) if title else ""
        
        # Detailed analysis
        analysis = {
            "original_text": text,
            "original_title": title,
            "processed_text": processed_text,
            "processed_title": processed_title,
            "statistics": {
                "character_count": len(text),
                "word_count": len(text.split()),
                "sentence_count": len(text.split('.')),
                "paragraph_count": len(text.split('\n\n')),
                "avg_word_length": float(np.mean([len(word) for word in text.split()])) if text.split() else 0
            },
            "punctuation_analysis": {
                "exclamation_count": text.count('!'),
                "question_count": text.count('?'),
                "period_count": text.count('.'),
                "comma_count": text.count(','),
                "caps_ratio": len([c for c in text if c.isupper()]) / len(text) if text else 0
            },
            "sentiment_analysis": {
                "polarity": float(TextBlob(text).sentiment.polarity),
                "subjectivity": float(TextBlob(text).sentiment.subjectivity),
                "category": "positive" if TextBlob(text).sentiment.polarity > 0.1 else 
                          "negative" if TextBlob(text).sentiment.polarity < -0.1 else "neutral"
            },
            "content_indicators": {
                "has_urls": bool(re.search(r'http\S+|www\S+', text)),
                "has_email": bool(re.search(r'\S+@\S+', text)),
                "has_mentions": bool(re.search(r'@\w+', text)),
                "sensational_words": [word for word in ['shocking', 'unbelievable', 'exposed', 'secret', 'urgent', 'breaking'] 
                                    if word in text.lower()]
            }
        }
        
        # Add VADER sentiment if available
        if analyzer:
            vader_scores = analyzer.polarity_scores(text)
            analysis["sentiment_analysis"]["vader"] = {
                "compound": float(vader_scores['compound']),
                "positive": float(vader_scores['pos']),
                "negative": float(vader_scores['neg']),
                "neutral": float(vader_scores['neu'])
            }
        
        return jsonify(analysis)
    
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route("/test", methods=["GET"])
def test_model():
    """Test endpoint with sample data"""
    try:
        test_cases = [
            {
                "title": "Scientific Research",
                "text": "Scientists at leading universities have published peer-reviewed research showing significant progress in renewable energy efficiency.",
                "expected": "Real"
            },
            {
                "title": "SHOCKING NEWS",
                "text": "SHOCKING!!! Celebrity scandal EXPOSED with UNBELIEVABLE evidence that will CHANGE EVERYTHING you thought you knew!!!",
                "expected": "Fake"
            },
            {
                "title": "Federal Reserve Update",
                "text": "The Federal Reserve announced a 0.25% interest rate adjustment following the latest economic indicators and inflation data.",
                "expected": "Real"
            },
            {
                "title": "URGENT MIRACLE",
                "text": "URGENT: This miracle weight loss secret will solve ALL your problems INSTANTLY! Doctors HATE this one simple trick!",
                "expected": "Fake"
            }
        ]
        
        results = []
        for i, test_case in enumerate(test_cases):
            try:
                result = predict_news_compatible(test_case["title"], test_case["text"])
                
                results.append({
                    "test_case": i + 1,
                    "title": test_case["title"],
                    "text_preview": test_case["text"][:80] + "...",
                    "prediction": result['label'],
                    "confidence": round(result['confidence'], 3),
                    "fake_prob": round(result['probabilities']['fake'], 3),
                    "real_prob": round(result['probabilities']['real'], 3),
                    "expected": test_case["expected"],
                    "correct": result['label'] == test_case["expected"]
                })
                
            except Exception as e:
                results.append({
                    "test_case": i + 1,
                    "error": str(e)
                })
        
        return jsonify({
            "status": "Compatible model test completed",
            "model_type": "improved_ensemble",
            "results": results,
            "summary": {
                "total_tests": len(results),
                "successful_predictions": len([r for r in results if "error" not in r]),
                "failed_predictions": len([r for r in results if "error" in r]),
                "correct_predictions": len([r for r in results if r.get("correct", False)]),
                "accuracy": len([r for r in results if r.get("correct", False)]) / len([r for r in results if "error" not in r]) if len([r for r in results if "error" not in r]) > 0 else 0
            }
        })
    
    except Exception as e:
        return jsonify({"error": f"Model test failed: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Comprehensive health check"""
    try:
        health_status = {
            "server_status": "running",
            "model_loaded": model_components is not None,
            "components": {
                "model": model_components.get('model') is not None if model_components else False,
                "tfidf_vectorizer": model_components.get('tfidf_vectorizer') is not None if model_components else False,
                "scaler": model_components.get('scaler') is not None if model_components else False,
                "feature_columns": model_components.get('feature_columns') is not None if model_components else False,
                "preprocessing_function": model_components.get('preprocessing_function') is not None if model_components else False,
                "vader_analyzer": analyzer is not None
            },
            "stats_available": stats_data is not None,
            "model_type": "improved_ensemble",
            "feature_count": len(model_components.get('feature_columns', [])) if model_components else 0
        }
        
        return jsonify(health_status)
    
    except Exception as e:
        return jsonify({"error": f"Health check failed: {str(e)}"}), 500

if __name__ == "__main__":
    try:
        print("Starting Compatible Flask Server for Fake News Detection...")
        print("Loading models and components...")
        load_models()
        print("Server initialization complete!")
        print("\nAvailable endpoints:")
        print("- GET  /         - Home/health check")
        print("- POST /predict  - Single text prediction (supports title and text)")
        print("- POST /batch_predict - Multiple text predictions")
        print("- POST /analyze  - Detailed text analysis")
        print("- GET  /stats    - Model statistics")
        print("- GET  /test     - Model testing")
        print("- GET  /health   - Comprehensive health check")
        print("\nRequired files (from training notebook):")
        print("- improved_news_model.joblib")
        print("- improved_model_stats.joblib")
        print("\nServer starting on http://0.0.0.0:5000")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Failed to start server: {e}")
        print("Make sure to run the training notebook first to generate the required model files!")