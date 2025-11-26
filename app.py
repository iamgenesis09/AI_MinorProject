from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import re
from urllib.parse import urlparse
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# ============================================================================
# LOAD TRAINED MODELS
# ============================================================================
print("Loading trained models...")

# Load models
with open('logistic_regression_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

with open('random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('xgboost_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

nn_model = keras.models.load_model('neural_network_model.h5')

# Load scaler and label encoder
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

print("✓ All models loaded successfully!")

# ============================================================================
# FEATURE EXTRACTION FUNCTION
# ============================================================================
def extract_url_features(url):
    """Extract features from URL"""
    features = {}
    
    try:
        features['url_length'] = len(url)
        features['num_dots'] = url.count('.')
        features['num_hyphens'] = url.count('-')
        features['num_underscores'] = url.count('_')
        features['num_slashes'] = url.count('/')
        features['num_question'] = url.count('?')
        features['num_equal'] = url.count('=')
        features['num_at'] = url.count('@')
        features['num_ampersand'] = url.count('&')
        features['num_exclamation'] = url.count('!')
        features['num_space'] = url.count(' ')
        features['num_tilde'] = url.count('~')
        features['num_comma'] = url.count(',')
        features['num_plus'] = url.count('+')
        features['num_asterisk'] = url.count('*')
        features['num_hashtag'] = url.count('#')
        features['num_dollar'] = url.count('$')
        features['num_percent'] = url.count('%')
        
        parsed = urlparse(url)
        domain = parsed.netloc if parsed.netloc else url.split('/')[0]
        
        features['domain_length'] = len(domain)
        features['has_ip'] = 1 if re.match(r'\d+\.\d+\.\d+\.\d+', domain) else 0
        
        path = parsed.path
        features['path_length'] = len(path)
        
        features['has_https'] = 1 if url.startswith('https') else 0
        features['has_http'] = 1 if url.startswith('http') else 0
        
        features['num_digits'] = sum(c.isdigit() for c in url)
        features['num_letters'] = sum(c.isalpha() for c in url)
        
        suspicious_keywords = ['login', 'verify', 'account', 'secure', 'update', 
                               'banking', 'signin', 'ebay', 'paypal', 'admin',
                               'confirm', 'password', 'suspend']
        features['has_suspicious_keywords'] = sum(1 for keyword in suspicious_keywords if keyword in url.lower())
        
        def calculate_entropy(string):
            if not string:
                return 0
            entropy = 0
            for x in range(256):
                p_x = float(string.count(chr(x))) / len(string)
                if p_x > 0:
                    entropy += - p_x * np.log2(p_x)
            return entropy
        
        features['domain_entropy'] = calculate_entropy(domain)
        
        tld = domain.split('.')[-1] if '.' in domain else ''
        features['tld_length'] = len(tld)
        
        features['num_subdomains'] = domain.count('.') - 1 if domain.count('.') > 0 else 0
        
    except Exception as e:
        for key in ['url_length', 'num_dots', 'num_hyphens', 'num_underscores', 
                    'num_slashes', 'num_question', 'num_equal', 'num_at', 
                    'num_ampersand', 'num_exclamation', 'num_space', 'num_tilde',
                    'num_comma', 'num_plus', 'num_asterisk', 'num_hashtag',
                    'num_dollar', 'num_percent', 'domain_length', 'has_ip',
                    'path_length', 'has_https', 'has_http', 'num_digits',
                    'num_letters', 'has_suspicious_keywords', 'domain_entropy',
                    'tld_length', 'num_subdomains']:
            features[key] = 0
    
    return features

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
def predict_url(url):
    """Predict if URL is malicious using ensemble of models"""
    
    # Extract features
    features = extract_url_features(url)
    features_array = np.array(list(features.values())).reshape(1, -1)
    
    # Scale features for LR and NN
    features_scaled = scaler.transform(features_array)
    
    # Get predictions from all models
    lr_proba = lr_model.predict_proba(features_scaled)[0]
    rf_proba = rf_model.predict_proba(features_array)[0]
    xgb_proba = xgb_model.predict_proba(features_array)[0]
    nn_proba = nn_model.predict(features_scaled, verbose=0)[0]
    
    # Ensemble prediction (average)
    ensemble_proba = (lr_proba + rf_proba + xgb_proba + nn_proba) / 4
    prediction = np.argmax(ensemble_proba)
    confidence = ensemble_proba[prediction] * 100
    
    # Get label
    predicted_label = label_encoder.inverse_transform([prediction])[0]
    
    # Individual model predictions
    individual_predictions = {
        'Logistic Regression': label_encoder.inverse_transform([np.argmax(lr_proba)])[0],
        'Random Forest': label_encoder.inverse_transform([np.argmax(rf_proba)])[0],
        'XGBoost': label_encoder.inverse_transform([np.argmax(xgb_proba)])[0],
        'Neural Network': label_encoder.inverse_transform([np.argmax(nn_proba)])[0]
    }
    
    return {
        'url': url,
        'prediction': predicted_label,
        'confidence': float(confidence),
        'is_safe': predicted_label == 'benign',
        'individual_predictions': individual_predictions,
        'risk_level': 'LOW' if predicted_label == 'benign' else 'HIGH'
    }

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/')
def home():
    """Serve the frontend HTML"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for single URL prediction"""
    try:
        data = request.get_json()
        url = data.get('url', '')
        
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        
        result = predict_url(url)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """API endpoint for batch URL prediction"""
    try:
        data = request.get_json()
        urls = data.get('urls', [])
        
        if not urls:
            return jsonify({'error': 'No URLs provided'}), 400
        
        results = []
        for url in urls:
            try:
                result = predict_url(url.strip())
                results.append(result)
            except Exception as e:
                results.append({
                    'url': url,
                    'error': str(e)
                })
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': True,
        'available_labels': label_encoder.classes_.tolist()
    })

# ============================================================================
# RUN SERVER
# ============================================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("MALWARE URL DETECTION API SERVER")
    print("="*70)
    print("\nAPI Endpoints:")
    print("  - POST /api/predict         - Predict single URL")
    print("  - POST /api/batch-predict   - Predict multiple URLs")
    print("  - GET  /api/health          - Health check")
    print("\n" + "="*70)
    print("\nStarting server on http://localhost:5000")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)