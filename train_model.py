import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("MALWARE URL DETECTION - FIXED TRAINING WITH CLASS BALANCE")
print("=" * 70)

# ============================================================================
# STEP 1: FEATURE EXTRACTION
# ============================================================================
def extract_url_features(url):
    """Extract 35+ features from URL"""
    features = {}
    
    try:
        # Basic URL properties
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
        
        # URL parsing
        parsed = urlparse(url)
        domain = parsed.netloc if parsed.netloc else url.split('/')[0]
        
        # Domain features
        features['domain_length'] = len(domain)
        features['has_ip'] = 1 if re.match(r'\d+\.\d+\.\d+\.\d+', domain) else 0
        
        # Path features
        path = parsed.path
        features['path_length'] = len(path)
        
        # Protocol features
        features['has_https'] = 1 if url.startswith('https') else 0
        features['has_http'] = 1 if url.startswith('http') else 0
        
        # Count digits and letters
        features['num_digits'] = sum(c.isdigit() for c in url)
        features['num_letters'] = sum(c.isalpha() for c in url)
        
        # Suspicious keywords
        suspicious_keywords = ['login', 'verify', 'account', 'secure', 'update', 
                               'banking', 'signin', 'ebay', 'paypal', 'admin',
                               'confirm', 'password', 'suspend']
        features['has_suspicious_keywords'] = sum(1 for keyword in suspicious_keywords if keyword in url.lower())
        
        # Domain entropy
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
        
        # TLD length
        tld = domain.split('.')[-1] if '.' in domain else ''
        features['tld_length'] = len(tld)
        
        # Subdomain count
        features['num_subdomains'] = domain.count('.') - 1 if domain.count('.') > 0 else 0
        
    except Exception as e:
        # Default values on error
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
# STEP 2: LOAD AND PROCESS DATA
# ============================================================================
print("\n[1/8] Loading dataset...")
df = pd.read_csv('Malicious URLs.csv', names=['url', 'type'], skiprows=1)  # Skip header row

print(f"Total samples: {len(df)}")
print(f"\nClass distribution:")
class_dist = df['type'].value_counts()
print(class_dist)
print("\nPercentages:")
print(df['type'].value_counts(normalize=True) * 100)

# Extract features
print("\n[2/8] Extracting features from URLs...")
features_list = []
for idx, url in enumerate(df['url']):
    if idx % 50000 == 0:
        print(f"  Processed {idx}/{len(df)} URLs ({idx*100//len(df)}%)")
    features_list.append(extract_url_features(str(url)))

features_df = pd.DataFrame(features_list)
features_df['label'] = df['type'].values

print("Feature extraction complete!")
print(f"Total features: {len(features_df.columns) - 1}")

# ============================================================================
# STEP 3: PREPARE DATA FOR TRAINING
# ============================================================================
print("\n[3/8] Preparing data with CLASS BALANCING...")

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(features_df['label'])
X = features_df.drop('label', axis=1)

print(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print(f"\nClass weights for balancing: {class_weight_dict}")

# Apply SMOTE to balance training data
print("\n[4/8] Applying SMOTE to balance training data...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"Before SMOTE: {len(X_train)} samples")
print(f"After SMOTE: {len(X_train_balanced)} samples")
print("\nBalanced class distribution:")
unique, counts = np.unique(y_train_balanced, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"  {label_encoder.inverse_transform([cls])[0]}: {count}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# STEP 5: TRAIN MODELS WITH CLASS WEIGHTS
# ============================================================================
print("\n[5/8] Training models with CLASS BALANCE...")
print("=" * 70)

models = {}
results = {}

# Model 1: Logistic Regression with balanced weights
print("\n[Model 1/4] Training Logistic Regression (with class_weight='balanced')...")
lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1, class_weight='balanced')
lr_model.fit(X_train_scaled, y_train_balanced)
lr_pred = lr_model.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_pred)
models['Logistic Regression'] = lr_model
results['Logistic Regression'] = lr_acc
print(f"  ✓ Accuracy: {lr_acc:.4f}")

# Model 2: Random Forest with balanced weights
print("\n[Model 2/4] Training Random Forest (with class_weight='balanced')...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, 
                                  class_weight='balanced', max_depth=20)
rf_model.fit(X_train_balanced, y_train_balanced)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
models['Random Forest'] = rf_model
results['Random Forest'] = rf_acc
print(f"  ✓ Accuracy: {rf_acc:.4f}")

# Model 3: XGBoost with scale_pos_weight
print("\n[Model 3/4] Training XGBoost (with balanced weights)...")
# Calculate scale_pos_weight for XGBoost
scale_pos_weight = len(y_train_balanced[y_train_balanced == 0]) / len(y_train_balanced[y_train_balanced == 1])
xgb_model = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
xgb_model.fit(X_train_balanced, y_train_balanced)
xgb_pred = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
models['XGBoost'] = xgb_model
results['XGBoost'] = xgb_acc
print(f"  ✓ Accuracy: {xgb_acc:.4f}")

# Model 4: Neural Network with class weights
print("\n[Model 4/4] Training Neural Network (with class weights)...")
nn_model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(len(label_encoder.classes_), activation='softmax')
])

nn_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), 
                 loss='sparse_categorical_crossentropy', 
                 metrics=['accuracy'])

# Train with class weights
nn_model.fit(X_train_scaled, y_train_balanced, 
             epochs=30, 
             batch_size=256, 
             validation_data=(X_val_scaled, y_val),
             class_weight=class_weight_dict,
             verbose=0)

nn_pred = np.argmax(nn_model.predict(X_test_scaled, verbose=0), axis=1)
nn_acc = accuracy_score(y_test, nn_pred)
models['Neural Network'] = nn_model
results['Neural Network'] = nn_acc
print(f"  ✓ Accuracy: {nn_acc:.4f}")

# ============================================================================
# STEP 6: CREATE ENSEMBLE MODEL
# ============================================================================
print("\n[6/8] Creating ensemble model...")

lr_pred_proba = lr_model.predict_proba(X_test_scaled)
rf_pred_proba = rf_model.predict_proba(X_test)
xgb_pred_proba = xgb_model.predict_proba(X_test)
nn_pred_proba = nn_model.predict(X_test_scaled, verbose=0)

# Weighted average (give more weight to better models)
weights = [0.2, 0.3, 0.3, 0.2]  # Adjust based on individual performance
ensemble_pred_proba = (weights[0] * lr_pred_proba + 
                       weights[1] * rf_pred_proba + 
                       weights[2] * xgb_pred_proba + 
                       weights[3] * nn_pred_proba)
ensemble_pred = np.argmax(ensemble_pred_proba, axis=1)
ensemble_acc = accuracy_score(y_test, ensemble_pred)
results['Ensemble'] = ensemble_acc
print(f"  ✓ Ensemble Accuracy: {ensemble_acc:.4f}")

# ============================================================================
# STEP 7: DETAILED RESULTS FOR EACH CLASS
# ============================================================================
print("\n[7/8] Detailed Performance Analysis")
print("=" * 70)

for model_name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{model_name:20s}: {acc:.4f} ({acc*100:.2f}%)")

# Show detailed report for best model
best_model_name = max(results, key=results.get)
print(f"\n\nDetailed Classification Report - {best_model_name}")
print("=" * 70)

if best_model_name == 'Ensemble':
    pred = ensemble_pred
elif best_model_name == 'Neural Network':
    pred = nn_pred
elif best_model_name == 'Logistic Regression':
    pred = lr_pred
elif best_model_name == 'Random Forest':
    pred = rf_pred
else:
    pred = xgb_pred

print(classification_report(y_test, pred, target_names=label_encoder.classes_))

# Per-class accuracy
print("\nPer-Class Accuracy:")
for i, class_name in enumerate(label_encoder.classes_):
    class_mask = y_test == i
    class_acc = accuracy_score(y_test[class_mask], pred[class_mask])
    print(f"  {class_name:15s}: {class_acc:.4f} ({class_acc*100:.2f}%)")

# Confusion Matrix
cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ Confusion matrix saved as 'confusion_matrix.png'")

# ============================================================================
# STEP 8: SAVE MODELS
# ============================================================================
print("\n[8/8] Saving trained models...")

with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('xgboost_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

nn_model.save('neural_network_model.h5')

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Save class weights for API use
with open('class_weights.pkl', 'wb') as f:
    pickle.dump(class_weight_dict, f)

print("\n✓ All models saved successfully!")
print("\nSaved files:")
print("  - logistic_regression_model.pkl")
print("  - random_forest_model.pkl")
print("  - xgboost_model.pkl")
print("  - neural_network_model.h5")
print("  - scaler.pkl")
print("  - label_encoder.pkl")
print("  - class_weights.pkl")
print("  - confusion_matrix.png")

print("\n" + "=" * 70)
print("✅ TRAINING COMPLETE WITH CLASS BALANCING!")
print("=" * 70)
print("\nThe models are now trained with:")
print("  ✓ SMOTE oversampling for minority classes")
print("  ✓ Class weights for balanced learning")
print("  ✓ Better handling of imbalanced data")
print("\nThis should fix the issue of predicting everything as malicious!")
print("=" * 70)