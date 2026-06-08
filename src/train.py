"""
Complete ML Training for Electricity Theft Detection
Based on the paper: KNN, DT, RF, Bagging, ANN
"""

import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("ML MODEL TRAINING FOR ELECTRICITY THEFT DETECTION")
print("="*60)

# Load the saved data
print("\n📂 Loading dataset...")
df = pd.read_csv('results/loaded_data.csv')
print(f"✅ Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# Show columns
print(f"\n📋 Columns: {list(df.columns)}")

# The target column for theft detection is 'theft' (shows Normal or Theft type)
# The 'Class' column is consumer type (16 types)

target_col = 'theft'  # This is what we want to predict
consumer_col = 'Class'  # Consumer type (16 categories)

print(f"\n🎯 Target column: '{target_col}'")
print(f"   Unique values: {df[target_col].unique()}")

# Check distribution
print(f"\n📊 Target distribution:")
print(df[target_col].value_counts())

# Prepare features - drop the first unnamed column (0) and target columns
X = df.drop(columns=[target_col, consumer_col, '0'])  # Features: energy consumption data
y = df[target_col]  # Target: Normal vs Theft types

print(f"\n📊 Feature columns ({len(X.columns)} features):")
for i, col in enumerate(X.columns, 1):
    print(f"   {i}. {col}")

# Encode target labels (convert text to numbers)
print("\n🔧 Encoding target labels...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"   Classes: {dict(zip(le.classes_, range(len(le.classes_))))}")

# Normalize features (as per paper - MinMaxScaler between 0 and 1)
print("\n📊 Normalizing features (MinMaxScaler 0-1)...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Train-test split (80-20 as per paper)
print("\n✂️ Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"   Training: {X_train.shape[0]:,} samples")
print(f"   Test: {X_test.shape[0]:,} samples")

# Initialize models (parameters as per paper)
models = {
    'KNN': KNeighborsClassifier(n_neighbors=10),
    'DT': DecisionTreeClassifier(random_state=42),
    'RF': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Bagging': BaggingClassifier(n_estimators=10, random_state=42, n_jobs=-1),
    'ANN': MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
}

results = {}

print("\n" + "="*60)
print("TRAINING MODELS...")
print("="*60)

for name, model in models.items():
    print(f"\n📊 Training {name}...")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_test, y_pred)
    
    # Calculate AUC (for multi-class)
    try:
        y_prob = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
    except:
        auc = 0
    
    # 5-fold cross validation (as per paper)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    results[name] = {
        'Accuracy': accuracy,
        'F1-Score': f1,
        'Kappa': kappa,
        'AUC': auc,
        'CV_Mean': cv_scores.mean(),
        'CV_Std': cv_scores.std()
    }
    
    print(f"   ✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   Kappa: {kappa:.4f}")
    print(f"   AUC: {auc:.4f}")
    print(f"   5-Fold CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Create results table
print("\n" + "="*60)
print("FINAL RESULTS COMPARISON")
print("="*60)

results_df = pd.DataFrame(results).T
results_df = results_df.round(4)
print(results_df.to_string())

# Save results
os.makedirs('results', exist_ok=True)
results_df.to_csv('results/model_comparison.csv')
print("\n💾 Saved: results/model_comparison.csv")

# Find best model
best_model = results_df['Accuracy'].idxmax()
print(f"\n🏆 BEST MODEL: {best_model}")
print(f"   Accuracy: {results_df.loc[best_model, 'Accuracy']*100:.2f}%")

# Confusion matrix for best model
print(f"\n📊 Generating confusion matrix for {best_model}...")
best_model_obj = models[best_model]
y_pred_best = best_model_obj.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

# Get class names
class_names = le.classes_
print(f"\n Confusion Matrix ({best_model}):")
print("="*60)
print("Rows: Actual, Columns: Predicted")
print(f"Classes: {list(class_names)}")
print("-"*60)
print(cm)
print("="*60)

# Save confusion matrix
np.savetxt('results/confusion_matrix.csv', cm, delimiter=',', fmt='%d')
print("💾 Saved: results/confusion_matrix.csv")

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)
print("\n📁 Results saved in 'results' folder:")
print("   - model_comparison.csv (all model metrics)")
print("   - confusion_matrix.csv (best model)")