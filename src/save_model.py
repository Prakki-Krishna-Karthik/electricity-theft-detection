"""
Save trained model - Accepts uploaded CSV file for training
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys

print("="*60)
print("TRAINING MODEL FROM UPLOADED DATASET")
print("="*60)

# Get CSV file path from command line argument
if len(sys.argv) < 2:
    print("❌ No CSV file provided!")
    print("Usage: python save_model.py <path_to_csv>")
    sys.exit(1)

csv_path = sys.argv[1]
print(f"\n📂 Loading dataset from: {csv_path}")

# Load data
df = pd.read_csv(csv_path)
print(f"✅ Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# Remove Theft6 (as per paper recommendation)
print("\n🔧 Removing Theft6...")
before = len(df)
df = df[df['theft'] != 'Theft6']
after = len(df)
print(f"   Removed: {before - after:,} rows")
print(f"   New size: {after:,} rows")

# Feature columns
feature_cols = ['Electricity:Facility [kW](Hourly)', 
                'Fans:Electricity [kW](Hourly)',
                'Cooling:Electricity [kW](Hourly)',
                'Heating:Electricity [kW](Hourly)',
                'InteriorLights:Electricity [kW](Hourly)',
                'InteriorEquipment:Electricity [kW](Hourly)',
                'Gas:Facility [kW](Hourly)',
                'Heating:Gas [kW](Hourly)',
                'InteriorEquipment:Gas [kW](Hourly)',
                'Water Heater:WaterSystems:Gas [kW](Hourly)']

X = df[feature_cols]
y = df['theft']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\n📋 Classes: {dict(zip(le.classes_, range(len(le.classes_))))}")

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\n✂️ Training samples: {X_train.shape[0]:,}")
print(f"   Test samples: {X_test.shape[0]:,}")

# Train Random Forest
print("\n🌲 Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy*100:.2f}%")

# Save models
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/rf_balanced_smote.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le, 'models/label_encoder.pkl')
joblib.dump(feature_cols, 'models/feature_names.pkl')

print("\n✅ Models saved to 'models/' folder:")
print("   - rf_balanced_smote.pkl")
print("   - scaler.pkl")
print("   - label_encoder.pkl")
print("   - feature_names.pkl")
print("="*60)
