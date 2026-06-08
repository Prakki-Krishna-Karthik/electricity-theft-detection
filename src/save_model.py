"""
Save trained model - WITH DATASET DOWNLOAD
"""

import pandas as pd
import numpy as np
import joblib
import os
import urllib.request
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

print("="*60)
print("SAVING TRAINED MODEL (Theft6 Removed)")
print("="*60)

# ============================================
# DOWNLOAD DATASET IF NOT EXISTS
# ============================================
def download_dataset():
    """Download ETD2022 dataset from Mendeley"""
    url = "https://data.mendeley.com/public-files/datasets/c3c7329tj1/files/8f3c4c9e-6f5b-4f2c-9a1e-3b8c7d6e5f4a/file_downloaded"
    os.makedirs('data', exist_ok=True)
    file_path = 'data/etd2022.csv'
    
    if not os.path.exists(file_path):
        print("📥 Downloading dataset (this may take a few minutes)...")
        urllib.request.urlretrieve(url, file_path)
        print("✅ Download complete!")
    return file_path

# Load data
print("\n📂 Loading dataset...")
data_path = 'results/loaded_data.csv'

if not os.path.exists(data_path):
    # Try to download from original source
    data_path = download_dataset()
    df = pd.read_csv(data_path)
else:
    df = pd.read_csv(data_path)

print(f"✅ Loaded: {df.shape[0]:,} rows")

# Remove Theft6
print("\n🔧 Removing Theft6...")
df = df[df['theft'] != 'Theft6']
print(f"   New size: {df.shape[0]:,} rows")

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

# Encode
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Train
print("\n🌲 Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy*100:.2f}%")

# Save
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/rf_balanced_smote.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le, 'models/label_encoder.pkl')
joblib.dump(feature_cols, 'models/feature_names.pkl')

print("\n✅ Model saved to 'models/' folder")
print("="*60)
