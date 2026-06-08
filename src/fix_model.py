"""
Fix: Train model that doesn't default to Theft3
Run this ONCE to create the fixed model
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("="*70)
print("TRAINING FIXED MODEL - NO DEFAULT BIAS")
print("="*70)

# Get CSV file path - either from command line or use default
csv_path = None

if len(sys.argv) > 1:
    csv_path = sys.argv[1]
    print(f"📂 Loading from command line argument: {csv_path}")
else:
    # Try possible paths
    possible_paths = ['results/loaded_data.csv', 'loaded_data.csv', '../results/loaded_data.csv', 'temp_dataset.csv']
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            print(f"📂 Found dataset at: {path}")
            break

if csv_path is None or not os.path.exists(csv_path):
    print("❌ Dataset not found!")
    print("   Please ensure loaded_data.csv is in the current directory")
    sys.exit(1)

# Load data
try:
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
except Exception as e:
    print(f"❌ Error loading CSV: {e}")
    sys.exit(1)

# Remove Theft6
df = df[df['theft'] != 'Theft6']
print(f"✅ After removing Theft6: {df.shape[0]:,} rows")

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
print(f"\n📋 Classes: {list(le.classes_)}")

# Scale
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\n✂️ Training samples: {X_train.shape[0]:,}")
print(f"   Test samples: {X_test.shape[0]:,}")

# Custom class weights - Give Theft3 LESS weight so it's not default
class_counts = pd.Series(y_train).value_counts()
class_weights = {}

for class_idx, count in class_counts.items():
    class_name = le.inverse_transform([class_idx])[0]
    if class_name == 'Theft3':
        class_weights[class_idx] = 0.3  # Lower weight for Theft3
    elif class_name == 'Normal':
        class_weights[class_idx] = 0.2  # Lower weight for Normal
    else:
        class_weights[class_idx] = 2.0  # Higher weight for other theft types

print("\n📊 Class weights applied:")
for class_idx, weight in class_weights.items():
    print(f"   {le.inverse_transform([class_idx])[0]}: {weight}")

# Train
print("\n🌲 Training Random Forest (this may take 2-3 minutes)...")
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight=class_weights,
    verbose=0
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy*100:.2f}%")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Test cases
print("\n" + "="*70)
print("TESTING FIXED MODEL")
print("="*70)

test_cases = {
    "Theft1 (50% reduction)": [11.02, 1.80, 0, 0, 2.30, 4.10, 68.30, 62.00, 1.67, 4.63],
    "Theft4 (Real sample)": [18.77, 1.64, 0, 0, 4.14, 10.46, 89.15, 68.65, 13.98, 6.52],
    "Theft5 (Real sample)": [34.46, 2.99, 0, 0, 7.52, 19.00, 155.54, 118.29, 25.40, 11.85],
}

for name, values in test_cases.items():
    scaled = scaler.transform([values])
    pred = model.predict(scaled)[0]
    pred_class = le.inverse_transform([pred])[0]
    probs = model.predict_proba(scaled)[0]
    confidence = max(probs) * 100
    print(f"\n{name}: {pred_class} ({confidence:.1f}%)")

# Save models
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/rf_fixed_model.pkl')
joblib.dump(scaler, 'models/scaler_fixed.pkl')
joblib.dump(le, 'models/label_encoder_fixed.pkl')
joblib.dump(feature_cols, 'models/feature_names_fixed.pkl')

print("\n" + "="*70)
print("✅ FIXED MODEL SAVED SUCCESSFULLY!")
print("   models/rf_fixed_model.pkl")
print("="*70)
