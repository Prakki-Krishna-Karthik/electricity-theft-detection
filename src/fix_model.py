"""
Fix: Train model with tolerance for Normal variations
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
print("TRAINING FIXED MODEL - WITH NORMAL TOLERANCE")
print("="*70)

# Get CSV file path
csv_path = None
if len(sys.argv) > 1:
    csv_path = sys.argv[1]
    print(f"📂 Loading from: {csv_path}")
else:
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

# ============================================
# CREATE RANGE-BASED FEATURES
# ============================================
print("\n📊 Creating range-based features...")

# Calculate normal mean and std for each feature
normal_data = df[df['theft'] == 'Normal']
normal_means = normal_data[feature_cols].mean()
normal_stds = normal_data[feature_cols].std()

print("\n📈 Normal value ranges:")
for col in feature_cols:
    print(f"   {col}: {normal_means[col]:.2f} ± {normal_stds[col]:.2f}")

# Create deviation features
deviation_features = []
for col in feature_cols:
    deviation_col = f'{col}_deviation'
    X[deviation_col] = np.abs(X[col] - normal_means[col]) / (normal_stds[col] + 0.01)
    deviation_features.append(deviation_col)

# Add ratio features
X['Elec_Gas_Ratio'] = X['Electricity:Facility [kW](Hourly)'] / (X['Gas:Facility [kW](Hourly)'] + 0.01)
X['Gas_Elec_Ratio'] = X['Gas:Facility [kW](Hourly)'] / (X['Electricity:Facility [kW](Hourly)'] + 0.01)

# Combine all features
all_features = feature_cols + deviation_features + ['Elec_Gas_Ratio', 'Gas_Elec_Ratio']
X_final = X[all_features]

print(f"\n📊 Total features: {len(feature_cols)} original + {len(deviation_features)} deviation + 2 ratio = {len(all_features)}")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\n📋 Classes: {list(le.classes_)}")

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_final)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\n✂️ Training: {X_train.shape[0]:,} samples")
print(f"   Test: {X_test.shape[0]:,} samples")

# Class weights
class_counts = pd.Series(y_train).value_counts()
class_weights = {}

for class_idx, count in class_counts.items():
    class_name = le.inverse_transform([class_idx])[0]
    if class_name == 'Theft3':
        class_weights[class_idx] = 0.3
    elif class_name == 'Normal':
        class_weights[class_idx] = 0.5
    else:
        class_weights[class_idx] = 1.5

print("\n📊 Class weights:")
for class_idx, weight in class_weights.items():
    print(f"   {le.inverse_transform([class_idx])[0]}: {weight}")

# Train model
print("\n🌲 Training Random Forest...")
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

# ============================================
# TEST CASES
# ============================================
print("\n" + "="*70)
print("TESTING CASES")
print("="*70)

test_cases = {
    "Standard Normal": [22.04, 3.59, 0, 0, 4.59, 8.19, 136.59, 124.00, 3.34, 9.25],
    "Moderate Normal": [28.50, 4.20, 2.50, 1.80, 6.30, 12.50, 110.00, 95.00, 5.50, 11.00],
    "Theft1": [11.02, 1.80, 0, 0, 2.30, 4.10, 68.30, 62.00, 1.67, 4.63],
    "Theft2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}

for name, values in test_cases.items():
    # Calculate deviations
    deviations = []
    for i, col in enumerate(feature_cols):
        deviation = abs(values[i] - normal_means[col]) / (normal_stds[col] + 0.01)
        deviations.append(deviation)
    
    ratio1 = values[0] / (values[6] + 0.01)
    ratio2 = values[6] / (values[0] + 0.01)
    
    full = values + deviations + [ratio1, ratio2]
    scaled = scaler.transform([full])
    pred = model.predict(scaled)[0]
    pred_class = le.inverse_transform([pred])[0]
    probs = model.predict_proba(scaled)[0]
    confidence = max(probs) * 100
    
    print(f"\n{name}: {pred_class} ({confidence:.1f}%)")

# ============================================
# SAVE ALL MODELS
# ============================================
print("\n" + "="*70)
print("SAVING MODELS")
print("="*70)

os.makedirs('models', exist_ok=True)

joblib.dump(model, 'models/rf_fixed_model.pkl')
joblib.dump(scaler, 'models/scaler_fixed.pkl')
joblib.dump(le, 'models/label_encoder_fixed.pkl')
joblib.dump(all_features, 'models/feature_names_fixed.pkl')
joblib.dump(normal_means, 'models/normal_means.pkl')
joblib.dump(normal_stds, 'models/normal_stds.pkl')
joblib.dump(feature_cols, 'models/original_feature_names.pkl')

print("\n✅ Saved:")
print("   - models/rf_fixed_model.pkl")
print("   - models/scaler_fixed.pkl")
print("   - models/label_encoder_fixed.pkl")
print("   - models/feature_names_fixed.pkl")
print("   - models/normal_means.pkl")
print("   - models/normal_stds.pkl")
print("   - models/original_feature_names.pkl")

print("\n" + "="*70)
print("✅ FIXED MODEL WITH NORMAL TOLERANCE SAVED!")
print("="*70)
