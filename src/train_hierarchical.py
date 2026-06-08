"""
Hierarchical Model: Binary detection first, then multi-class
This fixes Theft4/Theft5 detection
"""
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("="*70)
print("HIERARCHICAL MODEL TRAINING")
print("="*70)

# Load data
df = pd.read_csv('results/loaded_data.csv')
print(f"✅ Loaded: {df.shape[0]:,} rows")

# Remove Theft6
df = df[df['theft'] != 'Theft6']
print(f"✅ After removing Theft6: {df.shape[0]:,} rows")

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

# Scale
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ============================================
# STAGE 1: Binary Classifier (Normal vs Theft)
# ============================================
print("\n" + "="*70)
print("STAGE 1: Binary Classifier (Normal vs Any Theft)")
print("="*70)

y_binary = (y != 'Normal').astype(int)

# Split
X_train, X_test, y_train_bin, y_test_bin = train_test_split(
    X_scaled, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

print(f"Training: {X_train.shape[0]:,} samples")
print(f"Test: {X_test.shape[0]:,} samples")

# Train binary classifier
binary_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=1)
binary_model.fit(X_train, y_train_bin)

# Evaluate
y_pred_bin = binary_model.predict(X_test)
binary_accuracy = accuracy_score(y_test_bin, y_pred_bin)
print(f"\n✅ Binary Accuracy: {binary_accuracy*100:.2f}%")

# ============================================
# STAGE 2: Multi-class (Theft types only)
# ============================================
print("\n" + "="*70)
print("STAGE 2: Multi-class Classifier (Theft1-5 Only)")
print("="*70)

# Filter only theft samples
df_theft = df[df['theft'] != 'Normal']
X_theft = df_theft[feature_cols]
y_theft = df_theft['theft']

# Encode
le = LabelEncoder()
y_theft_encoded = le.fit_transform(y_theft)

# Scale theft data
X_theft_scaled = scaler.transform(X_theft)

print(f"\n📊 Theft class distribution before SMOTE:")
for cls, count in y_theft.value_counts().items():
    pct = (count / len(y_theft)) * 100
    print(f"   {cls}: {count:,} ({pct:.1f}%)")

# Split
X_theft_train, X_theft_test, y_theft_train, y_theft_test = train_test_split(
    X_theft_scaled, y_theft_encoded, test_size=0.2, random_state=42, stratify=y_theft_encoded
)

# Apply SMOTE
print("\n🔄 Applying SMOTE to theft classes...")
smote = SMOTE(random_state=42, k_neighbors=3)
X_theft_balanced, y_theft_balanced = smote.fit_resample(X_theft_train, y_theft_train)

print(f"✅ After SMOTE: {X_theft_balanced.shape[0]:,} samples")

# Show balanced distribution
unique, counts = np.unique(y_theft_balanced, return_counts=True)
print("\n📊 Balanced theft class distribution:")
for idx, count in zip(unique, counts):
    print(f"   {le.inverse_transform([idx])[0]}: {count:,}")

# Train theft classifier
print("\n🌲 Training Theft Type Classifier...")
theft_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced',
    verbose=1
)
theft_model.fit(X_theft_balanced, y_theft_balanced)

# Evaluate
y_theft_pred = theft_model.predict(X_theft_test)
theft_accuracy = accuracy_score(y_theft_test, y_theft_pred)
print(f"\n✅ Theft-only Accuracy: {theft_accuracy*100:.2f}%")

print("\n📊 Theft Classification Report:")
print(classification_report(y_theft_test, y_theft_pred, target_names=le.classes_))

# ============================================
# SAVE MODELS
# ============================================
print("\n" + "="*70)
print("SAVING MODELS")
print("="*70)

os.makedirs('models', exist_ok=True)

joblib.dump(binary_model, 'models/binary_model.pkl')
joblib.dump(theft_model, 'models/theft_model.pkl')
joblib.dump(scaler, 'models/hierarchical_scaler.pkl')
joblib.dump(le, 'models/theft_label_encoder.pkl')
joblib.dump(feature_cols, 'models/feature_names.pkl')

print("\n✅ Saved:")
print("   - models/binary_model.pkl (Normal vs Theft)")
print("   - models/theft_model.pkl (Theft1-5)")
print("   - models/hierarchical_scaler.pkl")
print("   - models/theft_label_encoder.pkl")

# ============================================
# TEST THE HIERARCHICAL MODEL
# ============================================
print("\n" + "="*70)
print("TESTING HIERARCHICAL MODEL")
print("="*70)

test_cases = {
    "Normal (real values)": [22.04, 3.59, 0, 0, 4.59, 8.19, 136.59, 124.00, 3.34, 9.25],
    "Theft1 (50% reduction)": [11.02, 1.80, 0, 0, 2.30, 4.10, 68.30, 62.00, 1.67, 4.63],
    "Theft2 (All zeros)": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Theft5 (Constant 30)": [30, 30, 30, 30, 30, 30, 30, 30, 30, 30],
    "Theft4 (Mean pattern)": [35, 10, 12, 8, 9, 20, 40, 35, 10, 12],
    "Mixed (45 elec, 35 gas)": [45, 12, 8, 5, 10, 25, 35, 20, 8, 10],
}

print("\n📋 Test Results:")
print("-"*50)

for name, values in test_cases.items():
    scaled = scaler.transform([values])
    
    # Stage 1: Is it theft?
    is_theft = binary_model.predict(scaled)[0]
    
    if is_theft == 0:
        prediction = "Normal"
        confidence = "N/A"
        status = "🟢"
    else:
        # Stage 2: Which theft type?
        theft_pred = theft_model.predict(scaled)[0]
        probs = theft_model.predict_proba(scaled)[0]
        confidence = max(probs) * 100
        prediction = le.inverse_transform([theft_pred])[0]
        status = "🔴"
    
    print(f"{status} {name}: {prediction} (Confidence: {confidence if prediction != 'Normal' else 'N/A'})")

print("\n" + "="*70)
print("✅ HIERARCHICAL MODEL READY!")
print("="*70)
print("\n📌 This model:")
print("   - First detects if theft is happening (Binary)")
print("   - Then identifies which theft type (Multi-class)")
print("   - Should fix Theft4/Theft5 detection issues")