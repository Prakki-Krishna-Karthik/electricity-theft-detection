"""
Save trained model - WITH THEFT6 REMOVED
This matches the paper's recommendation for 94%+ accuracy
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

print("="*60)
print("SAVING TRAINED MODEL (Theft6 Removed)")
print("="*60)

# Load data
print("\n📂 Loading dataset...")
df = pd.read_csv('results/loaded_data.csv')
print(f"✅ Loaded: {df.shape[0]:,} rows")

# ============================================
# KEY FIX: REMOVE THEFT6 (as per paper)
# ============================================
print("\n🔧 Removing Theft6 (paper recommendation for better accuracy)...")
print(f"   Before removal: {df['theft'].nunique()} classes")
df = df[df['theft'] != 'Theft6']
print(f"   After removal: {df['theft'].nunique()} classes")
print(f"   New size: {df.shape[0]:,} rows")

# Show class distribution after removal
print("\n📊 Class distribution after removing Theft6:")
for cls, count in df['theft'].value_counts().items():
    pct = (count / len(df)) * 100
    print(f"   {cls}: {count:,} ({pct:.1f}%)")

# Prepare features
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

# Normalize
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\n✂️ Training: {X_train.shape[0]:,} samples")
print(f"   Test: {X_test.shape[0]:,} samples")

# Train Random Forest
print("\n🌲 Training Random Forest (6-Class)...")
rf_model = RandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    n_jobs=-1,
    class_weight='balanced'
)
rf_model.fit(X_train, y_train)

# Evaluate
y_pred = rf_model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy*100:.2f}%")

# Save model
joblib.dump(rf_model, 'models/rf_theft_detector.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le, 'models/label_encoder.pkl')
joblib.dump(feature_cols, 'models/feature_names.pkl')

print("\n✅ Saved:")
print("   - models/rf_theft_detector.pkl")
print("   - models/scaler.pkl")
print("   - models/label_encoder.pkl")
print("   - models/feature_names.pkl")

print("\n" + "="*60)
print("✅ MODEL SAVED SUCCESSFULLY!")
print(f"   Accuracy: {accuracy*100:.2f}%")
print("   Classes: 6 (Theft6 removed)")
print("="*60)