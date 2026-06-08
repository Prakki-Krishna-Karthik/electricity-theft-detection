"""
Train Balanced Model with SMOTE - REMOVING THEFT6
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("TRAINING BALANCED MODEL WITH SMOTE (Theft6 Removed)")
print("="*70)

# Load data
df = pd.read_csv('results/loaded_data.csv')
print(f"✅ Loaded: {df.shape[0]:,} rows")

# ============================================
# REMOVE THEFT6
# ============================================
print("\n🔧 Removing Theft6...")
df = df[df['theft'] != 'Theft6']
print(f"   New size: {df.shape[0]:,} rows")

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

print("\n📊 Class distribution after removing Theft6:")
for cls, count in y.value_counts().items():
    pct = (count / len(y)) * 100
    print(f"   {cls}: {count:,} ({pct:.1f}%)")

# Encode
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Normalize
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Apply SMOTE
print("\n🔄 Applying SMOTE...")
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"   After SMOTE: {X_train_balanced.shape[0]:,} samples")

# Train
print("\n🌲 Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
model.fit(X_train_balanced, y_train_balanced)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Model Accuracy: {accuracy*100:.2f}%")

# Test custom case
test_case = np.array([[45.00, 12.00, 8.00, 5.00, 10.00, 25.00, 35.00, 20.00, 8.00, 10.00]])
test_scaled = scaler.transform(test_case)
pred = model.predict(test_scaled)[0]
pred_class = le.inverse_transform([pred])[0]
probs = model.predict_proba(test_scaled)[0]
confidence = max(probs) * 100

print(f"\n🔮 Test case prediction: {pred_class} (Confidence: {confidence:.1f}%)")

# Save
joblib.dump(model, 'models/rf_balanced_smote.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(le, 'models/label_encoder.pkl')
joblib.dump(feature_cols, 'models/feature_names.pkl')

print("\n✅ Model saved as: models/rf_balanced_smote.pkl")
print("="*70)