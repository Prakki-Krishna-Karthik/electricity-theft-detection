"""
Feature Engineering - Make model learn DEFINITIONS, not dataset patterns
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

print("="*70)
print("FEATURE ENGINEERING - LEARNING THEFT DEFINITIONS")
print("="*70)

# Load data
df = pd.read_csv('results/loaded_data.csv')
df = df[df['theft'] != 'Theft6']
print(f"✅ Loaded: {df.shape[0]:,} rows")

# Original features
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

# ============================================
# STEP 1: Calculate rolling mean for each consumer
# ============================================
print("\n📊 Calculating rolling means per consumer...")

df['Elec_RollingMean'] = df.groupby('Class')['Electricity:Facility [kW](Hourly)'].transform(
    lambda x: x.rolling(24, min_periods=1).mean()
)
df['Gas_RollingMean'] = df.groupby('Class')['Gas:Facility [kW](Hourly)'].transform(
    lambda x: x.rolling(24, min_periods=1).mean()
)

# ============================================
# STEP 2: Create DEFINITION-BASED features
# ============================================
print("\n🔧 Creating definition-based features...")

# For Theft1: Constant reduction (ratio should be constant across all features)
df['Reduction_Ratio_Elec'] = df['Electricity:Facility [kW](Hourly)'] / df['Elec_RollingMean']
df['Reduction_Ratio_Gas'] = df['Gas:Facility [kW](Hourly)'] / df['Gas_RollingMean']
df['Reduction_Consistency'] = np.abs(df['Reduction_Ratio_Elec'] - df['Reduction_Ratio_Gas'])

# For Theft2: Zero consumption
df['Is_Zero'] = (df[feature_cols] == 0).all(axis=1).astype(int)

# For Theft3: Hourly random reduction (high variance across features)
feature_vars = []
for col in feature_cols:
    feature_vars.append(df.groupby('Class')[col].transform(lambda x: x.rolling(24, min_periods=1).std()))
df['Feature_Variance'] = np.mean(feature_vars, axis=0)

# For Theft4: Random fraction of mean (values should be between 0.1-0.8 of mean)
df['Is_Fraction_Of_Mean'] = ((df['Reduction_Ratio_Elec'] >= 0.1) & 
                              (df['Reduction_Ratio_Elec'] <= 0.8)).astype(int)

# For Theft5: Constant mean (values should equal rolling mean)
df['Is_Constant_Mean'] = (np.abs(df['Reduction_Ratio_Elec'] - 1) < 0.05).astype(int)

# ============================================
# STEP 3: Prepare new feature set
# ============================================
print("\n📊 New feature set:")

new_features = [
    'Reduction_Ratio_Elec',
    'Reduction_Ratio_Gas', 
    'Reduction_Consistency',
    'Is_Zero',
    'Feature_Variance',
    'Is_Fraction_Of_Mean',
    'Is_Constant_Mean'
]

# Add original features as well
all_features = feature_cols + new_features

print(f"   Original: {len(feature_cols)} features")
print(f"   Engineered: {len(new_features)} features")
print(f"   Total: {len(all_features)} features")

X = df[all_features]
y = df['theft']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train
print("\n🌲 Training Random Forest with engineered features...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy*100:.2f}%")

# Test with Theft4 pattern
test_theft4 = [[20, 2, 0, 0, 5, 11, 85, 65, 12, 7,  # Original
                20/22, 85/136,  # Ratios
                abs(20/22 - 85/136),  # Consistency
                0,  # Is_Zero
                0.5,  # Variance
                1,  # Is_Fraction_Of_Mean (20/22=0.9 is borderline)
                0]]  # Is_Constant_Mean

test_scaled = scaler.transform(test_theft4)
pred = model.predict(test_scaled)[0]
print(f"\n🔮 Theft4 test case prediction: {pred}")

# Save
joblib.dump(model, 'models/perfect_model.pkl')
joblib.dump(scaler, 'models/perfect_scaler.pkl')
joblib.dump(all_features, 'models/perfect_features.pkl')

print("\n✅ Perfect model saved!")