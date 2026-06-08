# debug_prediction.py
import pandas as pd
import numpy as np
import joblib

# Load model and files
model = joblib.load('models/rf_theft_detector.pkl')
scaler = joblib.load('models/scaler.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')
feature_names = joblib.load('models/feature_names.pkl')

# Load original dataset
df = pd.read_csv('results/loaded_data.csv')

# Get actual normal values from dataset
normal_data = df[df['theft'] == 'Normal']
normal_sample = normal_data[feature_names].iloc[0]

print("="*60)
print("DIAGNOSTIC: Checking Model Behavior")
print("="*60)

# Test with actual normal value from dataset
print("\n1. Testing with ACTUAL normal value from dataset:")
print(f"   Values: {normal_sample.values}")
normal_scaled = scaler.transform([normal_sample.values])
pred = model.predict(normal_scaled)
pred_class = label_encoder.inverse_transform(pred)[0]
prob = model.predict_proba(normal_scaled)
confidence = np.max(prob) * 100

print(f"   Prediction: {pred_class}")
print(f"   Confidence: {confidence:.2f}%")

if pred_class != 'Normal':
    print("\n   ⚠️ ISSUE DETECTED! Model misclassifying real normal data.")
    print("   The model may have been trained incorrectly or scaling issue.")
else:
    print("\n   ✅ Model correctly identifies real normal data.")

# Check the values you entered
print("\n" + "="*60)
print("2. Your test values vs Real Normal values:")
print("="*60)

your_values = [95.0, 28.0, 50.0, 32.0, 18.0, 48.0, 60.0, 38.0, 22.0, 25.0]
real_normal = normal_sample.values

comparison = pd.DataFrame({
    'Feature': feature_names,
    'Your Values': your_values,
    'Real Normal': real_normal,
    'Difference': [y - r for y, r in zip(your_values, real_normal)]
})
print(comparison.to_string())

print("\n" + "="*60)
print("3. Checking class distribution in training:")
print("="*60)
print(df['theft'].value_counts())

# Test with multiple normal samples
print("\n" + "="*60)
print("4. Testing 10 random normal samples from dataset:")
print("="*60)  # Fixed: removed the extra = sign

normal_samples = normal_data[feature_names].sample(10)
correct = 0
for idx, row in normal_samples.iterrows():
    scaled = scaler.transform([row.values])
    pred = model.predict(scaled)[0]
    pred_class = label_encoder.inverse_transform([pred])[0]
    is_correct = pred_class == 'Normal'
    if is_correct:
        correct += 1
    print(f"   Sample {idx}: {pred_class} - {'✅' if is_correct else '❌'}")

print(f"\n   Accuracy on normal samples: {correct}/10 = {correct*10}%")