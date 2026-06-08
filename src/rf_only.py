"""
Random Forest Only - Best Model from Paper
7-Class and 6-Class Results in < 3 minutes
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
import time
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("RANDOM FOREST - BEST MODEL FOR THEFT DETECTION")
print("="*60)

# Load data
print("\n📂 Loading dataset...")
df = pd.read_csv('results/loaded_data.csv')
print(f"✅ Loaded: {df.shape[0]:,} rows")

# Prepare features
X = df.drop(columns=['theft', 'Class', '0'])
y = df['theft']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"\n📋 Classes found: {list(le.classes_)}")

# Normalize
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\n✂️ Training: {X_train.shape[0]:,} samples")
print(f"   Test: {X_test.shape[0]:,} samples")

# ============================================
# 7-CLASS RESULTS (All theft types)
# ============================================
print("\n" + "="*60)
print("7-CLASS RESULTS (All 6 theft types + Normal)")
print("="*60)

print("\n🌲 Training Random Forest on 7 classes...")
start = time.time()

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=1)
rf.fit(X_train, y_train)

train_time = time.time() - start
print(f"\n✅ Training complete in {train_time:.2f} seconds")

# Predict
y_pred = rf.predict(X_test)

# Calculate metrics
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
kappa = cohen_kappa_score(y_test, y_pred)

try:
    y_prob = rf.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')
except:
    auc = 0

print(f"\n📊 RESULTS (7-Class):")
print(f"   Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"   F1-Score: {f1:.4f}")
print(f"   Kappa: {kappa:.4f}")
print(f"   AUC: {auc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n📊 Confusion Matrix Shape: {cm.shape}")

# ============================================
# 6-CLASS RESULTS (Remove Theft6 - as per paper)
# ============================================
print("\n" + "="*60)
print("6-CLASS RESULTS (Removing Theft6 - Better Performance)")
print("="*60)

# Remove Theft6
df_6class = df[df['theft'] != 'Theft6']
print(f"\n📂 After removing Theft6: {df_6class.shape[0]:,} rows")

# Prepare features
X_6 = df_6class.drop(columns=['theft', 'Class', '0'])
y_6 = df_6class['theft']

# Encode
le6 = LabelEncoder()
y6_encoded = le6.fit_transform(y_6)
print(f"\n📋 Classes (6 classes): {list(le6.classes_)}")

# Normalize
X6_scaled = scaler.fit_transform(X_6)

# Split
X6_train, X6_test, y6_train, y6_test = train_test_split(
    X6_scaled, y6_encoded, test_size=0.2, random_state=42, stratify=y6_encoded
)

print("\n🌲 Training Random Forest on 6 classes...")
start = time.time()

rf6 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=1)
rf6.fit(X6_train, y6_train)

train_time_6 = time.time() - start
print(f"\n✅ Training complete in {train_time_6:.2f} seconds")

# Predict
y6_pred = rf6.predict(X6_test)

# Calculate metrics
acc6 = accuracy_score(y6_test, y6_pred)
f16 = f1_score(y6_test, y6_pred, average='weighted')
kappa6 = cohen_kappa_score(y6_test, y6_pred)

try:
    y6_prob = rf6.predict_proba(X6_test)
    auc6 = roc_auc_score(y6_test, y6_prob, multi_class='ovr', average='weighted')
except:
    auc6 = 0

print(f"\n📊 RESULTS (6-Class):")
print(f"   Accuracy: {acc6:.4f} ({acc6*100:.2f}%)")
print(f"   F1-Score: {f16:.4f}")
print(f"   Kappa: {kappa6:.4f}")
print(f"   AUC: {auc6:.4f}")

# Confusion matrix
cm6 = confusion_matrix(y6_test, y6_pred)
print(f"\n📊 Confusion Matrix Shape: {cm6.shape}")

# ============================================
# COMPARISON TABLE
# ============================================
print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)

comparison = pd.DataFrame({
    'Version': ['7-Class', '6-Class'],
    'Accuracy': [acc, acc6],
    'F1-Score': [f1, f16],
    'Kappa': [kappa, kappa6],
    'AUC': [auc, auc6]
})
comparison['Accuracy (%)'] = comparison['Accuracy'] * 100
print(comparison.to_string(index=False))

print(f"\n📈 IMPROVEMENT: {(acc6 - acc)*100:.2f}% increase by removing Theft6")

# ============================================
# SAVE RESULTS
# ============================================
print("\n💾 Saving results...")

os.makedirs('results', exist_ok=True)

# Save metrics
comparison.to_csv('results/final_comparison.csv', index=False)
print("✅ Saved: results/final_comparison.csv")

# Save confusion matrices
np.savetxt('results/cm_7class.csv', cm, delimiter=',', fmt='%d')
np.savetxt('results/cm_6class.csv', cm6, delimiter=',', fmt='%d')
print("✅ Saved: results/cm_7class.csv and results/cm_6class.csv")

# Save class names
with open('results/classes_7class.txt', 'w') as f:
    for name in le.classes_:
        f.write(f"{name}\n")

with open('results/classes_6class.txt', 'w') as f:
    for name in le6.classes_:
        f.write(f"{name}\n")
print("✅ Saved: class names")

print("\n" + "="*60)
print("✅ COMPLETE!")
print("="*60)
print(f"\n🏆 BEST RESULT FOR YOUR PAPER:")
print(f"   6-Class Random Forest Accuracy: {acc6*100:.2f}%")
print(f"   (Paper achieved 94.71% - You achieved {acc6*100:.2f}%)")
print("\n📁 All results saved in 'results' folder")