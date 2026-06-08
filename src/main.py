"""
Electricity Theft Detection - SRIP 2026
VIT Chennai
"""

import pandas as pd
import numpy as np
import os
import sys

print("="*60)
print("ELECTRICITY THEFT DETECTION SYSTEM")
print("SRIP 2026 - VIT Chennai")
print("="*60)

# Get file path from user
print("\n📂 Paste or type the full path to your CSV file:")
print("   (You can right-click in the terminal to paste)")
print()
file_path = input("👉 Path: ").strip().strip('"')

# Check if file exists
if not os.path.exists(file_path):
    print(f"\n❌ File not found: {file_path}")
    sys.exit()

print(f"\n✅ File found: {os.path.basename(file_path)}")

# Load dataset
print("\n📊 Loading dataset...")
df = pd.read_csv(file_path)
print(f"✅ Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# Show columns
print("\n📋 Column names:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i}. {col}")

# Show first 5 rows
print("\n📋 First 5 rows:")
print(df.head())

# Show basic info
print("\n📊 Dataset info:")
print(f"   Missing values: {df.isnull().sum().sum()}")
print(f"   Duplicates: {df.duplicated().sum()}")

# Try to identify target column
target_col = None
for col in ['Class', 'Label', 'Theft_Type', 'class', 'label', 'target']:
    if col in df.columns:
        target_col = col
        break

if target_col:
    print(f"\n✅ Target column found: '{target_col}'")
    print("\n   Class distribution:")
    print(df[target_col].value_counts())
else:
    print("\n⚠️ No standard target column found.")
    print(f"   Available columns: {list(df.columns)}")

# Save info
os.makedirs('results', exist_ok=True)
df.to_csv('results/loaded_data.csv', index=False)
print("\n💾 Saved copy to: results/loaded_data.csv")

print("\n" + "="*60)
print("✅ Dataset loaded successfully!")
print("="*60)