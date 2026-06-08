# find_real_theft45.py
import pandas as pd
import numpy as np

df = pd.read_csv('results/loaded_data.csv')

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

print("="*70)
print("REAL THEFT4 AND THEFT5 SAMPLES FROM YOUR DATASET")
print("="*70)

# Get Theft4 samples
theft4_samples = df[df['theft'] == 'Theft4'][feature_cols].head(3)
print("\n📊 REAL THEFT4 SAMPLES:")
for i in range(3):
    print(f"\nTheft4 Sample {i+1}:")
    for col, val in zip(feature_cols, theft4_samples.iloc[i].values):
        print(f"   {col}: {val:.2f}")

# Get Theft5 samples
theft5_samples = df[df['theft'] == 'Theft5'][feature_cols].head(3)
print("\n📊 REAL THEFT5 SAMPLES:")
for i in range(3):
    print(f"\nTheft5 Sample {i+1}:")
    for col, val in zip(feature_cols, theft5_samples.iloc[i].values):
        print(f"   {col}: {val:.2f}")

# Get Theft3 samples for comparison
theft3_samples = df[df['theft'] == 'Theft3'][feature_cols].head(3)
print("\n📊 REAL THEFT3 SAMPLES (for comparison):")
for i in range(3):
    print(f"\nTheft3 Sample {i+1}:")
    for col, val in zip(feature_cols, theft3_samples.iloc[i].values):
        print(f"   {col}: {val:.2f}")