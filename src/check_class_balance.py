# check_class_balance.py
import pandas as pd

df = pd.read_csv('results/loaded_data.csv')

print("="*60)
print("CLASS DISTRIBUTION IN YOUR DATASET")
print("="*60)

class_counts = df['theft'].value_counts()
print(class_counts)

print("\n" + "="*60)
print("CLASS PERCENTAGES")
print("="*60)
for cls, count in class_counts.items():
    pct = (count / len(df)) * 100
    print(f"{cls}: {pct:.1f}%")