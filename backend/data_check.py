import pandas as pd

# Load dataset
df = pd.read_csv("../data/bank_data.csv", sep=";")

print("=" * 60)
print("DATASET SHAPE")
print("=" * 60)
print(df.shape)

print("\n" + "=" * 60)
print("COLUMN NAMES")
print("=" * 60)
print(df.columns.tolist())

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)
print(df.dtypes)

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(df.isnull().sum())

print("\n" + "=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)
print(df.duplicated().sum())

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)
print(df["y"].value_counts())

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION (%)")
print("=" * 60)
print(df["y"].value_counts(normalize=True) * 100)

print("\n" + "=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)
print(df.head())