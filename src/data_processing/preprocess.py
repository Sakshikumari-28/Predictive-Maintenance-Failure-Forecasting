import pandas as pd
import os

# Load raw dataset
input_path = "data/raw/ai4i2020.csv"
df = pd.read_csv(input_path)

print("Original dataset shape:", df.shape)

# Remove duplicate rows
df = df.drop_duplicates()

# Check for missing values
print("Total missing values:", df.isnull().sum().sum())

# Save processed dataset
output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "cleaned_data.csv")
df.to_csv(output_path, index=False)

print("Processed dataset shape:", df.shape)
print("Cleaned dataset saved to:", output_path)