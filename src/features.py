import pandas as pd
import os

os.makedirs("features", exist_ok=True)

df = pd.read_csv("data/processed/cleaned.csv")

# Create family size feature
df["family_size"] = df["sibsp"] + df["parch"]

# Drop non-numeric columns if any remain
df = df.select_dtypes(include=["number"])

df.to_csv("features/features.csv", index=False)

print("Feature engineering completed successfully.")
