import pandas as pd
import os

os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv("data/raw/titanic.csv")

df["age"] = df["age"].fillna(df["age"].median())

df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

# Convert sex to numeric
df["sex"] = df["sex"].map({"male": 0, "female": 1})

# One-hot encode embarked
df = pd.get_dummies(df, columns=["embarked"], drop_first=True)

# Drop unnecessary columns
columns_to_drop = ["deck", "alive", "class", "who", "adult_male", "embark_town"]
df.drop(columns=columns_to_drop, inplace=True)

df.to_csv("data/processed/cleaned.csv", index=False)

print("Preprocessing completed successfully.")
