import os
import seaborn as sns

os.makedirs("data/raw", exist_ok=True)

df = sns.load_dataset("titanic")
df.to_csv("data/raw/titanic.csv", index=False)

print("Dataset downloaded successfully.")
