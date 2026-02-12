import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

os.makedirs("models", exist_ok=True)

df = pd.read_csv("features/features.csv")

X = df.drop("survived", axis=1)
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, "models/model.pkl")

print("Model training completed.")
