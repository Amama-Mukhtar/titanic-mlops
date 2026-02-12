import pandas as pd
import joblib

# Load model
model = joblib.load("models/model.pkl")

# Load features
df = pd.read_csv("features/features.csv")

X = df.drop("survived", axis=1)

# Generate predictions
predictions = model.predict(X)

# Save predictions
pd.DataFrame(predictions, columns=["prediction"]).to_csv(
    "results/predictions.csv", index=False
)

print("Predictions generated successfully.")
