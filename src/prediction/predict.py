import pandas as pd
import pickle

# 1. Load dataset
df = pd.read_csv("data/processed/cleaned_data.csv")

# 2. Select the same features used during training
features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = df[features]

# 3. Load trained model
with open("models/failure_prediction_model.pkl", "rb") as file:
    model = pickle.load(file)

# 4. Generate predictions
predictions = model.predict(X)
# Generate failure probability
failure_probability = model.predict_proba(X)[:, 1]
# 5. Add predictions to the dataset
df["Predicted Failure"] = predictions
df["Failure Probability"] = failure_probability
# Add risk band
df["Risk Band"] = pd.cut(
    df["Failure Probability"],
    bins=[-0.01, 0.30, 0.60, 1.00],
    labels=["Low", "Medium", "High"]
)
# 6. Save prediction results
df.to_csv("data/processed/predictions.csv", index=False)

print("Predictions generated successfully.")
print("Prediction file saved to data/processed/predictions.csv")