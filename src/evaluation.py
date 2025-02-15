import pandas as pd
import joblib
import yaml
from sklearn.metrics import mean_squared_error, r2_score

# Load config
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load model
model = joblib.load(config["model"]["model_path"])

# Load test data
X_test = pd.read_csv(config["data"]["processed_test_path"])
y_test = pd.read_csv("data/processed/y_test.csv")

# Make predictions
y_pred = model.predict(X_test)

# Compute metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print(f"Evaluation Metrics:\nMSE: {mse:.4f}\nR2 Score: {r2:.4f}")

# Save evaluation results
results = {"MSE": mse, "R2 Score": r2}
with open(config["artifacts"]["evaluation_path"], "w") as f:
    yaml.dump(results, f)
print(f"Evaluation results saved at {config['artifacts']['evaluation_path']}")
