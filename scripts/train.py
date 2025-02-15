import pandas as pd
import joblib
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load config
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load data
X_train = pd.read_csv(config["data"]["processed_train_path"])
y_train = pd.read_csv("data/processed/y_train.csv")
X_test = pd.read_csv(config["data"]["processed_test_path"])
y_test = pd.read_csv("data/processed/y_test.csv")

# Train model
model = RandomForestRegressor(**config["model"]["hyperparameters"])
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Model MSE: {mse}")

# Save model
joblib.dump(model, config["model"]["model_path"])
print(f"Model saved at {config['model']['model_path']}")
