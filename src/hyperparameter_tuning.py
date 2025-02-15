import pandas as pd
import joblib
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Load config
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load data
X_train = pd.read_csv(config["data"]["processed_train_path"])
y_train = pd.read_csv("data/processed/y_train.csv")

# Define parameter grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10]
}

# Perform grid search
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring="neg_mean_squared_error")
grid_search.fit(X_train, y_train.values.ravel())

# Print best parameters
best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")

# Save best model
joblib.dump(grid_search.best_estimator_, config["model"]["model_path"])
print(f"Best model saved at {config['model']['model_path']}")
