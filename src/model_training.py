import pandas as pd
import yaml
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load configuration
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

class ModelTrainer:
    def __init__(self):
        self.model = RandomForestRegressor(**config["model"]["hyperparameters"])

    def train(self, X_train, y_train):
        """Trains the model."""
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """Evaluates the model."""
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")
        return mse

    def save_model(self):
        """Saves the trained model."""
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(self.model, "artifacts/best_model.pkl")

if __name__ == "__main__":
    X_train = pd.read_csv(config["data"]["processed_train_path"])
    y_train = pd.read_csv("data/processed/y_train.csv")

    trainer = ModelTrainer()
    trainer.train(X_train, y_train)
    
    X_test = pd.read_csv(config["data"]["processed_test_path"])
    y_test = pd.read_csv("data/processed/y_test.csv")

    trainer.evaluate(X_test, y_test)
    trainer.save_model()
    print("Model Training Completed.")
