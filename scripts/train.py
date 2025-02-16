import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import logging
import logging.config

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import load_data, preprocess_data, save_preprocessing_artifacts
from src.feature_engineering import select_features
from src.model_training import train_models
from src.evaluation import evaluate_model
from src.hyperparameter_tuning import hyperparameter_tuning
from src.save_artifacts import save_model_artifacts

# Load logging configuration
with open("config/logging.yaml", "r") as file:
    logging.config.dictConfig(yaml.safe_load(file))

logger = logging.getLogger(__name__)

# Load configuration
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

def main():
    try:
        logger.info("Loading and preprocessing data...")
        # Load and preprocess data
        df = load_data(config["data"]["raw_path"])
        df_cleaned, label_encoders, scaler = preprocess_data(df)

        # Feature selection and engineering
        df_selected, dropped_features = select_features(df_cleaned)

        # Split data into train and test sets
        X = df_selected.drop(columns=["median_house_value"])
        y = df_selected["median_house_value"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["training"]["test_size"], random_state=config["training"]["random_state"])

        logger.info("Training models...")
        # Train models
        trained_models = train_models(X_train, y_train, config)

        # Evaluate models
        results = {}
        for name, model in trained_models.items():
            y_pred = model.predict(X_test)
            results[name] = evaluate_model(y_test, y_pred)

        logger.info("Hyperparameter tuning for the best model...")
        # Hyperparameter tuning for the best model (XGBoost)
        best_params = hyperparameter_tuning(X_train, y_train, config["training"]["random_state"])
        optimized_model = xgb.XGBRegressor(objective="reg:squarederror", **best_params, random_state=config["training"]["random_state"])
        optimized_model.fit(X_train, y_train)

        # Evaluate the optimized model
        y_pred_optimized = optimized_model.predict(X_test)
        optimized_results = evaluate_model(y_test, y_pred_optimized)

        # Save model and preprocessing artifacts
        save_model_artifacts(optimized_model, scaler, label_encoders, config["model"]["path"], config["model"]["scaler_path"], config["model"]["label_encoders_path"])
        save_preprocessing_artifacts(scaler, label_encoders, config["model"]["scaler_path"], config["model"]["label_encoders_path"])

        # Save train and test datasets
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        train_data.to_csv(config["data"]["processed_path"] + '/train.csv', index=False)
        test_data.to_csv(config["data"]["processed_path"] + '/test.csv', index=False)

        logger.info("Model Training Completed. Results:")
        logger.info(results)
        logger.info("Optimized Model Results:")
        logger.info(optimized_results)
        logger.info("Train and Test datasets saved!")

    except Exception as e:
        logger.exception("Exception occurred during training: %s", str(e))

if __name__ == "__main__":
    main()