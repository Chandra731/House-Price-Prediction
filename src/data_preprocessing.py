import pandas as pd
import numpy as np
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Load configuration
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoders = {}

    def load_data(self):
        """Loads raw dataset from the specified path."""
        data_path = config["data"]["raw_path"]
        df = pd.read_csv(data_path)
        return df

    def preprocess(self, df):
        """Handles missing values, encodes categorical variables, and scales numerical features."""
        target = config["preprocessing"]["target_column"]
        numerical_features = config["preprocessing"]["numerical_features"]
        categorical_features = config["preprocessing"]["categorical_features"]

        # Fill missing values
        df.fillna(df.median(numeric_only=True), inplace=True)

        # Encode categorical features
        for col in categorical_features:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.encoders[col] = le

        # Scale numerical features
        df[numerical_features] = self.scaler.fit_transform(df[numerical_features])

        # Split dataset
        X = df.drop(columns=[target])
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["preprocessing"]["test_size"], random_state=config["preprocessing"]["random_state"])
        
        return X_train, X_test, y_train, y_test

    def save_preprocessing_artifacts(self):
        """Saves scaler and encoders as artifacts."""
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(self.scaler, "artifacts/scaler.pkl")
        joblib.dump(self.encoders, "artifacts/label_encoders.pkl")

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data()
    X_train, X_test, y_train, y_test = preprocessor.preprocess(df)
    preprocessor.save_preprocessing_artifacts()

    # Save processed data
    os.makedirs("data/processed", exist_ok=True)
    X_train.to_csv("data/processed/train.csv", index=False)
    X_test.to_csv("data/processed/test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    
    print("Data Preprocessing Completed.")
