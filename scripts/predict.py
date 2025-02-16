import sys
import os
import pandas as pd
import joblib

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import preprocess_data

# Load the trained model and preprocessing artifacts
model = joblib.load("artifacts/optimized_xgboost_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")
label_encoders = joblib.load("artifacts/label_encoders.pkl")

def make_prediction(input_data):
    input_df = pd.DataFrame(input_data, index=[0])
    preprocessed_data, _, _ = preprocess_data(input_df, label_encoders)

    # Apply scaling
    scaled_data = scaler.transform(preprocessed_data)

    # Make prediction
    prediction_scaled = model.predict(scaled_data)

    # Apply inverse scaling to get the true value
    prediction = scaler.inverse_transform([prediction_scaled])[0][0]
    return prediction