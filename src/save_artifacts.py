import joblib
import os

def save_model_artifacts(model, scaler, label_encoders, model_path, scaler_path, label_encoders_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoders, label_encoders_path)
    print("\nModel and preprocessing artifacts saved successfully!")