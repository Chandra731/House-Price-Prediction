import joblib

def load_model(model_path, scaler_path, label_encoders_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoders = joblib.load(label_encoders_path)
    return model, scaler, label_encoders