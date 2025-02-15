from flask import Flask, request, jsonify
import joblib
import pandas as pd
import yaml

# Load configuration
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Load model and preprocessing artifacts
model = joblib.load("artifacts/best_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint for model inference."""
    try:
        data = request.get_json()
        df = pd.DataFrame(data)
        df_scaled = scaler.transform(df)
        prediction = model.predict(df_scaled)
        return jsonify({"predicted_price": prediction.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
