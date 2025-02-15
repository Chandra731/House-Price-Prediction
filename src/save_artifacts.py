import joblib
import yaml
import os

# Load config
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Define paths
model_path = config["model"]["model_path"]
vectorizer_path = config["model"]["vectorizer_path"]
evaluation_path = config["artifacts"]["evaluation_path"]

# Ensure artifacts directory exists
os.makedirs(os.path.dirname(model_path), exist_ok=True)

def save_model(model):
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")

def save_vectorizer(vectorizer):
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Vectorizer saved at {vectorizer_path}")

def save_evaluation_results(results):
    with open(evaluation_path, "w") as f:
        yaml.dump(results, f)
    print(f"Evaluation results saved at {evaluation_path}")
