import joblib
import yaml
import os

# Load config file
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

class ModelLoader:
    def __init__(self):
        self.model_path = config["model"]["model_path"]
        self.vectorizer_path = config["model"]["vectorizer_path"]
        self.model = None
        self.vectorizer = None

    def load_model(self):
        """Loads trained model and vectorizer."""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"Model loaded from {self.model_path}")
        else:
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        if os.path.exists(self.vectorizer_path):
            self.vectorizer = joblib.load(self.vectorizer_path)
            print(f"Vectorizer loaded from {self.vectorizer_path}")
        else:
            raise FileNotFoundError(f"Vectorizer file not found at {self.vectorizer_path}")

        return self.model, self.vectorizer

if __name__ == "__main__":
    loader = ModelLoader()
    model, vectorizer = loader.load_model()
