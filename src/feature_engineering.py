import pandas as pd
import yaml

# Load configuration
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

class FeatureEngineering:
    def __init__(self):
        pass

    def feature_transformations(self, df):
        """Applies feature transformations like log transformation, polynomial features, etc."""
        if "sqft" in df.columns:
            df["log_sqft"] = df["sqft"].apply(lambda x: np.log1p(x))
        return df

if __name__ == "__main__":
    df = pd.read_csv(config["data"]["processed_train_path"])
    fe = FeatureEngineering()
    df = fe.feature_transformations(df)
    df.to_csv(config["data"]["processed_train_path"], index=False)
    print("Feature Engineering Completed.")
