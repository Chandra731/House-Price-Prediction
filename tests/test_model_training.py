import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from src.model_training import train_models
import yaml

# Load configuration
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

@pytest.fixture
def sample_data():
    data = {
        'longitude': [-122.23, -122.22, -122.24, -122.25],
        'latitude': [37.88, 37.86, 37.85, 37.84],
        'housing_median_age': [41, 21, 52, 52],
        'total_rooms': [880, 7099, 1467, 1274],
        'total_bedrooms': [129, 1106, 190, 235],
        'population': [322, 2401, 496, 558],
        'households': [126, 1138, 177, 219],
        'median_income': [8.3252, 8.3014, 7.2574, 5.6431],
        'median_house_value': [452600, 358500, 352100, 341300],
        'ocean_proximity_NEAR BAY': [1, 1, 1, 1]
    }
    return pd.DataFrame(data)

def test_train_models(sample_data):
    X = sample_data.drop(columns=["median_house_value"])
    y = sample_data["median_house_value"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["training"]["test_size"], random_state=config["training"]["random_state"])
    
    trained_models = train_models(X_train, y_train, config)
    
    assert len(trained_models) == 4
    for model in trained_models.values():
        assert hasattr(model, "predict")