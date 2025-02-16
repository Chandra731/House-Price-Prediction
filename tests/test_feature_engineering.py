import pytest
import pandas as pd
from src.feature_engineering import select_features

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
        'ocean_proximity_NEAR BAY': [1, 1, 1, 1]
    }
    return pd.DataFrame(data)

def test_select_features(sample_data):
    df_selected, dropped_features = select_features(sample_data, threshold=0.8)
    
    assert 'total_bedrooms' in dropped_features or 'households' in dropped_features
    assert 'total_rooms' in df_selected.columns
    assert 'total_bedrooms' not in df_selected.columns or 'households' not in df_selected.columns