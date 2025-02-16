import pandas as pd

def select_features(df):
    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population'] / df['households']

    features_to_drop = ['total_rooms', 'total_bedrooms', 'population']
    df = df.drop(columns=features_to_drop)

    return df, features_to_drop