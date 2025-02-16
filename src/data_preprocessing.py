import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df, label_encoders=None, scaler=None):
    if label_encoders is None:
        label_encoders = {}
    if scaler is None:
        scaler = StandardScaler()

    # Separate numerical & categorical columns
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    # Fill numerical columns with Median
    imputer_num = SimpleImputer(strategy="median")
    df[num_cols] = imputer_num.fit_transform(df[num_cols])

    # Fill categorical columns with Mode
    imputer_cat = SimpleImputer(strategy="most_frequent")
    df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])

    # Encode categorical features
    for feature in cat_cols:
        if feature not in label_encoders:
            label_encoders[feature] = LabelEncoder()
        df[feature] = label_encoders[feature].fit_transform(df[feature])

    # Apply scaling to numerical features
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df, label_encoders, scaler

def save_preprocessing_artifacts(scaler, label_encoders, scaler_path, label_encoders_path):
    import joblib
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoders, label_encoders_path)