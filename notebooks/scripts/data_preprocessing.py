# data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def clean_data(data, drop_dup = False):
    """Clean and handle missing and duplicate values."""

    # Drop Na/missing values
    data = data.dropna()

    #Drop duplicate values?
    if drop_dup == True:
        data = data.drop_duplicates()

    return data

def feature_engineering(data):
    """Perform feature engineering on the dataset."""
    # Turn risk level from categories to numbers

    RiskLevel = {'low risk':0, 
        'mid risk':1, 
        'high risk':2}

    # apply using map
    data['RiskLevel'] = data['RiskLevel'].map(RiskLevel).astype(float)

    # Remove outlier point
    data = data.drop(data.index[data.HeartRate == 7])
    
    # Drop Heart Rate variable
    data = data.drop(["HeartRate"], axis=1)

    # Drop Diastolic BP variable
    data.drop(['DiastolicBP'],axis=1)
    
    return data

def preprocess_data(file_path, drop_dup, scale = False):
    """Complete end-to-end data preprocessing pipeline."""
    # Load data
    data = load_data(file_path)

    # Clean data
    data = clean_data(data, drop_dup)

    # Feature engineering
    data = feature_engineering(data)

    # Split features and target variable
    X = data.drop('RiskLevel', axis=1)
    y = data['RiskLevel']

    # Standardize features (optional)
    if scale == True:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X, y