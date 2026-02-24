# Data Preprocessing Module
# This module contains functions for data cleaning and preprocessing

def load_data(filepath):
    """Load dataset from file"""
    pass

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def preprocess_data(df):
    """
    Preprocess the credit card fraud dataset.

    Steps:
    1. Separate features and target
    2. Perform stratified train-test split
    3. Scale 'Amount' and 'Time' features
    4. Apply SMOTE on training data
    """

    # 1️⃣ Separate features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # 2️⃣ Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 3️⃣ Feature Scaling
    scaler = StandardScaler()

    X_train[['Amount', 'Time']] = scaler.fit_transform(
        X_train[['Amount', 'Time']]
    )

    X_test[['Amount', 'Time']] = scaler.transform(
        X_test[['Amount', 'Time']]
    )

    # 4️⃣ Apply SMOTE only on training data
    smote = SMOTE(random_state=42)

    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train, y_train
    )

    return X_train_resampled, X_test, y_train_resampled, y_test
