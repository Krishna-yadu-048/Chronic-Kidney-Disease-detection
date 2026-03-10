import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import joblib
import os

from src.data_loader import TARGET_COL, SELECTED_FEATURES


def split_features_target(df: pd.DataFrame):
    X = df[SELECTED_FEATURES].values
    y = df[TARGET_COL].values
    return X, y


def fit_scaler(X_train, save_dir: str = "outputs"):
    os.makedirs(save_dir, exist_ok=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
    return X_scaled, scaler


def apply_scaler(X, save_dir: str = "outputs"):
    scaler = joblib.load(os.path.join(save_dir, "scaler.pkl"))
    return scaler.transform(X)


def oversample_minority(X_train, y_train, random_state: int = 42):
    """
    Upsample minority class to match majority class size.
    Only ever applied to training data — never test or validation.
    """
    X_maj = X_train[y_train == 1]
    y_maj = y_train[y_train == 1]
    X_min = X_train[y_train == 0]
    y_min = y_train[y_train == 0]

    X_min_up, y_min_up = resample(
        X_min, y_min, replace=True,
        n_samples=len(X_maj), random_state=random_state
    )

    X_bal = np.vstack([X_maj, X_min_up])
    y_bal = np.hstack([y_maj, y_min_up])
    idx   = np.random.RandomState(random_state).permutation(len(y_bal))
    return X_bal[idx], y_bal[idx]


def preprocess_single(input_dict: dict, save_dir: str = "outputs"):
    """Preprocess one patient's values for inference."""
    row = [float(input_dict.get(f, 0) or 0) for f in SELECTED_FEATURES]
    X   = np.array(row).reshape(1, -1)
    return apply_scaler(X, save_dir=save_dir)
