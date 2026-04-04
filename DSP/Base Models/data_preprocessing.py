# Shared preprocessing pipeline used by all models.
# Handles feature selection, label encoding, median imputation,
# time feature extraction and train/test splitting.
# ActualDurationMins is only used as the target variable, never as a feature.

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Fixed random seed for reproducibility
RANDOM_SEED = 89

# Pre-operative features available before surgery begins.
# Post-operative columns and free-text columns are excluded
# to avoid data leakage or impractical inputs.
CATEGORICAL_FEATURES = [
    "TheatreRoom",
    "anaesthetic_desc",
    "listing_cons_code",           # listing consultant
    "theat_surg_1_national_code",  # operating surgeon
    "admission_type",
    "ProcedureDescription",
]

NUMERIC_FEATURES = [
    "ExpectedDurationMins",
    "age_at_operation",
    "ASAScore",
    "intended_management",
    "sex_national_code",
]

TARGET = "ActualDurationMins"

# Path to the cleaned dataset (looked for inside this folder)
DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "NBT_SmallSet_Final.csv"
)

# Label encoders stored globally so they can be reused after preprocessing
_label_encoders: dict[str, LabelEncoder] = {}


# Returns the fitted label encoders (available after preprocess() is called)
def get_label_encoders() -> dict[str, LabelEncoder]:
    return _label_encoders


# Extracts the hour of day from the into_theatre column and adds it as theatre_hour
def _extract_hour(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    parsed = pd.to_datetime(df["into_theatre"], format="%H:%M:%S", errors="coerce")
    df["theatre_hour"] = parsed.dt.hour
    # Fill any unparseable times with the median hour
    df["theatre_hour"] = df["theatre_hour"].fillna(df["theatre_hour"].median())
    return df


# Loads the dataset, encodes features, and splits into train/test.
# filepath: path to CSV (defaults to NBT_SmallSet_Final.csv)
# test_size: fraction held out for testing (default 0.2 = 80:20 split)
# include_expected: if True, includes ExpectedDurationMins as a feature
# Returns X_train, X_test, y_train, y_test, feature_names, df
def preprocess(
    filepath: str | None = None,
    test_size: float = 0.2,
    include_expected: bool = True,
) -> tuple:
    global _label_encoders

    # Load data
    if filepath is None:
        filepath = DEFAULT_DATA_PATH
    df = pd.read_csv(filepath)
    print(f"Loaded dataset: {len(df)} rows from {os.path.basename(filepath)}")

    # Extract hour-of-day feature
    df = _extract_hour(df)

    # Select features
    numeric_feats = NUMERIC_FEATURES.copy()
    if not include_expected and "ExpectedDurationMins" in numeric_feats:
        numeric_feats.remove("ExpectedDurationMins")
    numeric_feats.append("theatre_hour")

    all_features = CATEGORICAL_FEATURES + numeric_feats

    # Fill missing categorical values with "Unknown"
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("Unknown")

    # Label encode categorical features
    _label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        _label_encoders[col] = le

    # Fill missing numeric values with the column median
    for col in numeric_feats:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Build feature matrix and target vector
    X = df[all_features].values
    y = df[TARGET].values
    feature_names = all_features

    # 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED
    )

    print(f"Features ({len(feature_names)}): {feature_names}")
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"Include ExpectedDurationMins: {include_expected}")

    return X_train, X_test, y_train, y_test, feature_names, df


# Saves a copy of the original dataset with the model's predictions as a new column.
# Reads the original (non-encoded) CSV so the output is human-readable.
def save_predictions_csv(
    df: pd.DataFrame,
    y_pred_all: np.ndarray,
    model_name: str,
    output_dir: str | None = None,
):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions")
    os.makedirs(output_dir, exist_ok=True)

    original_df = pd.read_csv(DEFAULT_DATA_PATH)
    original_df[model_name] = np.round(y_pred_all, 2)

    filename = f"{model_name.replace(' ', '_')}_predictions.csv"
    path = os.path.join(output_dir, filename)
    original_df.to_csv(path, index=False)
    print(f"Saved predictions to: {filename}")
    return path


# Standalone test
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, features, df = preprocess()
    print(f"\nPreprocessing complete. Shape: X_train={X_train.shape}, X_test={X_test.shape}")
