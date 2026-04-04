# Linear Regression Model
# Baseline ML model that assumes a linear relationship between
# pre-operative features and the actual duration of the operation.
# Used as a minimum performance benchmark for the other models.

import os
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add parent directory to path so we can import the preprocessing module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import preprocess, save_predictions_csv, RANDOM_SEED


MODEL_NAME = "Linear Regression"


# Trains a Linear Regression model, evaluates it, and saves predictions.
# include_expected: whether to include ExpectedDurationMins as a feature
# Returns a dict with model_name, mae, rmse, r2, cv_mae_mean, cv_mae_std, y_test, y_pred
def train_and_evaluate(include_expected: bool = True):
    print("=" * 60)
    print(f"  {MODEL_NAME}")
    print(f"  Using human estimate: {include_expected}")
    print("=" * 60)

    # Load and preprocess data
    X_train, X_test, y_train, y_test, feature_names, df = preprocess(
        include_expected=include_expected
    )

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f"\nModel trained on {len(X_train)} samples.")

    # Evaluate on the test set
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- Test Set Results ---")
    print(f"  MAE:  {mae:.2f} mins")
    print(f"  RMSE: {rmse:.2f} mins")
    print(f"  R²:   {r2:.4f}")

    # 5-fold cross-validation for reliability
    # Uses negative MAE (sklearn convention), so we negate back
    cv_scores = cross_val_score(
        model, np.vstack([X_train, X_test]), np.concatenate([y_train, y_test]),
        cv=5, scoring="neg_mean_absolute_error", n_jobs=-1
    )
    cv_mae = -cv_scores
    print(f"\n--- 5-Fold Cross-Validation (MAE) ---")
    print(f"  Folds:  {np.round(cv_mae, 2)}")
    print(f"  Mean:   {cv_mae.mean():.2f} mins")
    print(f"  StdDev: {cv_mae.std():.2f} mins")
    print(f"  95% CI: {cv_mae.mean():.2f} ± {1.96 * cv_mae.std():.2f} mins")

    # Feature coefficients
    print(f"\n--- Feature Coefficients ---")
    coefs = sorted(zip(feature_names, model.coef_), key=lambda x: abs(x[1]), reverse=True)
    for name, coef in coefs:
        print(f"  {name:>35s}: {coef:+.4f}")

    # Predict on the full dataset and save CSV
    X_all = df[feature_names].values
    y_pred_all = model.predict(X_all)
    save_predictions_csv(df, y_pred_all, MODEL_NAME)

    print("=" * 60)

    return {
        "model_name": MODEL_NAME,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "cv_mae_mean": cv_mae.mean(),
        "cv_mae_std": cv_mae.std(),
        "y_test": y_test,
        "y_pred": y_pred,
    }


# Run as standalone script
if __name__ == "__main__":
    results = train_and_evaluate(include_expected=True)
