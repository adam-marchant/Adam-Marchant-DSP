# Linear Regression - No Prediction Time Variant
# Predicts ActualDurationMins without using ExpectedDurationMins.

import os
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import preprocess, save_predictions_csv, RANDOM_SEED

MODEL_NAME = "Linear Regression"


# Trains a Linear Regression model (no ExpectedDurationMins), evaluates it,
# and saves predictions.
# Returns a dict with model_name, mae, rmse, r2, cv_mae_mean, cv_mae_std, y_test, y_pred
def train_and_evaluate():
    print("=" * 60)
    print(f"  {MODEL_NAME}  (No Expected Duration)")
    print("=" * 60)

    X_train, X_test, y_train, y_test, feature_names, df = preprocess()

    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f"\nModel trained on {len(X_train)} samples.")

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- Test Set Results ---")
    print(f"  MAE:  {mae:.2f} mins")
    print(f"  RMSE: {rmse:.2f} mins")
    print(f"  R²:   {r2:.4f}")

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

    print(f"\n--- Feature Coefficients ---")
    coefs = sorted(zip(feature_names, model.coef_), key=lambda x: abs(x[1]), reverse=True)
    for name, coef in coefs:
        print(f"  {name:>35s}: {coef:+.4f}")

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


if __name__ == "__main__":
    results = train_and_evaluate()
