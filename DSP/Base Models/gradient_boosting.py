# Gradient Boosting (XGBoost) Model
# Builds trees sequentially where each new tree corrects errors from
# the previous one. One of the best algorithms for structured/tabular data.
# Includes built-in regularisation to prevent overfitting.

import os
import sys
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add parent directory to path so we can import the preprocessing module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import preprocess, save_predictions_csv, RANDOM_SEED

MODEL_NAME = "Gradient Boosting (XGBoost)"


# Trains an XGBoost Regressor, evaluates it, and saves predictions.
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
    # n_estimators=300: number of boosting rounds
    # max_depth=6: moderate depth per tree
    # learning_rate=0.1: step size shrinkage to prevent overfitting
    # subsample=0.8: use 80% of data per tree (stochastic boosting)
    # colsample_bytree=0.8: use 80% of features per tree
    # reg_alpha=0.1: L1 regularisation
    # reg_lambda=1.0: L2 regularisation
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,  # suppress XGBoost warnings
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    print(f"\nModel trained: {model.n_estimators} boosting rounds, "
          f"max_depth={model.max_depth}, lr={model.learning_rate}")

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

    # Feature importance ranking
    print(f"\n--- Feature Importance ---")
    importances = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    for name, imp in importances:
        bar = "█" * int(imp * 50)
        print(f"  {name:>35s}: {imp:.4f}  {bar}")

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
