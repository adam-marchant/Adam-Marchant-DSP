# Neural Network (Multi-Layer Perceptron) Model
# Feedforward neural network to capture non-linear patterns.
# Architecture: Input -> 128 -> 64 -> 32 -> 1 (all ReLU with BatchNorm and Dropout)
# Uses StandardScaler, early stopping, and learning rate reduction.

import os
import sys
import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Suppress TensorFlow info and warning logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=UserWarning)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# Add parent directory to path so we can import the preprocessing module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import preprocess, save_predictions_csv, RANDOM_SEED

# Set random seeds for reproducibility
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

MODEL_NAME = "Neural Network (MLP)"


# Builds and compiles the MLP model.
# Uses BatchNormalisation to stabilise training, Dropout (0.3) to prevent
# overfitting, and Huber loss for robustness to outliers.
def _build_model(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(32, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(1)  # linear output for regression
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="huber",
        metrics=["mae"],
    )
    return model


# Trains a Neural Network, evaluates it, and saves predictions.
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

    # Scale features (critical for neural networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    model = _build_model(input_dim=X_train_scaled.shape[1])

    # Callbacks for training control
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    print(f"\nTraining neural network...")
    model.fit(
        X_train_scaled, y_train,
        validation_split=0.15,  # 15% of training data used for validation
        epochs=200,
        batch_size=64,
        callbacks=[early_stop, reduce_lr],
        verbose=0,
    )
    print(f"Training complete.")

    # Evaluate on the test set
    y_pred = model.predict(X_test_scaled, verbose=0).flatten()

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n--- Test Set Results ---")
    print(f"  MAE:  {mae:.2f} mins")
    print(f"  RMSE: {rmse:.2f} mins")
    print(f"  R²:   {r2:.4f}")

    # Manual 5-fold cross-validation
    # Neural networks are not compatible with sklearn's cross_val_score,
    # so we implement CV manually.
    print(f"\n--- 5-Fold Cross-Validation (MAE) ---")
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_mae_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_all), 1):
        X_cv_train, X_cv_val = X_all[train_idx], X_all[val_idx]
        y_cv_train, y_cv_val = y_all[train_idx], y_all[val_idx]

        # Scale within each fold to prevent data leakage
        fold_scaler = StandardScaler()
        X_cv_train_s = fold_scaler.fit_transform(X_cv_train)
        X_cv_val_s = fold_scaler.transform(X_cv_val)

        fold_model = _build_model(input_dim=X_cv_train_s.shape[1])
        fold_model.fit(
            X_cv_train_s, y_cv_train,
            validation_split=0.15,
            epochs=200,
            batch_size=64,
            callbacks=[
                callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                        restore_best_weights=True),
                callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                            patience=5, min_lr=1e-6),
            ],
            verbose=0,
        )

        fold_pred = fold_model.predict(X_cv_val_s, verbose=0).flatten()
        fold_mae = mean_absolute_error(y_cv_val, fold_pred)
        cv_mae_scores.append(fold_mae)
        print(f"  Fold {fold}: MAE = {fold_mae:.2f} mins")

    cv_mae = np.array(cv_mae_scores)
    print(f"  Mean:   {cv_mae.mean():.2f} mins")
    print(f"  StdDev: {cv_mae.std():.2f} mins")
    print(f"  95% CI: {cv_mae.mean():.2f} ± {1.96 * cv_mae.std():.2f} mins")

    # Predict on the full dataset and save CSV
    X_all_scaled = scaler.transform(df[feature_names].values)
    y_pred_all = model.predict(X_all_scaled, verbose=0).flatten()
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
