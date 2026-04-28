# DSP Project — Surgical Duration Prediction

Predicting the actual duration of surgical operations at Southmead Hospital using machine learning, benchmarked against the existing human estimate (`ExpectedDurationMins`).

---

## Project Overview

Theatre scheduling relies heavily on surgeons' manual time estimates. This project investigates whether ML models can predict `ActualDurationMins` from pre-operative patient and procedural data, and whether this is possible even when the human estimate is withheld.

Two parallel model suites are included:

| Suite | Folder | `ExpectedDurationMins` included? |
|---|---|---|
| **Base Models** | `Base Models - Copy/` | ✅ Yes |
| **No Prediction Time (NPT) Models** | `No Prediction Time Models - Copy/` | ❌ No |

The human baseline (MAE = 39.25 min, RMSE = 57.26 min, R² = 0.4785) is used as the reference point for both suites.

---

## Dataset — `NBT_SmallSet_Final.csv`

A cleaned extract of surgical records from Southmead Hospital.

**Rows:** 11,645 operations  
**Target variable:** `ActualDurationMins` — the ground truth operation length in minutes

### Columns

| Column | Type | Description |
|---|---|---|
| `TheatreRoom` | categorical | Operating theatre identifier |
| `admission_type` | integer | Admission category code |
| `SessionIDdesc` | string | Human-readable session label (consultant + theatre) |
| `intended_management` | integer | Planned management pathway code |
| `actual_proc_1_procedure_code` | string | OPCS-4 procedure code |
| `ProcedureDescription` | string | Text description of the procedure |
| `theatre_notes` | string | Free-text pre-operative notes (not used as a feature) |
| `ExpectedDurationMins` | integer | Surgeon's pre-operative time estimate |
| `into_theatre` | time (HH:MM:SS) | Time the patient entered theatre |
| `anaesthetic_start_time` | time (HH:MM:SS) | Anaesthetic start |
| `incision` | time (HH:MM:SS) | Knife-to-skin time |
| `closure` | time (HH:MM:SS) | Wound closure time |
| `out_of_theatre` | time (HH:MM:SS) | Time patient left theatre |
| `operation_end_time` | time (HH:MM:SS) | Surgeon's recorded end time |
| `operation_length_mins` | integer | Duration computed from timestamps |
| `sex_national_code` | integer | Patient sex (national coding) |
| `age_at_operation` | integer | Patient age in years |
| `ASAScore` | integer | ASA physical status classification (1–5) |
| `anaesthetic_desc` | categorical | Anaesthetic type (GA, LA, Spinal, etc.) |
| `listing_cons_code` | string | National code of the listing consultant |
| `theat_surg_1_national_code` | string | National code of the operating surgeon |
| `ActualDurationMins` | integer | **Target — actual operation duration in minutes** |

---

## Repository Structure

```
DSP Project/
│
├── NBT_SmallSet_Final.csv               # Source dataset (place in both model folders)
│
├── Base Models - Copy/                  # Models that use ExpectedDurationMins
│   ├── data_preprocessing.py
│   ├── linear_regression.py
│   ├── random_forest.py
│   ├── gradient_boosting.py
│   ├── neural_network.py
│   └── model_comparison.py
│
└── No Prediction Time Models - Copy/    # Models without ExpectedDurationMins
    ├── data_preprocessing.py
    ├── linear_regression.py
    ├── random_forest.py
    ├── gradient_boosting.py
    ├── neural_network.py
    └── model_comparison.py
```

Each suite is self-contained; `NBT_SmallSet_Final.csv` must be placed in the same folder as the scripts (the default data path resolves relative to `__file__`).

---

## Models

Four ML models are implemented identically across both suites. Each exposes a single `train_and_evaluate()` entry point and returns a standardised results dict.

### Linear Regression (`linear_regression.py`)
Baseline model assuming a linear mapping from features to duration. Useful as a performance floor. Reports feature coefficients ranked by absolute magnitude.

### Random Forest (`random_forest.py`)
Ensemble of 200 decision trees (`max_depth=20`, `min_samples_leaf=5`). Handles non-linear feature interactions and mixed types well. Reports feature importances.

### Gradient Boosting — XGBoost (`gradient_boosting.py`)
Sequential boosting with 300 rounds (`max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, L1+L2 regularisation). Typically the strongest model on structured tabular data. Reports feature importances.

### Neural Network — MLP (`neural_network.py`)
Four-layer feedforward network: `Input → 128 → 64 → 32 → 1` (ReLU activations, BatchNormalization, Dropout 0.3 at each hidden layer). Trained with Adam (`lr=0.001`), Huber loss, early stopping (patience 15), and learning-rate reduction on plateau. Requires `StandardScaler` — applied per fold during CV to avoid leakage.

---

## Preprocessing (`data_preprocessing.py`)

Both suites share the same pipeline logic, differing only in whether `ExpectedDurationMins` is included.

**Feature set used in Base Models (11 features):**

| Category | Features |
|---|---|
| Categorical (label-encoded) | `TheatreRoom`, `anaesthetic_desc`, `listing_cons_code`, `theat_surg_1_national_code`, `admission_type`, `ProcedureDescription` |
| Numeric | `ExpectedDurationMins`, `age_at_operation`, `ASAScore`, `intended_management`, `sex_national_code` |
| Derived | `theatre_hour` (hour of day extracted from `into_theatre`) |

**In the NPT suite**, `ExpectedDurationMins` is removed, leaving **10 features**.

**Split:** 80% train / 20% test, stratified by `RANDOM_SEED = 89`.

**Outputs:** Each model saves a copy of the original dataset with a predictions column appended to a `predictions/` subfolder.

---

## Model Comparison (`model_comparison.py`)

Orchestrates all four models sequentially, collects metrics, and generates comparison outputs.

**Metrics reported:**
- MAE (Mean Absolute Error, minutes)
- RMSE (Root Mean Squared Error, minutes)
- R² (coefficient of determination)
- 5-fold cross-validation MAE with 95% confidence interval

**Outputs saved to `graphs/`:**

| File | Contents |
|---|---|
| `model_comparison_bar_charts[_no_expected].png` | Side-by-side MAE / RMSE / R² bar charts vs human baseline |
| `model_comparison_scatter_grid[_no_expected].png` | 2×2 predicted vs actual scatter plots, one per model |
| `model_comparison_cv_mae[_no_expected].png` | CV MAE bar chart with 95% CI error bars and human baseline line |

A plain-text summary table is also saved as `model_comparison_summary.txt`.

**Human baseline** (pre-loaded, not re-computed):

| MAE | RMSE | R² |
|---|---|---|
| 39.25 min | 57.26 min | 0.4785 |

---

## Installation

```bash
pip install numpy pandas scikit-learn xgboost tensorflow matplotlib
```

Python 3.10+ recommended (the code uses `str | None` union syntax).

---

## Usage

**Run the full Base Models comparison:**

```bash
cd "Base Models - Copy"
python model_comparison.py
```

**Run the No Prediction Time comparison:**

```bash
cd "No Prediction Time Models - Copy"
python model_comparison.py
```

**Run a single model in isolation:**

```bash
python random_forest.py          # or linear_regression.py, gradient_boosting.py, neural_network.py
```

All scripts resolve the dataset path relative to their own location, so they must be run from within their folder or the `DEFAULT_DATA_PATH` constant in `data_preprocessing.py` must be updated.

---

## Reproducibility

All models use `RANDOM_SEED = 89` for train/test splitting, sklearn model initialisation, TensorFlow/NumPy random state, and cross-validation shuffling. Results should be identical across runs on the same hardware and library versions.