# Model Comparison - No Prediction Time Variant
# Runs all four models (without ExpectedDurationMins) and compares
# their results against the human baseline.
# Answers: how well can we predict surgery duration without any human time estimate?

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from linear_regression import train_and_evaluate as lr_run
from random_forest import train_and_evaluate as rf_run
from gradient_boosting import train_and_evaluate as gb_run
from neural_network import train_and_evaluate as nn_run

# Human baseline metrics
HUMAN_BASELINE = {
    "model_name": "Human Estimate",
    "mae": 39.25,
    "rmse": 57.26,
    "r2": 0.4785,
    "cv_mae_mean": None,
    "cv_mae_std": None,
}

# Directory for saving graphs
GRAPH_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "graphs"
)
os.makedirs(GRAPH_DIR, exist_ok=True)


# Prints a formatted comparison table to the console
def _print_comparison_table(all_results: list[dict]):
    print("\n" + "=" * 80)
    print("  MODEL COMPARISON SUMMARY  (No ExpectedDurationMins)")
    print("=" * 80)
    print(f"{'Model':<28} {'MAE':>8} {'RMSE':>8} {'R²':>8}  {'CV MAE (5-fold)':>20}")
    print("-" * 80)
    for r in all_results:
        cv_str = (
            f"{r['cv_mae_mean']:.2f} ± {r['cv_mae_std']:.2f}"
            if r["cv_mae_mean"] is not None
            else "N/A"
        )
        print(
            f"{r['model_name']:<28} {r['mae']:>8.2f} {r['rmse']:>8.2f} "
            f"{r['r2']:>8.4f}  {cv_str:>20}"
        )
    print("=" * 80)

    ml_results = [r for r in all_results if r["model_name"] != "Human Estimate"]
    best = min(ml_results, key=lambda r: r["mae"])
    print(f"\n  Best model by MAE: {best['model_name']}  (MAE = {best['mae']:.2f})")

    improvement_mae = HUMAN_BASELINE["mae"] - best["mae"]
    improvement_pct = (improvement_mae / HUMAN_BASELINE["mae"]) * 100
    if improvement_mae > 0:
        print(
            f"  ↓ {improvement_mae:.2f} mins better than human estimate "
            f"({improvement_pct:.1f}% improvement)"
        )
    else:
        print(
            f"  ↑ {abs(improvement_mae):.2f} mins worse than human estimate "
            f"({abs(improvement_pct):.1f}% worse)"
        )
    print()


# Saves the comparison summary table to a .txt file in this folder
def _save_summary_txt(all_results: list[dict]):
    lines = []
    lines.append("=" * 80)
    lines.append("  MODEL COMPARISON SUMMARY  (No ExpectedDurationMins)")
    lines.append("=" * 80)
    lines.append(f"{'Model':<28} {'MAE':>8} {'RMSE':>8} {'R²':>8}  {'CV MAE (5-fold)':>20}")
    lines.append("-" * 80)
    for r in all_results:
        cv_str = (
            f"{r['cv_mae_mean']:.2f} ± {r['cv_mae_std']:.2f}"
            if r["cv_mae_mean"] is not None
            else "N/A"
        )
        lines.append(
            f"{r['model_name']:<28} {r['mae']:>8.2f} {r['rmse']:>8.2f} "
            f"{r['r2']:>8.4f}  {cv_str:>20}"
        )
    lines.append("=" * 80)

    ml_results = [r for r in all_results if r["model_name"] != "Human Estimate"]
    best = min(ml_results, key=lambda r: r["mae"])
    lines.append(f"")
    lines.append(f"  Best model by MAE: {best['model_name']}  (MAE = {best['mae']:.2f})")

    improvement_mae = HUMAN_BASELINE["mae"] - best["mae"]
    improvement_pct = (improvement_mae / HUMAN_BASELINE["mae"]) * 100
    if improvement_mae > 0:
        lines.append(
            f"  {improvement_mae:.2f} mins better than human estimate "
            f"({improvement_pct:.1f}% improvement)"
        )
    else:
        lines.append(
            f"  {abs(improvement_mae):.2f} mins worse than human estimate "
            f"({abs(improvement_pct):.1f}% worse)"
        )

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_comparison_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved summary text        → {path}")


# Bar charts comparing MAE, RMSE, R2 (no expected duration variant)
def _plot_metric_bars(all_results: list[dict]):
    names = [r["model_name"] for r in all_results]
    short_names = []
    for n in names:
        if n == "Human Estimate":
            short_names.append("Human\nEstimate")
        elif n == "Linear Regression":
            short_names.append("Linear\nRegression")
        elif n == "Random Forest":
            short_names.append("Random\nForest")
        elif n == "Gradient Boosting (XGBoost)":
            short_names.append("XGBoost")
        elif n == "Neural Network (MLP)":
            short_names.append("Neural\nNetwork")
        else:
            short_names.append(n)

    maes = [r["mae"] for r in all_results]
    rmses = [r["rmse"] for r in all_results]
    r2s = [r["r2"] for r in all_results]

    colours = ["#888888", "#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Model Comparison Against Human Baseline\n(No ExpectedDurationMins)",
        fontsize=15, fontweight="bold",
    )

    # MAE
    bars = axes[0].bar(short_names, maes, color=colours, edgecolor="black", linewidth=0.5)
    axes[0].set_title("Mean Absolute Error (MAE)", fontsize=12)
    axes[0].set_ylabel("Minutes")
    for bar, val in zip(bars, maes):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    # RMSE
    bars = axes[1].bar(short_names, rmses, color=colours, edgecolor="black", linewidth=0.5)
    axes[1].set_title("Root Mean Squared Error (RMSE)", fontsize=12)
    axes[1].set_ylabel("Minutes")
    for bar, val in zip(bars, rmses):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    # R²
    bars = axes[2].bar(short_names, r2s, color=colours, edgecolor="black", linewidth=0.5)
    axes[2].set_title("R² Score", fontsize=12)
    axes[2].set_ylabel("R²")
    axes[2].set_ylim(0, 1.0)
    for bar, val in zip(bars, r2s):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, "model_comparison_bar_charts_no_expected.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved bar chart comparison → {path}")


# 2x2 scatter grid (no expected duration variant)
def _plot_scatter_grid(ml_results: list[dict]):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        "Predicted vs Actual Duration – All Models\n(No ExpectedDurationMins)",
        fontsize=15, fontweight="bold",
    )
    axes = axes.flatten()

    for idx, r in enumerate(ml_results):
        ax = axes[idx]
        y_test = r["y_test"]
        y_pred = r["y_pred"]

        ax.scatter(y_test, y_pred, alpha=0.3, s=10, color="#4C72B0")
        lims = [0, max(y_test.max(), y_pred.max()) * 1.05]
        ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Actual Duration (mins)")
        ax.set_ylabel("Predicted Duration (mins)")
        ax.set_title(
            f"{r['model_name']}\n"
            f"MAE={r['mae']:.1f}  RMSE={r['rmse']:.1f}  R²={r['r2']:.3f}",
            fontsize=10,
        )
        ax.legend(fontsize=8)
        ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, "model_comparison_scatter_grid_no_expected.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved scatter grid        → {path}")


# CV MAE bar chart (no expected duration variant)
def _plot_cv_comparison(ml_results: list[dict]):
    names = [r["model_name"] for r in ml_results]
    means = [r["cv_mae_mean"] for r in ml_results]
    cis = [1.96 * r["cv_mae_std"] for r in ml_results]
    colours = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, means, yerr=cis, capsize=6, color=colours,
                  edgecolor="black", linewidth=0.5)
    ax.set_title(
        "5-Fold Cross-Validation MAE (95% CI)\n(No ExpectedDurationMins)",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylabel("MAE (mins)")
    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{m:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax.axhline(y=HUMAN_BASELINE["mae"], color="red", linestyle="--", linewidth=1.2,
               label=f"Human Baseline MAE ({HUMAN_BASELINE['mae']:.1f})")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(GRAPH_DIR, "model_comparison_cv_mae_no_expected.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved CV MAE comparison   → {path}")


# Runs every model (without ExpectedDurationMins) and produces comparison outputs
def run_all():
    print(f"\n{'#' * 70}")
    print(f"#  RUNNING ALL MODELS (WITHOUT human estimate)")
    print(f"{'#' * 70}\n")

    results = []

    print("\n[1/4] Linear Regression")
    results.append(lr_run())

    print("\n[2/4] Random Forest")
    results.append(rf_run())

    print("\n[3/4] Gradient Boosting (XGBoost)")
    results.append(gb_run())

    print("\n[4/4] Neural Network (MLP)")
    results.append(nn_run())

    all_results = [HUMAN_BASELINE] + results

    _print_comparison_table(all_results)

    print("Generating comparison charts...")
    _plot_metric_bars(all_results)
    _plot_scatter_grid(results)
    _plot_cv_comparison(results)
    _save_summary_txt(all_results)
    print("All comparison outputs saved.\n")

    return results


if __name__ == "__main__":
    run_all()
