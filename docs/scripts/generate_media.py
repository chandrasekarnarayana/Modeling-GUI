#!/usr/bin/env python
"""
Generate documentation screenshots and a short promo video from synthetic examples.

The outputs land in docs/screenshots/ and docs/assets/.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import matplotlib

# Always use a headless backend for CI/automation.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import patches  # noqa: E402
from sklearn.metrics import auc, precision_recall_curve, roc_curve  # noqa: E402

from modeling_gui.visualization import plot_forecast  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCREENSHOTS = ROOT / "screenshots"
ASSETS = ROOT / "assets"
SCREENSHOTS.mkdir(parents=True, exist_ok=True)
ASSETS.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> Path:
    path = SCREENSHOTS / name
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def make_smart_analyze_run() -> Path:
    df_preview = pd.DataFrame(
        {
            "age": [34, 28, 41, 36],
            "income": [72000, 54000, 91000, 68000],
            "tenure_months": [24, 12, 36, 18],
            "target": [1, 0, 1, 0],
        }
    )
    metrics = pd.DataFrame(
        {
            "Metric": ["Accuracy", "ROC AUC", "F1", "LogLoss"],
            "Value": [0.912, 0.941, 0.903, 0.312],
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 7))

    axes[0].axis("off")
    axes[0].set_title("Demo dataset preview", fontsize=13, fontweight="bold")
    table = axes[0].table(
        cellText=df_preview.values,
        colLabels=df_preview.columns,
        colColours=["#f4f6fb"] * df_preview.shape[1],
        loc="center",
        cellLoc="center",
    )
    table.scale(1, 1.4)

    axes[1].axis("off")
    axes[1].set_title("Smart Analyze (AutoML)", fontsize=13, fontweight="bold")
    axes[1].text(0.0, 0.9, "Detected: classification", fontsize=12, fontweight="semibold")
    axes[1].text(0.0, 0.82, "Best model: Gradient Boosting", fontsize=12)
    axes[1].text(0.0, 0.74, "Primary metrics (Business): ROC AUC, F1", fontsize=11, color="#444")
    axes[1].text(0.0, 0.66, "Leaderboard models compared: 5", fontsize=11, color="#444")
    axes[1].text(0.0, 0.58, "Train/test split: 80/20, random_state=42", fontsize=11, color="#444")
    axes[1].text(0.0, 0.50, "Preprocessing: numeric-only, missing=none", fontsize=11, color="#444")

    axes[1].table(
        cellText=metrics.values,
        colLabels=metrics.columns,
        colColours=["#f4f6fb", "#f4f6fb"],
        cellLoc="center",
        bbox=[0.0, 0.05, 0.9, 0.35],
    )
    return _save(fig, "smart_analyze_run.png")


def make_leaderboard() -> Path:
    models = ["GradientBoosting", "RandomForest", "LogisticRegression", "Baseline"]
    scores = [0.912, 0.901, 0.854, 0.701]
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b3"]
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(models, scores, color=colors)
    ax.set_xlabel("ROC AUC")
    ax.set_xlim(0.6, 1.0)
    ax.invert_yaxis()
    ax.set_title("Leaderboard", fontsize=14, fontweight="bold")
    for bar, score in zip(bars, scores):
        ax.text(score + 0.01, bar.get_y() + bar.get_height() / 2, f"{score:.3f}", va="center")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    return _save(fig, "smart_analyze_leaderboard.png")


def make_tuning() -> Path:
    params = {
        "n_estimators": (200, 50, 500),
        "learning_rate": (0.08, 0.01, 0.3),
        "max_depth": (4, 2, 8),
        "subsample": (0.9, 0.5, 1.0),
    }
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Hyperparameter Tuning (Gradient Boosting)", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(params) + 1)
    ax.axis("off")
    for i, (name, (value, low, high)) in enumerate(params.items(), start=1):
        y = len(params) - i + 1
        ax.add_patch(
            patches.Rectangle((0.05, y - 0.25), 0.9, 0.35, edgecolor="#d0d7de", facecolor="#f8f9fb", lw=1.2)
        )
        ax.text(0.07, y + 0.03, name, fontsize=11, fontweight="semibold")
        norm_val = (value - low) / (high - low)
        ax.plot([0.12, 0.88], [y - 0.07, y - 0.07], color="#cbd5e1", lw=2.5)
        ax.plot([0.12 + norm_val * 0.76], [y - 0.07], marker="o", color="#4c72b0", markersize=8)
        ax.text(0.9, y + 0.03, f"{value:g}", ha="right", fontsize=10, color="#444")
        ax.text(0.12, y - 0.16, f"Range: {low:g} – {high:g}", fontsize=9, color="#666")
    ax.text(0.05, 0.3, "Preset: Balanced (speed vs accuracy)", fontsize=10, color="#444")
    ax.text(0.05, 0.15, "Next step: Run tuned candidates", fontsize=10, color="#444")
    return _save(fig, "smart_analyze_tuning.png")


def make_metrics_curves() -> Path:
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    y_scores = rng.random(200) * 0.6 + y_true * 0.4  # lift positives
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, color="#4c72b0", lw=2, label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], color="#ccc", lw=1.5, linestyle="--")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(recall, precision, color="#55a868", lw=2, label=f"PR AUC = {pr_auc:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision–Recall Curve")
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.3)
    fig.suptitle("Classification metrics", fontsize=14, fontweight="bold")
    return _save(fig, "metrics_roc_pr.png")


def make_forecast_plot() -> Path:
    idx = pd.date_range("2024-01-01", periods=20, freq="D")
    y_train = pd.Series(np.linspace(100, 120, 14) + np.sin(np.linspace(0, 3, 14)) * 2, index=idx[:14])
    y_valid = pd.Series(np.linspace(121, 127, 4) + np.array([0.8, 0.6, -0.2, 0.4]), index=idx[14:18])
    y_pred = pd.Series(np.linspace(121.5, 127.5, 4), index=idx[14:18])
    y_forecast = pd.Series(np.linspace(128, 132, 2), index=pd.date_range(idx[18], periods=2, freq="D"))
    fig = plot_forecast(idx, y_train, y_valid, y_pred, y_forecast)
    fig.suptitle("Forecast example", fontsize=14, fontweight="bold")
    return _save(fig, "forecast_example.png")


def make_notebook_export() -> Path:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")
    ax.set_title("Notebook export preview", fontsize=14, fontweight="bold", loc="left")
    lines = [
        "[1] Load data and detect types",
        "df = pd.read_csv('demo_quickstart.csv')",
        "summary = profile_dataframe(df)",
        "",
        "[2] Train best model (AutoML)",
        "best_model, metrics = run_best_model(df, target='target_class')",
        "print(metrics)  # {'roc_auc': 0.94, 'f1': 0.90, 'accuracy': 0.91}",
        "",
        "[3] Explain predictions",
        "global_exp = explain_global(best_model, df, feature_names=df.columns)",
        "plot_feature_importance(global_exp.feature_names, global_exp.importance_values)",
        "",
        "[4] Save artifacts",
        "best_model.save('artifacts/best_model.pkl')",
        "pd.DataFrame(metrics, index=[0]).to_csv('artifacts/metrics.csv', index=False)",
    ]
    y = 0.92
    for line in lines:
        ax.text(0.02, y, line, fontsize=11, family="monospace", va="top")
        y -= 0.055
    ax.add_patch(
        patches.FancyBboxPatch(
            (0.015, 0.08),
            0.97,
            0.82,
            boxstyle="round,pad=0.02",
            facecolor="#f7f9fb",
            edgecolor="#d0d7de",
            linewidth=1.2,
        )
    )
    return _save(fig, "notebook_export_cells.png")


def make_global_importance() -> Path:
    features = ["tenure_months", "income", "age", "contract_type", "region"]
    values = [0.42, 0.31, 0.18, 0.07, 0.05]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(features, values, color="#4c72b0")
    ax.invert_yaxis()
    ax.set_xlabel("Mean |importance|")
    ax.set_title("Global feature importance", fontsize=13, fontweight="bold")
    for f, v in zip(features, values):
        ax.text(v + 0.01, f, f"{v:.2f}", va="center")
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, "global_importance.png")


def make_scenario_testing() -> Path:
    base_pred = 0.42
    adjusted_pred = 0.58
    xs = np.linspace(0, 100, 50)
    ys = 0.3 + 0.0045 * xs
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(xs, ys, color="#4c72b0", lw=2)
    axes[0].axvline(60, color="#55a868", linestyle="--", label="Slider position")
    axes[0].set_xlabel("Feature value (e.g., tenure_months)")
    axes[0].set_ylabel("Predicted probability")
    axes[0].set_title("Partial dependence (Scenario)", fontsize=12, fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].axis("off")
    axes[1].add_patch(
        patches.FancyBboxPatch(
            (0.05, 0.35),
            0.9,
            0.3,
            boxstyle="round,pad=0.02",
            facecolor="#f7f9fb",
            edgecolor="#d0d7de",
            linewidth=1.2,
        )
    )
    axes[1].text(0.08, 0.58, "Scenario results", fontsize=13, fontweight="bold")
    axes[1].text(0.08, 0.50, f"Baseline prediction: {base_pred:.2f}", fontsize=11, color="#444")
    axes[1].text(0.08, 0.42, f"With slider change: {adjusted_pred:.2f}", fontsize=11, color="#444")
    axes[1].text(0.08, 0.34, f"Delta: {(adjusted_pred - base_pred):.2f}", fontsize=11, color="#55a868", fontweight="bold")
    axes[1].text(0.08, 0.24, "Tip: Try moving tenure_months lower to reduce risk.", fontsize=10, color="#666")
    return _save(fig, "scenario_testing.png")


def _slide_canvas(title: str, bullets: list[str], image_path: Path, name: str) -> Path:
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.axis("off")
    # Background panel
    ax.add_patch(
        patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color="#0f172a", zorder=0)
    )
    ax.add_patch(
        patches.Rectangle((0.52, 0.08), 0.44, 0.84, transform=ax.transAxes, color="#111827", zorder=1, alpha=0.85)
    )
    # Screenshot
    img = plt.imread(image_path)
    ax.imshow(img, extent=(0.54, 0.96, 0.12, 0.88), zorder=2)
    # Text
    ax.text(0.04, 0.88, title, fontsize=20, fontweight="bold", color="white", ha="left")
    for i, bullet in enumerate(bullets):
        ax.text(
            0.04,
            0.78 - i * 0.12,
            f"• {bullet}",
            fontsize=14,
            color="#e5e7eb",
            ha="left",
        )
    ax.text(0.04, 0.08, "Modeling-GUI — No-code ML & stats desktop app", fontsize=11, color="#cbd5e1")
    return _save(fig, name)


def make_promo_slides(smart_run: Path, leaderboard: Path, tuning: Path, metrics: Path, forecast: Path, notebook: Path) -> list[Path]:
    slides = [
        _slide_canvas(
            "Modeling-GUI: From CSV to insight",
            [
                "Load a CSV, click Smart Analyze, get metrics and plots instantly",
                "Transparent steps: preprocessing, model choices, and evaluation",
                "Works offline as a desktop app — perfect for analysts and students",
            ],
            smart_run,
            "demo_slide_01_intro.png",
        ),
        _slide_canvas(
            "Smart Analyze with AutoML",
            [
                "Detects problem type and runs tuned candidates for you",
                "Highlights primary metrics by domain (business, finance, science)",
                "One-click summary plus detailed \"What I did\" report",
            ],
            leaderboard,
            "demo_slide_02_smart.png",
        ),
        _slide_canvas(
            "Tune & compare models",
            [
                "Quick presets with sensible ranges for accuracy vs. speed",
                "Side-by-side leaderboard and feature importance plots",
                "Export comparisons and keep history in project files",
            ],
            tuning,
            "demo_slide_03_tuning.png",
        ),
        _slide_canvas(
            "Forecasting and classification visuals",
            [
                "Forecast overlays with validation/backtest metrics",
                "ROC/PR curves to communicate classification quality",
                "Coach bar guides you on what to improve next",
            ],
            metrics,
            "demo_slide_04_metrics.png",
        ),
        _slide_canvas(
            "Shareable outputs",
            [
                "Export notebook, reports, and bundles with artifacts",
                "Save/load .mgui projects for reproducibility",
                "Drift checks and snapshots to track model changes",
            ],
            notebook,
            "demo_slide_05_export.png",
        ),
    ]
    return slides


def maybe_build_video(slides: list[Path]) -> Path | None:
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found; skipping video generation.")
        return None
    list_file = SCREENSHOTS / "demo_slides.txt"
    with list_file.open("w", encoding="utf-8") as f:
        for slide in slides:
            f.write(f"file '{slide.as_posix()}'\n")
            f.write("duration 3\n")
        # Repeat last frame for pause at end
        f.write(f"file '{slides[-1].as_posix()}'\n")
    video_path = ASSETS / "demo_promo.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-vf",
        "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1",
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]
    subprocess.run(cmd, check=True)
    return video_path


def main() -> None:
    smart_run = make_smart_analyze_run()
    leaderboard = make_leaderboard()
    tuning = make_tuning()
    metrics = make_metrics_curves()
    forecast = make_forecast_plot()
    notebook = make_notebook_export()
    make_global_importance()
    make_scenario_testing()
    slides = make_promo_slides(smart_run, leaderboard, tuning, metrics, forecast, notebook)
    video = maybe_build_video(slides)
    print("Screenshots generated in:", SCREENSHOTS)
    if video:
        print("Promo video generated at:", video)


if __name__ == "__main__":
    main()
