from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.preprocessing import StandardScaler


matplotlib.use("Agg")
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette(["#0F4C81", "#E07A5F", "#2A9D8F", "#F4A261", "#264653"])


def _save_figure(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_sensor_correlation(frame: pd.DataFrame, signals: list[str], output_path: Path) -> None:
    plt.figure(figsize=(9, 7))
    correlation = frame[signals].corr()
    sns.heatmap(correlation, cmap="coolwarm", center=0, square=True)
    plt.title("Top Signal Correlation Map")
    _save_figure(output_path)


def plot_pca_projection(frame: pd.DataFrame, signals: list[str], output_path: Path) -> None:
    sample = frame[signals + ["rul"]].sample(n=min(4000, len(frame)), random_state=42)
    scaled = StandardScaler().fit_transform(sample[signals])
    embedding = PCA(n_components=2, random_state=42).fit_transform(scaled)
    pca_frame = pd.DataFrame(embedding, columns=["pc1", "pc2"])
    pca_frame["rul_bucket"] = pd.cut(
        sample["rul"],
        bins=[-1, 15, 30, 60, 120, 999],
        labels=["0-15", "16-30", "31-60", "61-120", "120+"],
    )
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=pca_frame, x="pc1", y="pc2", hue="rul_bucket", alpha=0.65, s=35)
    plt.title("PCA Projection of Multivariate Sensor State")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    _save_figure(output_path)


def plot_sensor_trends(trend_frame: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=trend_frame, x="lifecycle_bin", y="mean_z_value", hue="signal", marker="o")
    plt.title("Sensor Drift Across the Engine Lifecycle")
    plt.xlabel("Lifecycle Bucket")
    plt.ylabel("Average z-scored sensor value")
    _save_figure(output_path)


def plot_model_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    score_frame = metrics.melt(
        id_vars=["model", "task"],
        value_vars=["f1", "average_precision", "auroc"],
        var_name="metric",
        value_name="score",
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=score_frame, x="score", y="model", hue="metric")
    plt.xlim(0, 1.0)
    plt.title("Model Comparison on Holdout Engines")
    plt.xlabel("Score")
    plt.ylabel("")
    _save_figure(output_path)


def plot_precision_recall_curves(scored_windows: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(9, 7))
    model_to_target = {
        "Isolation Forest": "anomaly_within_20",
        "One-Class SVM": "anomaly_within_20",
        "Random Forest": "failure_within_30",
        "XGBoost": "failure_within_30",
    }
    for model_name, target_column in model_to_target.items():
        PrecisionRecallDisplay.from_predictions(
            scored_windows[target_column],
            scored_windows[f"{model_name}_score"],
            name=model_name,
            ax=plt.gca(),
        )
    plt.title("Precision-Recall Curves")
    _save_figure(output_path)


def plot_signal_importance(signal_importance: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    top_signals = signal_importance.head(10).sort_values("importance")
    sns.barplot(data=top_signals, x="importance", y="signal", hue="signal", palette="crest", legend=False)
    plt.title("Most Informative Sensors and Operating Settings")
    plt.xlabel("Mean normalized importance")
    plt.ylabel("")
    _save_figure(output_path)


def plot_early_warning_heatmap(alert_summary: pd.DataFrame, output_path: Path) -> None:
    summary = (
        alert_summary.groupby("model", as_index=False)
        .agg(alert_rate=("alerted", "mean"), median_lead_time=("lead_time_cycles", "median"))
        .fillna(0)
    )
    heatmap_frame = summary.set_index("model")[["alert_rate", "median_lead_time"]]
    plt.figure(figsize=(8, 4))
    sns.heatmap(heatmap_frame, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Early Warning Coverage and Lead Time")
    _save_figure(output_path)


def plot_showcase_banner(metrics: pd.DataFrame, alert_summary: pd.DataFrame, output_path: Path) -> None:
    best_classifier = metrics.loc[metrics["task"] == "failure_classification"].sort_values("f1", ascending=False).iloc[0]
    best_anomaly = metrics.loc[metrics["task"] == "anomaly_detection"].sort_values("f1", ascending=False).iloc[0]
    lead_time = (
        alert_summary.loc[alert_summary["model"] == best_classifier["model"], "lead_time_cycles"]
        .dropna()
        .median()
    )
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.set_facecolor("#102542")
    figure.patch.set_facecolor("#102542")
    axis.axis("off")
    axis.text(0.03, 0.78, "Predictive Maintenance Showcase", fontsize=24, color="white", fontweight="bold")
    axis.text(0.03, 0.52, "NASA C-MAPSS turbofan degradation | anomaly detection + failure risk modeling", fontsize=12, color="#C7D3DD")
    axis.text(0.03, 0.24, f"Best classifier: {best_classifier['model']}  F1={best_classifier['f1']:.2f}", fontsize=14, color="#FFD166")
    axis.text(0.53, 0.24, f"Best anomaly model: {best_anomaly['model']}  F1={best_anomaly['f1']:.2f}", fontsize=14, color="#80ED99")
    axis.text(0.03, 0.08, f"Median early-warning lead time: {lead_time:.0f} cycles", fontsize=13, color="#F8F9FA")
    _save_figure(output_path)


def write_summary_report(
    metrics: pd.DataFrame,
    ranking: pd.DataFrame,
    alert_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    classifier_row = metrics.loc[metrics["task"] == "failure_classification"].sort_values("f1", ascending=False).iloc[0]
    anomaly_row = metrics.loc[metrics["task"] == "anomaly_detection"].sort_values("f1", ascending=False).iloc[0]
    top_signals = ", ".join(ranking.head(5)["signal"].tolist())
    median_lead_time = (
        alert_summary.loc[alert_summary["model"] == classifier_row["model"], "lead_time_cycles"]
        .dropna()
        .median()
    )
    report = f"""# Early Warning Summary Report

## Executive summary

- Best supervised failure classifier: **{classifier_row['model']}** with F1 **{classifier_row['f1']:.2f}**, AP **{classifier_row['average_precision']:.2f}**, and AUROC **{classifier_row['auroc']:.2f}**.
- Best unsupervised anomaly detector: **{anomaly_row['model']}** with F1 **{anomaly_row['f1']:.2f}**, AP **{anomaly_row['average_precision']:.2f}**, and AUROC **{anomaly_row['auroc']:.2f}**.
- Median early warning lead time for the top classifier: **{median_lead_time:.0f} cycles** before the 30-cycle failure horizon.
- Most informative signals: **{top_signals}**.

## Operational interpretation

- Sensors with the strongest degradation patterns consistently drift away from their healthy operating baseline as engines approach failure.
- The supervised models are better suited for explicit maintenance triage when historical failure labels exist.
- The anomaly detectors still add value for cold-start monitoring because they flag departures from the healthy manifold without needing failure labels.

## Recommendation

- Use the supervised classifier for scheduled maintenance planning.
- Use the anomaly model as an always-on shadow monitor to escalate unusual behavior sooner.
- Review the top-signal trend plots during root-cause analysis to explain why the alert fired.
"""
    output_path.write_text(report, encoding="utf-8")
