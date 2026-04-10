from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import get_project_paths
from .data import load_cmapss_fd001, rank_sensors
from .features import build_window_features, lifecycle_trend_frame, save_window_snapshot
from .models import train_and_evaluate
from .reporting import (
    plot_early_warning_heatmap,
    plot_model_comparison,
    plot_pca_projection,
    plot_precision_recall_curves,
    plot_sensor_correlation,
    plot_sensor_trends,
    plot_showcase_banner,
    plot_signal_importance,
    write_summary_report,
)


def run_full_pipeline(project_root: Path | None = None) -> dict[str, pd.DataFrame]:
    root = Path(project_root or Path(__file__).resolve().parents[2])
    paths = get_project_paths(root)
    for directory in [paths.raw_dir, paths.processed_dir, paths.figures_dir, paths.reports_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    train_frame, test_frame = load_cmapss_fd001(paths.raw_dir)
    ranking = rank_sensors(train_frame, top_n=8)
    top_signals = ranking.loc[ranking["is_top_signal"], "signal"].tolist()
    focus_signals = top_signals[:3]

    train_windows = build_window_features(train_frame)
    test_windows = build_window_features(test_frame)
    artifacts = train_and_evaluate(train_windows, test_windows)
    trend_frame = lifecycle_trend_frame(train_frame, focus_signals)

    artifacts.metrics.to_csv(paths.processed_dir / "model_metrics.csv", index=False)
    ranking.to_csv(paths.processed_dir / "sensor_signal_ranking.csv", index=False)
    artifacts.sensor_importance.to_csv(paths.processed_dir / "sensor_importance.csv", index=False)
    artifacts.alert_summary.to_csv(paths.processed_dir / "early_warning_alerts.csv", index=False)
    save_window_snapshot(artifacts.scored_test_windows, paths.processed_dir / "test_window_predictions_preview.csv")

    plot_showcase_banner(artifacts.metrics, artifacts.alert_summary, paths.figures_dir / "showcase_banner.png")
    plot_sensor_correlation(train_frame, top_signals, paths.figures_dir / "sensor_correlation_heatmap.png")
    plot_pca_projection(train_frame, top_signals, paths.figures_dir / "pca_projection.png")
    plot_sensor_trends(trend_frame, paths.figures_dir / "sensor_trends.png")
    plot_model_comparison(artifacts.metrics, paths.figures_dir / "model_comparison.png")
    plot_precision_recall_curves(artifacts.scored_test_windows, paths.figures_dir / "precision_recall_curves.png")
    plot_signal_importance(artifacts.sensor_importance, paths.figures_dir / "sensor_importance.png")
    plot_early_warning_heatmap(artifacts.alert_summary, paths.figures_dir / "early_warning_heatmap.png")
    write_summary_report(
        artifacts.metrics,
        artifacts.sensor_importance,
        artifacts.alert_summary,
        paths.reports_dir / "early_warning_summary.md",
    )

    return {
        "metrics": artifacts.metrics,
        "ranking": ranking,
        "sensor_importance": artifacts.sensor_importance,
        "alerts": artifacts.alert_summary,
        "train_windows": train_windows,
        "test_windows": artifacts.scored_test_windows,
    }
