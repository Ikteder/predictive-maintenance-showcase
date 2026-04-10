from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

INDEX_COLUMNS = ["unit_id", "cycle"]
SETTING_COLUMNS = [f"setting_{index}" for index in range(1, 4)]
SENSOR_COLUMNS = [f"sensor_{index}" for index in range(1, 22)]
SIGNAL_COLUMNS = SETTING_COLUMNS + SENSOR_COLUMNS

RAW_URLS = {
    "train": "https://raw.githubusercontent.com/jiaxiang-cheng/PyTorch-LSTM-for-RUL-Prediction/master/CMAPSSData/train_FD001.txt",
    "test": "https://raw.githubusercontent.com/jiaxiang-cheng/PyTorch-LSTM-for-RUL-Prediction/master/CMAPSSData/test_FD001.txt",
    "rul": "https://raw.githubusercontent.com/jiaxiang-cheng/PyTorch-LSTM-for-RUL-Prediction/master/CMAPSSData/RUL_FD001.txt",
}

WINDOW_SIZE = 30
WINDOW_STRIDE = 5
FAILURE_HORIZON = 30
ANOMALY_HORIZON = 20
HEALTHY_RUL_FLOOR = 100
RANDOM_STATE = 42


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    raw_dir: Path
    processed_dir: Path
    figures_dir: Path
    reports_dir: Path
    notebook_path: Path


def get_project_paths(project_root: Path) -> ProjectPaths:
    return ProjectPaths(
        root=project_root,
        raw_dir=project_root / "data" / "raw",
        processed_dir=project_root / "data" / "processed",
        figures_dir=project_root / "reports" / "figures",
        reports_dir=project_root / "reports",
        notebook_path=project_root / "notebooks" / "predictive_maintenance_analysis.ipynb",
    )
