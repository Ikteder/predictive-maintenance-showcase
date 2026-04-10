from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ANOMALY_HORIZON, FAILURE_HORIZON, SIGNAL_COLUMNS, WINDOW_SIZE, WINDOW_STRIDE


def _window_statistics(values: np.ndarray) -> dict[str, float]:
    x_axis = np.arange(values.shape[0])
    slope = float(np.polyfit(x_axis, values, 1)[0]) if values.shape[0] > 1 else 0.0
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "last": float(values[-1]),
        "delta": float(values[-1] - values[0]),
        "slope": slope,
    }


def build_window_features(
    frame: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    stride: int = WINDOW_STRIDE,
) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for unit_id, engine_frame in frame.groupby("unit_id", sort=True):
        engine_frame = engine_frame.sort_values("cycle").reset_index(drop=True)
        if len(engine_frame) < window_size:
            continue
        window_end_indexes = list(range(window_size - 1, len(engine_frame), stride))
        if window_end_indexes[-1] != len(engine_frame) - 1:
            window_end_indexes.append(len(engine_frame) - 1)
        for end_index in window_end_indexes:
            window = engine_frame.iloc[end_index - window_size + 1 : end_index + 1]
            feature_row: dict[str, float | int | str] = {
                "unit_id": int(unit_id),
                "split": str(engine_frame["split"].iat[-1]),
                "end_cycle": int(window["cycle"].iat[-1]),
                "true_rul": int(window["rul"].iat[-1]),
            }
            for column in SIGNAL_COLUMNS:
                statistics = _window_statistics(window[column].to_numpy(dtype=float))
                for stat_name, stat_value in statistics.items():
                    feature_row[f"{column}_{stat_name}"] = stat_value
            feature_row["failure_within_30"] = int(feature_row["true_rul"] <= FAILURE_HORIZON)
            feature_row["anomaly_within_20"] = int(feature_row["true_rul"] <= ANOMALY_HORIZON)
            records.append(feature_row)
    return pd.DataFrame.from_records(records)


def feature_columns(window_frame: pd.DataFrame) -> list[str]:
    excluded = {"unit_id", "split", "end_cycle", "true_rul", "failure_within_30", "anomaly_within_20"}
    return [column for column in window_frame.columns if column not in excluded]


def lifecycle_trend_frame(frame: pd.DataFrame, sensors: list[str], bins: int = 20) -> pd.DataFrame:
    enriched = frame.copy()
    max_cycle = enriched.groupby("unit_id")["cycle"].transform("max")
    enriched["lifecycle_pct"] = enriched["cycle"] / max_cycle
    enriched["lifecycle_bin"] = np.minimum((enriched["lifecycle_pct"] * bins).astype(int), bins - 1)
    melted = enriched.melt(
        id_vars=["unit_id", "cycle", "lifecycle_bin"],
        value_vars=sensors,
        var_name="signal",
        value_name="value",
    )
    melted["z_value"] = melted.groupby("signal")["value"].transform(
        lambda series: (series - series.mean()) / series.std(ddof=0)
    )
    return (
        melted.groupby(["lifecycle_bin", "signal"], as_index=False)["z_value"]
        .mean()
        .rename(columns={"z_value": "mean_z_value"})
    )


def save_window_snapshot(window_frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    preview = window_frame.head(25)
    preview.to_csv(destination, index=False)
