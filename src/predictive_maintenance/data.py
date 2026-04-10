from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from .config import INDEX_COLUMNS, RAW_URLS, SENSOR_COLUMNS, SETTING_COLUMNS, SIGNAL_COLUMNS


def _download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def ensure_dataset(raw_dir: Path) -> dict[str, Path]:
    files = {
        "train": raw_dir / "train_FD001.txt",
        "test": raw_dir / "test_FD001.txt",
        "rul": raw_dir / "RUL_FD001.txt",
    }
    for key, path in files.items():
        if not path.exists():
            _download_file(RAW_URLS[key], path)
    return files


def _read_engine_file(path: Path) -> pd.DataFrame:
    columns = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=columns, engine="python")
    frame["unit_id"] = frame["unit_id"].astype(int)
    frame["cycle"] = frame["cycle"].astype(int)
    return frame


def _attach_train_rul(frame: pd.DataFrame) -> pd.DataFrame:
    max_cycles = frame.groupby("unit_id")["cycle"].transform("max")
    enriched = frame.copy()
    enriched["rul"] = max_cycles - enriched["cycle"]
    enriched["split"] = "train"
    return enriched


def _attach_test_rul(test_frame: pd.DataFrame, rul_path: Path) -> pd.DataFrame:
    truth = pd.read_csv(rul_path, header=None, names=["final_rul"])
    truth["unit_id"] = truth.index + 1
    observed_cycles = (
        test_frame.groupby("unit_id")["cycle"].max().rename("observed_max_cycle").reset_index()
    )
    merged = test_frame.merge(truth, on="unit_id", how="left").merge(observed_cycles, on="unit_id", how="left")
    merged["rul"] = merged["observed_max_cycle"] + merged["final_rul"] - merged["cycle"]
    merged["split"] = "test"
    return merged.drop(columns=["final_rul", "observed_max_cycle"])


def load_cmapss_fd001(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = ensure_dataset(raw_dir)
    train_frame = _attach_train_rul(_read_engine_file(files["train"]))
    test_frame = _attach_test_rul(_read_engine_file(files["test"]), files["rul"])
    return train_frame, test_frame


def rank_sensors(train_frame: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    correlations = []
    for column in SIGNAL_COLUMNS:
        variance = train_frame[column].var()
        if variance == 0 or train_frame[column].nunique(dropna=False) <= 1:
            correlation = 0.0
        else:
            correlation = train_frame[column].corr(train_frame["rul"], method="spearman")
        if pd.isna(correlation):
            correlation = 0.0
        if pd.isna(variance):
            variance = 0.0
        correlations.append(
            {
                "signal": column,
                "abs_spearman_corr_with_rul": abs(float(correlation)),
                "variance": float(variance),
            }
        )
    ranking = pd.DataFrame(correlations).sort_values(
        ["abs_spearman_corr_with_rul", "variance"], ascending=[False, False]
    )
    ranking["is_top_signal"] = False
    ranking.loc[ranking.head(top_n).index, "is_top_signal"] = True
    return ranking.reset_index(drop=True)
