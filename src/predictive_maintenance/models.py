from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier

from .config import HEALTHY_RUL_FLOOR, RANDOM_STATE
from .features import feature_columns


@dataclass
class ModelArtifacts:
    metrics: pd.DataFrame
    scored_test_windows: pd.DataFrame
    sensor_importance: pd.DataFrame
    alert_summary: pd.DataFrame


def _best_threshold(y_true: pd.Series, scores: pd.Series) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        return 0.5
    f1_scores = (2 * precision[:-1] * recall[:-1]) / np.clip(precision[:-1] + recall[:-1], 1e-9, None)
    best_index = int(np.nanargmax(f1_scores))
    return float(thresholds[best_index])


def _binary_metrics(y_true: pd.Series, scores: pd.Series, threshold: float) -> dict[str, float]:
    predictions = (scores >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "auroc": float(roc_auc_score(y_true, scores)),
        "threshold": float(threshold),
        "positive_rate": float(predictions.mean()),
    }


def _aggregate_signal_importance(model_name: str, feature_names: list[str], importances: np.ndarray) -> pd.DataFrame:
    signal_scores: dict[str, float] = {}
    for feature_name, importance in zip(feature_names, importances, strict=False):
        signal = feature_name.rsplit("_", 1)[0]
        signal_scores[signal] = signal_scores.get(signal, 0.0) + float(importance)
    total = sum(signal_scores.values()) or 1.0
    rows = [
        {"model": model_name, "signal": signal, "importance": score / total}
        for signal, score in signal_scores.items()
    ]
    return pd.DataFrame(rows)


def _lead_time_summary(scored_windows: pd.DataFrame, model_name: str, threshold: float) -> pd.DataFrame:
    alerts = []
    for unit_id, engine_frame in scored_windows.groupby("unit_id", sort=True):
        model_scores = engine_frame[f"{model_name}_score"]
        engine_alerts = engine_frame.loc[model_scores >= threshold]
        earliest_alert = engine_alerts.iloc[0] if not engine_alerts.empty else None
        alerts.append(
            {
                "model": model_name,
                "unit_id": int(unit_id),
                "true_failure_window_present": int(engine_frame["failure_within_30"].max()),
                "alerted": int(earliest_alert is not None),
                "first_alert_cycle": int(earliest_alert["end_cycle"]) if earliest_alert is not None else None,
                "lead_time_cycles": int(earliest_alert["true_rul"]) if earliest_alert is not None else None,
                "max_failure_risk_score": float(model_scores.max()),
            }
        )
    return pd.DataFrame(alerts)


def train_and_evaluate(train_windows: pd.DataFrame, test_windows: pd.DataFrame) -> ModelArtifacts:
    features = feature_columns(train_windows)
    splitter = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=RANDOM_STATE)
    fit_index, validation_index = next(
        splitter.split(train_windows[features], train_windows["failure_within_30"], groups=train_windows["unit_id"])
    )
    fit_windows = train_windows.iloc[fit_index].reset_index(drop=True)
    validation_windows = train_windows.iloc[validation_index].reset_index(drop=True)

    x_fit = fit_windows[features]
    x_validation = validation_windows[features]
    x_test = test_windows[features]
    y_validation_failure = validation_windows["failure_within_30"]
    y_test_failure = test_windows["failure_within_30"]
    y_validation_anomaly = validation_windows["anomaly_within_20"]
    y_test_anomaly = test_windows["anomaly_within_20"]

    scaler = StandardScaler()
    x_fit_scaled = scaler.fit_transform(x_fit)
    x_validation_scaled = scaler.transform(x_validation)
    x_test_scaled = scaler.transform(x_test)
    x_healthy_scaled = x_fit_scaled[(fit_windows["true_rul"] >= HEALTHY_RUL_FLOOR).to_numpy()]

    isolation_forest = IsolationForest(n_estimators=400, contamination=0.08, random_state=RANDOM_STATE)
    isolation_forest.fit(x_healthy_scaled)
    isolation_validation_scores = -isolation_forest.decision_function(x_validation_scaled)
    isolation_threshold = _best_threshold(y_validation_anomaly, pd.Series(isolation_validation_scores))
    isolation_test_scores = -isolation_forest.decision_function(x_test_scaled)

    one_class_svm = OneClassSVM(gamma="scale", kernel="rbf", nu=0.08)
    one_class_svm.fit(x_healthy_scaled)
    ocsvm_validation_scores = -one_class_svm.decision_function(x_validation_scaled)
    ocsvm_threshold = _best_threshold(y_validation_anomaly, pd.Series(ocsvm_validation_scores))
    ocsvm_test_scores = -one_class_svm.decision_function(x_test_scaled)

    random_forest = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    random_forest.fit(x_fit, fit_windows["failure_within_30"])
    rf_validation_scores = random_forest.predict_proba(x_validation)[:, 1]
    rf_threshold = _best_threshold(y_validation_failure, pd.Series(rf_validation_scores))
    rf_test_scores = random_forest.predict_proba(x_test)[:, 1]

    xgboost = XGBClassifier(
        objective="binary:logistic",
        n_estimators=350,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=2,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=1,
    )
    xgboost.fit(x_fit, fit_windows["failure_within_30"])
    xgb_validation_scores = xgboost.predict_proba(x_validation)[:, 1]
    xgb_threshold = _best_threshold(y_validation_failure, pd.Series(xgb_validation_scores))
    xgb_test_scores = xgboost.predict_proba(x_test)[:, 1]

    metric_rows = [
        {
            "model": "Isolation Forest",
            "task": "anomaly_detection",
            **_binary_metrics(y_test_anomaly, pd.Series(isolation_test_scores), isolation_threshold),
        },
        {
            "model": "One-Class SVM",
            "task": "anomaly_detection",
            **_binary_metrics(y_test_anomaly, pd.Series(ocsvm_test_scores), ocsvm_threshold),
        },
        {
            "model": "Random Forest",
            "task": "failure_classification",
            **_binary_metrics(y_test_failure, pd.Series(rf_test_scores), rf_threshold),
        },
        {
            "model": "XGBoost",
            "task": "failure_classification",
            **_binary_metrics(y_test_failure, pd.Series(xgb_test_scores), xgb_threshold),
        },
    ]
    metrics = pd.DataFrame(metric_rows).sort_values(["task", "f1"], ascending=[True, False]).reset_index(drop=True)

    scored_test_windows = test_windows.copy()
    scored_test_windows["Isolation Forest_score"] = isolation_test_scores
    scored_test_windows["One-Class SVM_score"] = ocsvm_test_scores
    scored_test_windows["Random Forest_score"] = rf_test_scores
    scored_test_windows["XGBoost_score"] = xgb_test_scores

    sensor_importance = pd.concat(
        [
            _aggregate_signal_importance("Random Forest", features, random_forest.feature_importances_),
            _aggregate_signal_importance("XGBoost", features, xgboost.feature_importances_),
        ],
        ignore_index=True,
    )
    sensor_importance = (
        sensor_importance.groupby("signal", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    thresholds = {
        "Isolation Forest": isolation_threshold,
        "One-Class SVM": ocsvm_threshold,
        "Random Forest": rf_threshold,
        "XGBoost": xgb_threshold,
    }
    alert_summary = pd.concat(
        [_lead_time_summary(scored_test_windows, model_name, threshold) for model_name, threshold in thresholds.items()],
        ignore_index=True,
    )
    return ModelArtifacts(
        metrics=metrics,
        scored_test_windows=scored_test_windows,
        sensor_importance=sensor_importance,
        alert_summary=alert_summary,
    )
