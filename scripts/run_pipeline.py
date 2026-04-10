from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from predictive_maintenance.pipeline import run_full_pipeline


if __name__ == "__main__":
    outputs = run_full_pipeline(PROJECT_ROOT)
    metrics = outputs["metrics"]
    print("\nPipeline complete. Top model summary:\n")
    print(metrics[["model", "task", "f1", "average_precision", "auroc"]].to_string(index=False))

