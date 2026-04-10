from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def build_notebook() -> None:
    metrics = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "model_metrics.csv")
    ranking = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "sensor_signal_ranking.csv").head(10)
    importance = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "sensor_importance.csv").head(10)
    report_text = (PROJECT_ROOT / "reports" / "early_warning_summary.md").read_text(encoding="utf-8")

    notebook = nbf.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.cells = [
        nbf.v4.new_markdown_cell(
            "# Predictive Maintenance and Anomaly Detection\n"
            "This notebook accompanies the showcase project built on the NASA C-MAPSS FD001 turbofan degradation dataset."
        ),
        nbf.v4.new_markdown_cell(
            "## What this project demonstrates\n"
            "- Multivariate sensor analysis on real-world turbofan telemetry\n"
            "- Temporal window feature engineering\n"
            "- Unsupervised anomaly detection vs. supervised failure risk prediction\n"
            "- Interpretability through sensor importance and trend analysis\n"
            "- Operational early-warning reporting"
        ),
        nbf.v4.new_markdown_cell(
            "## Reproducibility\n"
            "```python\n"
            "from pathlib import Path\n"
            "import sys\n"
            "project_root = Path('../').resolve()\n"
            "sys.path.insert(0, str(project_root / 'src'))\n"
            "from predictive_maintenance.pipeline import run_full_pipeline\n"
            "outputs = run_full_pipeline(project_root)\n"
            "outputs['metrics']\n"
            "```"
        ),
        nbf.v4.new_markdown_cell("## Model scorecard\n" + metrics.to_html(index=False)),
        nbf.v4.new_markdown_cell("![Showcase banner](../reports/figures/showcase_banner.png)"),
        nbf.v4.new_markdown_cell("## Sensor analysis"),
        nbf.v4.new_markdown_cell("### Most correlated signals with remaining useful life\n" + ranking.to_html(index=False)),
        nbf.v4.new_markdown_cell("![Correlation heatmap](../reports/figures/sensor_correlation_heatmap.png)"),
        nbf.v4.new_markdown_cell("![PCA projection](../reports/figures/pca_projection.png)"),
        nbf.v4.new_markdown_cell("![Sensor trends](../reports/figures/sensor_trends.png)"),
        nbf.v4.new_markdown_cell("## Model comparison"),
        nbf.v4.new_markdown_cell("![Model comparison](../reports/figures/model_comparison.png)"),
        nbf.v4.new_markdown_cell("![Precision recall curves](../reports/figures/precision_recall_curves.png)"),
        nbf.v4.new_markdown_cell("## Interpretability"),
        nbf.v4.new_markdown_cell("### Aggregated feature importance by signal\n" + importance.to_html(index=False)),
        nbf.v4.new_markdown_cell("![Signal importance](../reports/figures/sensor_importance.png)"),
        nbf.v4.new_markdown_cell("![Early warning heatmap](../reports/figures/early_warning_heatmap.png)"),
        nbf.v4.new_markdown_cell("## Early-warning summary\n" + report_text.replace("\n", "  \n")),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n\n"
            "project_root = Path('../').resolve()\n"
            "metrics = pd.read_csv(project_root / 'data' / 'processed' / 'model_metrics.csv')\n"
            "metrics"
        ),
    ]
    output_path = PROJECT_ROOT / "notebooks" / "predictive_maintenance_analysis.ipynb"
    output_path.write_text(nbf.writes(notebook), encoding="utf-8")


if __name__ == "__main__":
    build_notebook()
