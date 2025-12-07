"""
Reporting helpers for generating human-readable summaries and exporting reports.
"""

import json
from pathlib import Path
from typing import Dict, Any


def build_summary_text(summary: Dict[str, Any], metrics: Dict[str, Any]) -> str:
    """
    Convert structured summary and metrics into a human-readable multi-line text.
    """
    lines = []
    data = summary.get("data", {})
    preprocess = summary.get("preprocessing", {})
    modeling = summary.get("modeling", {})
    evaluation = summary.get("evaluation", {})

    lines.append("Data summary")
    lines.append(f"- Rows: {data.get('rows', '?')}, Columns: {data.get('cols', '?')}")
    lines.append(f"- Column types: {data.get('types', {})}")
    lines.append("")

    lines.append("Preprocessing steps")
    lines.append(f"- Missing values: {preprocess.get('missing', 'n/a')}")
    lines.append(f"- Scaling: {preprocess.get('scaling', 'n/a')}")
    lines.append(f"- Categorical handling: {preprocess.get('categoricals', 'n/a')}")
    lines.append("")

    lines.append("Modeling decisions")
    lines.append(f"- Problem type: {modeling.get('problem_type', 'n/a')}")
    lines.append(f"- Algorithms tried: {', '.join(modeling.get('algorithms', [])) or 'n/a'}")
    lines.append(f"- Best model: {modeling.get('best_model', 'n/a')}")
    lines.append(f"- CV: {modeling.get('cv', 'n/a')}")
    lines.append("")

    lines.append("Evaluation metrics")
    lines.append(f"- Train/test split: {evaluation.get('split', 'n/a')}")
    primary = evaluation.get("primary", {})
    secondary = evaluation.get("secondary", {})
    if primary:
        lines.append(f"- Primary metrics: {primary}")
    if secondary:
        lines.append(f"- Secondary metrics: {secondary}")
    if metrics:
        lines.append(f"- Raw metrics: {metrics}")

    return "\n".join(lines)


def export_report(path: str, summary_text: str) -> str:
    """
    Export a text/markdown report to the specified path.
    """
    dest = Path(path)
    dest.write_text(summary_text, encoding="utf-8")
    return str(dest.resolve())


def generate_narrative(summary: Dict[str, Any], metrics: Dict[str, Any], domain_preset: str) -> str:
    """
    Build a simple plain-language narrative of the analysis.
    """
    data = summary.get("data", {})
    modeling = summary.get("modeling", {})
    preprocessing = summary.get("preprocessing", {})
    evaluation = summary.get("evaluation", {})

    lines = []
    lines.append(f"# Modeling-GUI Narrative Report\n")
    lines.append(f"## Goal\nThis report summarizes an analysis for domain: **{domain_preset}**.\n")
    lines.append("## Data\n")
    lines.append(f"- Rows: {data.get('rows', '?')}, Columns: {data.get('cols', '?')}")
    lines.append(f"- Column types: {data.get('types', {})}\n")
    lines.append("## Preprocessing\n")
    lines.append(f"- Missing values: {preprocessing.get('missing', 'n/a')}")
    lines.append(f"- Scaling: {preprocessing.get('scaling', 'n/a')}")
    lines.append(f"- Categoricals: {preprocessing.get('categoricals', 'n/a')}\n")
    lines.append("## Modeling\n")
    lines.append(f"- Problem type: {modeling.get('problem_type', 'n/a')}")
    lines.append(f"- Algorithms tried: {', '.join(modeling.get('algorithms', [])) or 'n/a'}")
    lines.append(f"- Best model: {modeling.get('best_model', 'n/a')}")
    lines.append(f"- CV: {modeling.get('cv', 'n/a')}\n")
    lines.append("## Results\n")
    if metrics:
        for k, v in metrics.items():
            lines.append(f"- {k}: {v}")
    lines.append(f"- Train/Test split: {evaluation.get('split', 'n/a')}\n")
    lines.append("## Explainability\n")
    lines.append(f"- Top drivers: {modeling.get('algorithms', ['n/a'])[0] if modeling.get('algorithms') else 'n/a'} (see feature importances or SHAP if available)\n")
    lines.append("## Next Steps\n")
    lines.append("- Validate on new data or a hold-out set.\n- Consider feature engineering (dates, categoricals).\n- Rerun Smart Analyze with optional AutoML for alternatives.\n")
    return "\n".join(lines)
