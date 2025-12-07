"""
Lightweight model card definitions and helpers.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
import uuid
import json
from pathlib import Path


@dataclass
class ModelCard:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Model"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    problem_type: str = ""
    domain: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    data_summary: Dict[str, Any] = field(default_factory=dict)
    explainability_summary: str = ""
    notes: str = ""
    model_obj: Optional[Any] = None


def serialize_model_card(card: ModelCard) -> Dict[str, Any]:
    return {
        "id": card.id,
        "name": card.name,
        "created_at": card.created_at,
        "problem_type": card.problem_type,
        "domain": card.domain,
        "metrics": card.metrics,
        "data_summary": card.data_summary,
        "explainability_summary": card.explainability_summary,
        "notes": card.notes,
    }


def export_model_card(card: ModelCard, path: str) -> str:
    """Export a model card to Markdown or JSON."""
    out_path = Path(path)
    data = serialize_model_card(card)
    if out_path.suffix.lower() == ".json":
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        # default to markdown
        lines = [
            f"# Model Card: {card.name}",
            "",
            f"- ID: {card.id}",
            f"- Created: {card.created_at}",
            f"- Problem type: {card.problem_type}",
            f"- Domain: {card.domain}",
            "",
            "## Metrics",
        ]
        for k, v in card.metrics.items():
            lines.append(f"- {k}: {v}")
        lines.extend(
            [
                "",
                "## Data summary",
                json.dumps(card.data_summary, indent=2),
                "",
                "## Explainability summary",
                card.explainability_summary or "n/a",
                "",
                "## Notes",
                card.notes or "n/a",
            ]
        )
        out_path.write_text("\n".join(lines), encoding="utf-8")
    return str(out_path)
