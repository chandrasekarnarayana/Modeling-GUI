import tempfile
import json
from pathlib import Path

from modeling_gui.model_cards import ModelCard, export_model_card
from modeling_gui.utils.project import save_project, load_project


def test_model_card_roundtrip(tmp_path: tempfile.TemporaryDirectory):
    card = ModelCard(name="BestModel", problem_type="regression", domain="Generic", metrics={"R2": 0.9})
    project = {"model_cards": [card.__dict__]}
    path = tmp_path / "proj.mgui"
    save_project(path, project)
    loaded = load_project(path)
    assert loaded["model_cards"][0]["name"] == "BestModel"
    assert loaded["model_cards"][0]["metrics"]["R2"] == 0.9


def test_model_card_export(tmp_path: Path):
    card = ModelCard(
        name="ExportModel",
        problem_type="classification",
        domain="Business",
        metrics={"accuracy": 0.95},
        data_summary={"rows": 20},
        notes="export check",
    )
    md_path = tmp_path / "card.md"
    json_path = tmp_path / "card.json"
    export_model_card(card, md_path)
    export_model_card(card, json_path)
    assert md_path.exists()
    assert json_path.exists()
    data = json.loads(json_path.read_text())
    assert data["name"] == "ExportModel"
    assert data["metrics"]["accuracy"] == 0.95
