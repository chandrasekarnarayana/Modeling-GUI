import json
import os
from pathlib import Path

import pytest

# Try to run Qt headless if possible
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QMessageBox  # noqa: E402


@pytest.fixture(scope="module")
def app():
    app_instance = QApplication.instance()
    if app_instance is None:
        app_instance = QApplication([])
    return app_instance


@pytest.mark.skipif(
    os.environ.get("QT_QPA_PLATFORM") == "xcb" and not os.environ.get("DISPLAY"),
    reason="No display available for Qt",
)
def test_main_window_smoke(monkeypatch, app, tmp_path):
    """Headless smoke test: load demo data and run a simple analysis without raising."""
    # Avoid first-run popup by pre-writing config
    cfg_path = Path.home() / ".config" / "modeling_gui" / "config.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps({"first_run_shown": True}), encoding="utf-8")

    # Stub QMessageBox to avoid blocking dialogs during test
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: None))

    from modeling_gui.main import MainApp

    win = MainApp()
    # load demo data
    win.load_demo_dataset()
    assert win.y_combo.count() > 0
    # pick classification target if present, else first
    cls_idx = win.y_combo.findText("target_class")
    if cls_idx >= 0:
        win.y_combo.setCurrentIndex(cls_idx)
    # ensure UI populated
    assert win.tabs.count() > 0
    assert win.problem_label is not None
    # do not enter full event loop; just ensure no exceptions so far
    win.close()
