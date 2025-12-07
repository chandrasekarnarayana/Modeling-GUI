import os
import tempfile
from modeling_gui.bundles import export_bundle


def test_export_bundle_creates_zip():
    project = {"summary": {}, "reports": {"report.txt": "hello"}}
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
        path = tmp.name
    try:
        out = export_bundle(project, path)
        assert os.path.exists(out)
    finally:
        if os.path.exists(path):
            os.remove(path)
