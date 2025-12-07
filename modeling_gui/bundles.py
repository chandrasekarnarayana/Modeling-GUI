"""
Simple bundle exporter to zip up project artifacts.
"""
import json
import zipfile
from pathlib import Path
from typing import Dict


def export_bundle(project_data: Dict, path: str) -> str:
    """
    Create a zip file containing the .mgui JSON and any extra artifacts present in project_data.
    """
    dest = Path(path)
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("project.mgui", json.dumps(project_data, indent=2))
        if "reports" in project_data:
            for name, content in project_data["reports"].items():
                zf.writestr(f"reports/{name}", content)
    return str(dest.resolve())
