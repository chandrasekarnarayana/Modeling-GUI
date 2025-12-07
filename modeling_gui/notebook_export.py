"""Generate a reproducible notebook for a saved project/model."""

from typing import Dict, List

try:
    import nbformat as nbf
except ImportError:  # pragma: no cover - optional dependency
    nbf = None


def _code_cell(code: str):
    return nbf.v4.new_code_cell(code.strip("\n"))


def export_notebook(path: str, project: Dict) -> str:
    """
    Build a simple notebook that reloads data, applies preprocessing, trains/predicts, and computes metrics.
    """
    if nbf is None:
        raise ImportError("nbformat is required for notebook export. Install via `pip install nbformat`.")

    x_cols: List[str] = project.get("x_columns", []) or []
    y_col = project.get("y_column", "target")
    data_path = project.get("data_path", "data.csv")
    metrics = project.get("metrics", {})

    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell("# Modeling-GUI Reproducible Notebook"))
    cells.append(
        nbf.v4.new_code_cell(
            "import pandas as pd\nimport numpy as np\nfrom joblib import load\nfrom sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error\n"
        )
    )

    cells.append(_code_cell(f"df = pd.read_csv('{data_path}')\ndf.head()"))

    cells.append(
        _code_cell(
            "# Simple preprocessing: fill missing numeric with mean\n"
            "df_proc = df.copy()\n"
            "for col in df_proc.select_dtypes(include=['number']).columns:\n    df_proc[col] = df_proc[col].fillna(df_proc[col].mean())"
        )
    )

    cells.append(
        _code_cell(
            f"x_cols = {x_cols}\n"
            f"y_col = '{y_col}'\n"
            "X = df_proc[x_cols]\n"
            "y = df_proc[y_col] if y_col in df_proc else None"
        )
    )

    cells.append(
        _code_cell(
            "model = load('model.joblib')\n"
            "preds = model.predict(X)\n"
            "preds[:10]"
        )
    )

    cells.append(
        _code_cell(
            "# Basic metrics (conditional)\n"
            "if y is not None:\n"
            "    if y.dtype.kind in 'if':\n"
            "        rmse = np.sqrt(mean_squared_error(y, preds))\n        print('RMSE', rmse)\n"
            "    else:\n"
            "        acc = accuracy_score(y, preds)\n        f1 = f1_score(y, preds, average='macro')\n        print('Accuracy', acc)\n        print('Macro F1', f1)"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell("## Notes\n- Metrics exported from GUI: {}".format(metrics))
    )

    nb["cells"] = cells
    nbf.write(nb, path)
    return path
