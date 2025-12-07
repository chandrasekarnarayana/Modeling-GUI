# Development

## Setup

```bash
git clone https://github.com/chandrasekarnarayana/Modeling-GUI
cd Modeling-GUI
pip install -e .[dev]
```

## Commands

- Lint: `ruff check modeling_gui`
- Format check: `black --check modeling_gui`
- Tests: `pytest`
- Docs: `mkdocs serve` (live preview)
- Matplotlib backend: tests use `Agg` (headless). Plots are generated but not shown; to inspect, run plotting code in a notebook or script locally.

## Structure

- `modeling_gui/` modules: automl, explain, preprocessing, data_quality, drift, visualization, etc.
- GUI entry: `modeling_gui/main.py`
- Projects/reports/bundles: `utils/project.py`, `reporting.py`, `bundles.py`, `notebook_export.py`
- Tests: `tests/`
- Docs: `docs/`

## Contributions

- Follow black + ruff style.
- Add tests for new logic; keep GUI tests headless where possible.
- Update docs and changelog when adding features.
