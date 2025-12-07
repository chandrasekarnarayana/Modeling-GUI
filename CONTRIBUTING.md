# Contributing

Thanks for considering contributing to Modeling-GUI!

## Setup

```bash
git clone https://github.com/chandrasekarnarayana/Modeling-GUI
cd Modeling-GUI
pip install -e .[dev]
```

## Workflow

- Create a feature branch (`git checkout -b feature/your-feature`).
- Keep changes small and focused; add tests for new logic.
- Run checks locally:
  - `ruff check modeling_gui`
  - `black --check modeling_gui`
  - `pytest`
- Update docs (MkDocs) and changelog when adding features.
- Open a pull request with a clear description and screenshots if UI changes.

## Style & Structure

- Code: black formatting, ruff linting.
- Tests: use pytest; keep GUI-dependent tests minimal/headless.
- Docs: add/update under `docs/` and mkdocs nav.

## Questions

Open an issue on GitHub with details about your environment and expected vs actual behavior.
