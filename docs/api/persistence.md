# API: persistence & bundles

- Model persistence:
  - `save_model_bundle(path, model, metadata)`
  - `load_model_bundle(path)`
- Project persistence:
  - `save_project(path, project_data)`
  - `load_project(path)`
- Bundles:
  - `export_bundle(project_data, path)` → zip containing project + artifacts
- Notebook export:
  - `export_notebook(path, project)`
