# Shortcuts & Installation

## CLI entry points

- `run_modeling_gui` – launch the GUI.
- `python -m modeling_gui` – alternate launcher.
- `run_modeling_gui --demo` (if supported) – open with bundled demo dataset.
- `modeling-gui-install-shortcut` – create a desktop/menu shortcut with the app icon.

## Desktop shortcuts

### Windows

- Run `modeling-gui-install-shortcut` in PowerShell or CMD.
- Creates `Modeling-GUI.lnk` on the Desktop (uses packaged `.ico`).
- If `pywin32`/`winshell` are missing, the command will print how to install them.

### Linux

- Run `modeling-gui-install-shortcut`.
- Installs `~/.local/share/applications/modeling-gui.desktop` (and copies to `~/Desktop` if present).
- Uses the packaged PNG icon; launched via `run_modeling_gui`.

### macOS

- Run `modeling-gui-install-shortcut`.
- Creates `Modeling-GUI.command` on the Desktop that calls `python3 -m modeling_gui`.
- For a full `.app` bundle, use PyInstaller or similar (out of scope here).

## Keyboard shortcuts (in-app)

Access: **Help → Keyboard Shortcuts…** (also shown on first run). F1 opens the online docs.

| Shortcut | Action |
|----------|--------|
| Ctrl+O | Open CSV |
| Ctrl+S | Save project |
| Ctrl+Shift+S | Save model |
| F5 | Smart Analyze |
| Ctrl+R | Run selected model |
| Ctrl+Q | Quit |
| F1 | Open documentation |

## Tips

- First run may suggest creating a shortcut; it’s optional.
- Icons are packaged under `modeling_gui/resources/icons/` and are used automatically for the main window.
