"""
Cross-platform shortcut installer for Modeling-GUI.
"""

import os
import sys
import platform
from pathlib import Path


def _resource_path(*parts: str) -> str:
    base = Path(__file__).resolve().parent
    return str(base.joinpath("resources", *parts))


def install_shortcut() -> None:
    """
    Install a desktop/menu shortcut for the application.
    """
    system = platform.system()

    if system == "Windows":
        _install_windows_shortcut()
    elif system == "Linux":
        _install_linux_shortcut()
    elif system == "Darwin":
        _install_macos_shortcut()
    else:
        print(f"Unsupported OS for shortcut installation: {system}")


def _install_windows_shortcut() -> None:
    try:
        import winshell  # type: ignore
        from win32com.client import Dispatch  # type: ignore
    except ImportError:
        print("Please install pywin32 and winshell to create a Windows shortcut:")
        print("  pip install pywin32 winshell")
        return

    desktop = Path(os.path.expanduser("~")) / "Desktop"
    desktop.mkdir(parents=True, exist_ok=True)
    shortcut_path = desktop / "Modeling-GUI.lnk"

    python_exe = sys.executable
    target = python_exe
    arguments = "-m modeling_gui"
    icon_path = _resource_path("icons", "logo.ico")

    shell = Dispatch("WScript.Shell")
    shortcut = shell.CreateShortCut(str(shortcut_path))
    shortcut.TargetPath = target
    shortcut.Arguments = arguments
    shortcut.IconLocation = icon_path
    shortcut.WorkingDirectory = str(Path.home())
    shortcut.save()

    print(f"Windows shortcut created: {shortcut_path}")


def _install_linux_shortcut() -> None:
    home = Path(os.path.expanduser("~"))
    applications_dir = home / ".local" / "share" / "applications"
    applications_dir.mkdir(parents=True, exist_ok=True)

    desktop_file = applications_dir / "modeling-gui.desktop"
    icon_path = _resource_path("icons", "logo_256.png")
    if not os.path.exists(icon_path):
        icon_path = _resource_path("icons", "logo.png")

    exec_cmd = "run_modeling_gui"

    desktop_content = f"""[Desktop Entry]
Type=Application
Name=Modeling-GUI
Comment=No-code ML & stats GUI for CSV data
Exec={exec_cmd}
Icon={icon_path}
Terminal=false
Categories=Science;Utility;
"""
    desktop_file.write_text(desktop_content, encoding="utf-8")
    desktop_file.chmod(0o755)

    desktop_path = home / "Desktop"
    if desktop_path.exists():
        desktop_copy = desktop_path / "Modeling-GUI.desktop"
        desktop_copy.write_text(desktop_content, encoding="utf-8")
        desktop_copy.chmod(0o755)
        print(f"Linux desktop shortcut created: {desktop_copy}")

    print(f"Linux application shortcut created: {desktop_file}")


def _install_macos_shortcut() -> None:
    home = Path(os.path.expanduser("~"))
    desktop = home / "Desktop"
    desktop.mkdir(parents=True, exist_ok=True)

    script_path = desktop / "Modeling-GUI.command"
    script_content = """#!/bin/bash
# Simple launcher for Modeling-GUI
exec python3 -m modeling_gui
"""
    script_path.write_text(script_content, encoding="utf-8")
    script_path.chmod(0o755)

    print(f"macOS launcher script created: {script_path}")
    print("Tip: You can also build a native .app bundle using pyinstaller.")


if __name__ == "__main__":
    install_shortcut()
