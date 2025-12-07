#!/usr/bin/env python3
"""
demo_setup.py – Prepare a demo dataset and optionally launch Modeling-GUI with it preloaded.

Usage:
  python docs/scripts/demo_setup.py            # prepare dataset only
  python docs/scripts/demo_setup.py --launch   # prepare dataset and start the GUI with the data loaded

Options:
  --target COLUMN  Set the target column to preselect in the GUI (default: MedHouseVal for California housing).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from sklearn.datasets import fetch_california_housing


def prepare_dataset(csv_path: Path) -> Path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    ds = fetch_california_housing(as_frame=True)
    df = ds.frame
    df.to_csv(csv_path, index=False)
    print(f"[demo_setup] Dataset saved to {csv_path}")
    return csv_path


def launch_gui(csv_path: Path, target: str | None) -> None:
    # Import heavy GUI pieces only if launching
    from PyQt5.QtWidgets import QApplication
    from modeling_gui.main import MainApp
    from modeling_gui.utils.file_helper import load_csv

    app = QApplication.instance() or QApplication(sys.argv)
    win = MainApp()
    df = load_csv(str(csv_path))
    win.populate_from_dataframe(df)

    if target:
        idx = win.y_combo.findText(target)
        if idx >= 0:
            win.y_combo.setCurrentIndex(idx)

    win.show()
    print("[demo_setup] Modeling-GUI launched with dataset preloaded.")
    sys.exit(app.exec_())


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare demo data and optionally launch Modeling-GUI.")
    parser.add_argument("--launch", action="store_true", help="Launch Modeling-GUI with the dataset preloaded.")
    parser.add_argument("--target", default="MedHouseVal", help="Target column to preselect in the GUI.")
    parser.add_argument("--out", default="demo_data/california_housing.csv", help="Where to save the dataset.")
    args = parser.parse_args()

    csv_path = Path(args.out)
    prepare_dataset(csv_path)

    print("\n[demo_setup] Next steps:")
    print(f"  • CSV ready at: {csv_path}")
    print("  • Launch command (manual): python -m modeling_gui")
    if args.launch:
        launch_gui(csv_path, args.target)
    else:
        print("  • Or auto-launch now with: python docs/scripts/demo_setup.py --launch")


if __name__ == "__main__":
    main()
