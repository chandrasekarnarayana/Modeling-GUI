"""In-app logging view and handler."""

import logging
from typing import List
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QFileDialog, QHBoxLayout


class MemoryLogHandler(logging.Handler):
    def __init__(self, capacity: int = 200):
        super().__init__()
        self.capacity = capacity
        self.records: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - GUI oriented
        msg = self.format(record)
        self.records.append(msg)
        if len(self.records) > self.capacity:
            self.records.pop(0)

    def get_text(self) -> str:
        return "\n".join(self.records)


class LogDialog(QDialog):
    def __init__(self, handler: MemoryLogHandler, parent=None):
        super().__init__(parent)
        self.handler = handler
        self.setWindowTitle("Logs / Messages")
        layout = QVBoxLayout()
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setPlainText(self.handler.get_text())
        layout.addWidget(self.text)

        btns = QHBoxLayout()
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        save = QPushButton("Save log…")
        save.clicked.connect(self.save_log)
        btns.addWidget(refresh)
        btns.addWidget(save)
        btns.addStretch()
        layout.addLayout(btns)
        self.setLayout(layout)

    def refresh(self):  # pragma: no cover - GUI oriented
        self.text.setPlainText(self.handler.get_text())

    def save_log(self):  # pragma: no cover - GUI oriented
        path, _ = QFileDialog.getSaveFileName(self, "Save log", "", "Text Files (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.handler.get_text())
