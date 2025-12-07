"""
Coach-style helper for user-facing guidance messages.
Keeps message logic centralized and non-technical.
"""

from enum import Enum, auto
from typing import Optional


class CoachState(Enum):
    DATA_UNLOADED = auto()
    DATA_LOADED = auto()
    TARGET_SELECTED = auto()
    ANALYSIS_RUNNING = auto()
    ANALYSIS_DONE = auto()
    ERROR = auto()


class CoachManager:
    """
    Manages context-sensitive, human-friendly guidance messages.
    """

    BASE_MESSAGES = {
        CoachState.DATA_UNLOADED: "Step 1: Load your data (CSV).",
        CoachState.DATA_LOADED: "Great! Now choose what you want to predict (target column).",
        CoachState.TARGET_SELECTED: "Ready. Click 'Smart Analyze' to let the app find a good model.",
        CoachState.ANALYSIS_RUNNING: "Analyzing your data… this may take a moment.",
        CoachState.ANALYSIS_DONE: "Model is trained. You can review the summary and plots.",
        CoachState.ERROR: "Something went wrong. Please check your selections and try again.",
    }

    def __init__(self):
        self.state = CoachState.DATA_UNLOADED

    def update(self, state: CoachState, extra: Optional[str] = None) -> str:
        self.state = state
        base = self.BASE_MESSAGES.get(state, "")
        if extra:
            return f"{base} {extra}"
        return base
