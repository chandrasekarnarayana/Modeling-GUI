"""
Domain presets that guide messaging, metrics emphasis, and plot preferences.
"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class DomainConfig:
    name: str
    hints: str
    preferred_metrics: List[str]
    preferred_plots: List[str]


DOMAIN_PRESETS = {
    "Generic": DomainConfig(
        name="Generic",
        hints="Predict numbers or categories; Smart Analyze will choose automatically.",
        preferred_metrics=["r2", "rmse", "accuracy"],
        preferred_plots=["residuals", "confusion", "feature_importance"],
    ),
    "Finance": DomainConfig(
        name="Finance",
        hints="Predict numeric KPIs, risk scores, or prices. Time columns can be used for forecasting.",
        preferred_metrics=["rmse", "mae", "mape", "r2", "auc"],
        preferred_plots=["residuals", "feature_importance"],
    ),
    "Science": DomainConfig(
        name="Science",
        hints="Analyze experiments and measurements. Focus on fit quality.",
        preferred_metrics=["r2", "rmse"],
        preferred_plots=["residuals", "feature_importance"],
    ),
    "Business": DomainConfig(
        name="Business",
        hints="Customer churn, segmentation, marketing outcomes.",
        preferred_metrics=["accuracy", "auc", "f1"],
        preferred_plots=["confusion", "feature_importance"],
    ),
}


def get_domain_config(name: str) -> DomainConfig:
    """
    Return a domain config by name, defaulting to Generic.
    """
    return DOMAIN_PRESETS.get(name, DOMAIN_PRESETS["Generic"])
