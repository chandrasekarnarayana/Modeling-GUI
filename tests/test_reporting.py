from modeling_gui.reporting import build_summary_text


def test_build_summary_text_includes_sections():
    summary = {
        "data": {"rows": 10, "cols": 3, "types": {"numeric": 2, "categorical": 1}},
        "preprocessing": {"missing": "mean", "scaling": "standardize", "categoricals": "Auto-handled"},
        "modeling": {"problem_type": "regression", "algorithms": ["OLS", "GBR"], "best_model": "GBR", "cv": "5-fold"},
        "evaluation": {"split": "test_size=0.2", "primary": {"r2": 0.9}, "secondary": {}},
    }
    metrics = {"r2_train": 0.95, "r2_test": 0.9}
    text = build_summary_text(summary, metrics)
    assert "Data summary" in text
    assert "Preprocessing steps" in text
    assert "Modeling decisions" in text
    assert "Evaluation metrics" in text
