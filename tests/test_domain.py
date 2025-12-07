from modeling_gui.domain import get_domain_config


def test_domain_config_defaults():
    cfg = get_domain_config("Generic")
    assert "r2" in cfg.preferred_metrics
    assert cfg.name == "Generic"


def test_domain_specific_entries():
    finance = get_domain_config("Finance")
    assert "rmse" in finance.preferred_metrics
    assert "feature_importance" in finance.preferred_plots

    science = get_domain_config("Science")
    assert "r2" in science.preferred_metrics

    business = get_domain_config("Business")
    assert "accuracy" in business.preferred_metrics
