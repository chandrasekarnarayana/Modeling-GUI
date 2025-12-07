import joblib


def save_model_bundle(path, model, metadata):
    """
    Persist a trained model along with metadata.
    """
    bundle = {"model": model, "metadata": metadata}
    joblib.dump(bundle, path)
    return path


def load_model_bundle(path):
    """
    Load a persisted model bundle.
    """
    return joblib.load(path)
