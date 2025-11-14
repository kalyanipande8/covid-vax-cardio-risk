"""Modeling pipeline stubs.

Include training, evaluation, and explainability hooks here.
"""
from typing import Any, Dict


def train_dummy_model(X, y, **kwargs) -> Dict[str, Any]:
    """A placeholder training function that returns a trivial model dict.

    Replace with sklearn/pipeline when ready.
    """
    return {"type": "dummy", "n_samples": len(X) if X is not None else 0}


def explain_model_dummy(model, X):
    """Return a trivial explanation for the dummy model.

    When using real models, integrate SHAP or LIME here.
    """
    return [{"index": i, "importance": 0.0} for i, _ in enumerate(X or [])]
