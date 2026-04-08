"""
backend/category_classifier.py
-------------------------------
Resume category prediction using pre-trained Random Forest + TF-IDF.
Predicts which of 25 job categories a resume belongs to.
"""

import os
import re
import pickle
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_HERE, "..", "models")

_tfidf = None
_model = None
_le = None


def _load_models():
    global _tfidf, _model, _le
    if _model is not None:
        return True
    try:
        with open(os.path.join(_MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
            _tfidf = pickle.load(f)
        with open(os.path.join(_MODEL_DIR, "category_model.pkl"), "rb") as f:
            _model = pickle.load(f)
        with open(os.path.join(_MODEL_DIR, "label_encoder.pkl"), "rb") as f:
            _le = pickle.load(f)
        return True
    except FileNotFoundError:
        print("[category_classifier] Model files not found. Run: python -m backend.train_classifier")
        return False


def _clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s\+\#\.\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def predict_category(resume_text: str) -> dict:
    """
    Predict the job category of a resume.

    Args:
        resume_text: Raw or cleaned resume text

    Returns:
        dict with keys:
            - category (str): Predicted category name
            - confidence (float): Prediction confidence 0–1
            - top_3 (list[tuple]): Top 3 categories with probabilities
    """
    if not _load_models():
        return {
            "category": "UNKNOWN",
            "confidence": 0.0,
            "top_3": [],
        }

    cleaned = _clean(resume_text)
    if len(cleaned) < 20:
        return {"category": "UNKNOWN", "confidence": 0.0, "top_3": []}

    tfidf_vec = _tfidf.transform([cleaned])
    proba = _model.predict_proba(tfidf_vec)[0]

    top_indices = np.argsort(proba)[::-1][:3]
    top_3 = [
        (_le.classes_[idx], round(float(proba[idx]), 3))
        for idx in top_indices
    ]

    predicted_idx = top_indices[0]
    return {
        "category": _le.classes_[predicted_idx],
        "confidence": round(float(proba[predicted_idx]), 3),
        "top_3": top_3,
    }


def get_all_categories() -> list[str]:
    """Return all category labels."""
    if not _load_models():
        return []
    return list(_le.classes_)
