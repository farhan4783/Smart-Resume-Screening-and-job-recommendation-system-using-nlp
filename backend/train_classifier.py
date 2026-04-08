"""
backend/train_classifier.py
-----------------------------
Train a Random Forest + TF-IDF classifier on Resume.csv (25 categories).
Saves model artifacts to models/ directory.

Usage:
    python -m backend.train_classifier
"""

import os
import sys
import re
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.join(_HERE, "..")
_CSV_PATH = os.path.join(_PROJECT, "DataSet", "Resume", "Resume.csv")
_MODEL_DIR = os.path.join(_PROJECT, "models")


def clean_resume_text(text: str) -> str:
    """Clean raw resume text for TF-IDF vectorization."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s\+\#\.\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def train_and_save():
    """Train the classifier and save to disk."""
    print("[train_classifier] Loading Resume.csv ...")
    df = pd.read_csv(_CSV_PATH)

    # Check for required columns
    text_col = None
    for col in ["Resume_str", "Resume_html", "resume", "text"]:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        print(f"[train_classifier] ERROR: Could not find text column. Columns: {list(df.columns)}")
        sys.exit(1)

    print(f"[train_classifier] Using text column: '{text_col}', Category column: 'Category'")
    print(f"[train_classifier] Dataset size: {len(df)} resumes, {df['Category'].nunique()} categories")

    # Clean text
    df["clean_text"] = df[text_col].apply(clean_resume_text)
    df = df[df["clean_text"].str.len() > 50]  # Remove too-short entries

    # Encode labels
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["Category"])

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"]
    )

    # TF-IDF vectorizer
    print("[train_classifier] Fitting TF-IDF vectorizer ...")
    tfidf = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
        min_df=2,
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Random Forest classifier
    print("[train_classifier] Training Random Forest classifier ...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_tfidf, y_train)

    # Evaluate
    y_pred = rf.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[train_classifier] ✅ Test Accuracy: {acc * 100:.1f}%\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save artifacts
    os.makedirs(_MODEL_DIR, exist_ok=True)
    with open(os.path.join(_MODEL_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
    with open(os.path.join(_MODEL_DIR, "category_model.pkl"), "wb") as f:
        pickle.dump(rf, f)
    with open(os.path.join(_MODEL_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    print(f"[train_classifier] ✅ Models saved to {_MODEL_DIR}/")
    return acc


if __name__ == "__main__":
    train_and_save()
