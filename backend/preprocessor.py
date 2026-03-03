"""
backend/preprocessor.py
------------------------
NLTK-based text cleaning pipeline:
  tokenize → lowercase → remove stop-words → WordNet lemmatize
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data on first run
def _ensure_nltk():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("tokenizers/punkt_tab", "punkt_tab"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)

_ensure_nltk()

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))


def clean_text(raw_text: str) -> str:
    """
    Remove special characters, URLs, emails (for NLP tasks only).
    Returns lowercased, stripped string.
    """
    text = raw_text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)         # remove URLs
    text = re.sub(r"\S+@\S+", " ", text)                 # remove emails
    text = re.sub(r"[^a-z0-9\s\+\#\-\.]", " ", text)    # keep alphanumeric + tech chars
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatize(text: str) -> list[str]:
    """
    Full NLP pipeline: clean → tokenize → remove stop-words → lemmatize.
    Returns list of processed tokens.
    """
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    processed = [
        _lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok.isalpha() and tok not in _stop_words and len(tok) > 2
    ]
    return processed


def preprocess(text: str) -> tuple[list[str], str]:
    """
    Returns (token_list, joined_string) for downstream tasks.
    """
    tokens = tokenize_and_lemmatize(text)
    return tokens, " ".join(tokens)
