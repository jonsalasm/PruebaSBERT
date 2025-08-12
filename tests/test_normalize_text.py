import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from services import normalize_text


def test_equivalent_phrases_lemmatization():
    sentence1 = "I am using Python"
    sentence2 = "I use Python"
    assert normalize_text(sentence1) == normalize_text(sentence2)


def test_critical_terms_preserved():
    sentence = "Experience with Python using libraries"
    tokens = normalize_text(sentence).split()
    assert "with" in tokens
    assert "use" in tokens
