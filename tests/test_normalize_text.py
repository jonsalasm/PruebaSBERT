import os
import sys
import pytest
import nltk
from nltk.corpus import stopwords

# Stub out NLTK downloads and data-dependent functions
nltk.download = lambda *args, **kwargs: None
stopwords.words = lambda *args, **kwargs: ['i', 'am', 'be', 'a', 'the', 'is', 'using', 'with']
nltk.word_tokenize = lambda text: text.split()
nltk.pos_tag = lambda tokens: [(t, 'NN') for t in tokens]

class DummyWordnet:
    ADJ = 'a'
    NOUN = 'n'
    VERB = 'v'
    ADV = 'r'

nltk.corpus.wordnet = DummyWordnet()

class DummyLemmatizer:
    def lemmatize(self, word, pos=None):
        if word.endswith('ing'):
            return word[:-3] + 'e'
        return word

nltk.stem.WordNetLemmatizer = DummyLemmatizer

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
