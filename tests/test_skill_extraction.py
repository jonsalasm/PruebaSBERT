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
import services


def test_extract_skills_from_resume(monkeypatch):
    captured = {}

    def fake_extract_skills_from_text(text_blocks):
        captured['text_blocks'] = text_blocks
        return ['Python'], ['Teamwork']

    monkeypatch.setattr(services, 'extract_skills_from_text', fake_extract_skills_from_text)
    resume = "Python developer with great teamwork skills."
    hard, soft = services.extract_skills_from_resume(resume)

    assert captured['text_blocks'] == [resume]
    assert hard == ['Python']
    assert soft == ['Teamwork']


def test_evaluate_resume_against_job_skill_sets(monkeypatch):
    def fake_extract_resume(text):
        return ['Python', 'SQL'], ['communication', 'teamwork']

    def fake_extract_job(blocks):
        return ['Python', 'R'], ['communication', 'leadership']

    monkeypatch.setattr(services, 'extract_skills_from_resume', fake_extract_resume)
    monkeypatch.setattr(services, 'extract_skills_from_text', fake_extract_job)
    monkeypatch.setattr(services, 'find_best_sentence_match', lambda *a, **k: None)
    monkeypatch.setattr(services, 'extract_sentences', lambda text: [text])

    resume_text = "Resume text"
    job_title = "Data Scientist"
    responsibilities = []
    qualifications = []

    matches, scores, extra = services.evaluate_resume_against_job(
        resume_text, job_title, responsibilities, qualifications
    )

    assert set(extra['resume_hard_skills']) == {'Python', 'SQL'}
    assert set(extra['job_hard_skills']) == {'Python', 'R'}
    assert set(extra['matched_hard_skills']) == {'Python'}
    assert set(extra['missing_hard_skills']) == {'R'}
    assert set(extra['resume_soft_skills']) == {'communication', 'teamwork'}
    assert set(extra['job_soft_skills']) == {'communication', 'leadership'}
    assert set(extra['matched_soft_skills']) == {'communication'}
    assert set(extra['missing_soft_skills']) == {'leadership'}
