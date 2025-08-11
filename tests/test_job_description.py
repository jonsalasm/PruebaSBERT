import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from services import (
    combine_job_description,
    evaluate_resume_against_job,
)


def test_combine_job_description_returns_single_text_block():
    responsibilities = ["Do this", "Do that"]
    qualifications = ["Need skill", "Another skill"]
    result = combine_job_description(responsibilities, qualifications)
    expected = "Do this\nDo that\nNeed skill\nAnother skill"
    assert result == expected


def test_evaluate_resume_passes_combined_description(monkeypatch):
    responsibilities = ["Resp A"]
    qualifications = ["Qual A"]
    combined_expected = combine_job_description(responsibilities, qualifications)

    captured = {}

    def fake_extract(text):
        captured["text"] = text
        return [], []

    def fake_extract_resume(text):
        return [], []

    def fake_find_best_sentence_match(sentences, target, threshold=0.6):
        return None

    monkeypatch.setattr("services.extract_skills_from_text", fake_extract)
    monkeypatch.setattr("services.extract_skills_from_resume", fake_extract_resume)
    monkeypatch.setattr("services.find_best_sentence_match", fake_find_best_sentence_match)
    monkeypatch.setattr("services.extract_sentences", lambda text: [])

    evaluate_resume_against_job("", "", responsibilities, qualifications)

    assert captured["text"] == combined_expected

