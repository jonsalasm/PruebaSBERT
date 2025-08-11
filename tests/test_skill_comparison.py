import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from services import (
    extract_skills_from_resume,
    compare_skills,
    evaluate_resume_against_job,
)


def test_extract_skills_from_resume_delegates(monkeypatch):
    captured = {}

    def fake_extract(text):
        captured["text"] = text
        return ["python"], ["communication"]

    monkeypatch.setattr("services.extract_skills_from_text", fake_extract)

    hard, soft = extract_skills_from_resume("Resume text")
    assert captured["text"] == "Resume text"
    assert hard == ["python"]
    assert soft == ["communication"]


def test_compare_skills_reports_matches_and_missing():
    cv = (["python", "sql"], ["teamwork"])
    job = (["python", "docker"], ["teamwork", "leadership"])

    report = compare_skills(cv, job)

    assert report["matched_hard_skills"] == ["python"]
    assert report["missing_hard_skills"] == ["docker"]
    assert report["matched_soft_skills"] == ["teamwork"]
    assert report["missing_soft_skills"] == ["leadership"]


def test_evaluate_resume_returns_skill_report(monkeypatch):
    def fake_extract_resume(text):
        return ["python"], ["teamwork"]

    def fake_extract_job(text):
        return ["python", "docker"], ["teamwork", "leadership"]

    monkeypatch.setattr("services.extract_skills_from_resume", fake_extract_resume)
    monkeypatch.setattr("services.extract_skills_from_text", fake_extract_job)
    monkeypatch.setattr("services.find_best_sentence_match", lambda s, t, threshold=0.6: None)
    monkeypatch.setattr("services.extract_sentences", lambda text: [])

    _, _, extra = evaluate_resume_against_job(
        resume_text="",
        job_title="",
        responsibilities=[],
        qualifications_and_experience=[],
    )

    report = extra["skill_report"]
    assert report["matched_hard_skills"] == ["python"]
    assert report["missing_hard_skills"] == ["docker"]
    assert report["matched_soft_skills"] == ["teamwork"]
    assert report["missing_soft_skills"] == ["leadership"]
