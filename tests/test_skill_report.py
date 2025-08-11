import os
import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import services
from services import compare_skills, evaluate_resume_against_job


def test_compare_skills_basic():
    cv = (["Python", "Git"], ["Teamwork", "Communication"])
    job = (["Python", "Docker"], ["Communication", "Leadership"])
    report = compare_skills(cv, job)
    assert report["matched_hard_skills"] == ["Python"]
    assert report["missing_hard_skills"] == ["Docker"]
    assert report["matched_soft_skills"] == ["Communication"]
    assert report["missing_soft_skills"] == ["Leadership"]


def test_evaluate_resume_skill_report(monkeypatch):
    call = {"count": 0}

    def fake_extract_skills_from_text(text_blocks):
        if call["count"] == 0:
            call["count"] += 1
            return (["Python", "Docker"], ["Communication", "Teamwork"])
        return (["Python", "Git"], ["Leadership", "Communication"])

    monkeypatch.setattr(services, "extract_skills_from_text", fake_extract_skills_from_text)
    monkeypatch.setattr(services, "find_best_sentence_match", lambda *args, **kwargs: None)
    monkeypatch.setattr(services, "extract_years_experience", lambda *args, **kwargs: 0)

    matches, scores, extra = evaluate_resume_against_job(
        resume_text="sample resume",
        job_title="Developer",
        responsibilities=["Do things"],
        qualifications_and_experience=["Know stuff"],
    )

    expected = {
        "matched_hard_skills": ["Python"],
        "missing_hard_skills": ["Docker"],
        "matched_soft_skills": ["Communication"],
        "missing_soft_skills": ["Teamwork"],
    }
    assert extra["skill_report"] == expected
    assert extra["missing_soft_skills"] == expected["missing_soft_skills"]
