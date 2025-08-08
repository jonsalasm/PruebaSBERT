
from services import evaluate_resume_against_job
import json

if __name__ == "__main__":
    # Simulated resume text
    with open("data/sample_resume.txt", "r", encoding="utf-8") as f:
        resume_text = f.read()

    # Simulated job description text
    job_description = """
    We are seeking a candidate with experience in Python, AWS, and Docker.
    The role requires strong leadership, effective communication, and the ability to collaborate across teams.
    A minimum of 3 years of experience working with cloud infrastructure is preferred.
    """

    # Evaluate match
    matches, scores, extra = evaluate_resume_against_job(
        resume_text=resume_text,
        job_title="Cloud Infrastructure Engineer",
        responsibilities=[
            "Deploy cloud applications using AWS",
            "Collaborate with cross-functional teams",
            "Implement Docker-based microservices"
        ],
        qualifications_and_experience=[
            "3+ years of experience in Python and cloud platforms",
            "Bachelor degree in Computer Science or related field"
        ],
        required_location="remote"
    )

    # Output results
    print("MATCHES:")
    print(json.dumps(matches, indent=2))

    print("\nSECTION SCORES:")
    print(json.dumps(scores, indent=2))

    print("\nEXTRA INFO:")
    print(json.dumps(extra, indent=2))
