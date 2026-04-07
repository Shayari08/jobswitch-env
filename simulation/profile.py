"""Candidate profile management and ATS scoring.

Tracks resume versions, skill-role fit, and ATS optimization.
"""

import random


class CandidateProfile:
    """Manages the candidate's profile, resume versions, and ATS scores."""

    def __init__(self, seed: int = 42, skills: dict | None = None,
                 experience_months: int = 6, seniority: str = "junior"):
        self.rng = random.Random(seed)
        self.skills = skills or {"ml": 0.7, "backend": 0.6, "frontend": 0.3}
        self.experience_months = experience_months
        self.seniority = seniority
        self.resume_versions: dict[str, str] = {}  # company -> role_type
        self.ats_scores: dict[str, float] = {}  # company -> ATS score

    def tailor_resume(self, company: str, role_type: str) -> float:
        """Tailor resume for a company/role. Returns ATS improvement."""
        current_ats = self.ats_scores.get(company, 50.0)
        improvement = self.rng.uniform(8, 18)
        new_ats = min(100, current_ats + improvement)
        self.ats_scores[company] = new_ats
        self.resume_versions[company] = role_type
        return improvement

    def get_role_fit(self, required_skills: dict) -> float:
        """Compute fit score between candidate skills and role requirements."""
        if not required_skills:
            return 0.5
        total = 0.0
        count = 0
        for skill, weight in required_skills.items():
            candidate_level = self.skills.get(skill, 0.0)
            total += candidate_level * weight
            count += weight
        return total / max(count, 0.01)

    def to_dict(self) -> dict:
        return {
            "skills": self.skills,
            "experience_months": self.experience_months,
            "seniority": self.seniority,
        }
