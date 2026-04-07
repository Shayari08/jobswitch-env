import random


COMPANY_TEMPLATES = [
    {
        "name": "TechCorp",
        "true_hiring_state": True,
        "role_fit_score": 0.75,
        "salary_band": (90000, 130000),
        "growth_stage": "mature",
        "culture_match": 0.7,
    },
    {
        "name": "DataFlow",
        "true_hiring_state": True,
        "role_fit_score": 0.80,
        "salary_band": (95000, 140000),
        "growth_stage": "growth",
        "culture_match": 0.6,
    },
    {
        "name": "CloudBase",
        "true_hiring_state": False,
        "role_fit_score": 0.60,
        "salary_band": (85000, 125000),
        "growth_stage": "mature",
        "culture_match": 0.5,
    },
    {
        "name": "AIStartup",
        "true_hiring_state": True,
        "role_fit_score": 0.85,
        "salary_band": (80000, 150000),
        "growth_stage": "early",
        "culture_match": 0.8,
    },
    {
        "name": "MegaSoft",
        "true_hiring_state": True,
        "role_fit_score": 0.65,
        "salary_band": (100000, 160000),
        "growth_stage": "enterprise",
        "culture_match": 0.4,
    },
    {
        "name": "NeuralNet",
        "true_hiring_state": False,
        "role_fit_score": 0.70,
        "salary_band": (88000, 135000),
        "growth_stage": "growth",
        "culture_match": 0.65,
    },
]


class JobMarket:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.companies: dict[str, dict] = {}
        self._build_market()

    def _build_market(self):
        for template in COMPANY_TEMPLATES:
            # hiring_signal = noisy observation of hidden truth
            true_state = 1.0 if template["true_hiring_state"] else 0.0
            noise = self.rng.gauss(0, 0.15)
            hiring_signal = max(0.0, min(1.0, true_state + noise))

            self.companies[template["name"]] = {
                "name": template["name"],
                "true_hiring_state": template["true_hiring_state"],
                "hiring_signal": hiring_signal,
                "role_fit_score": template["role_fit_score"],
                "salary_band": template["salary_band"],
                "growth_stage": template["growth_stage"],
                "culture_match": template["culture_match"],
                "hiring_velocity": self.rng.uniform(0.3, 0.9),
                "recent_layoffs": self.rng.random() < 0.15,
                "role_demand_trend": self.rng.choice(
                    ["rising", "stable", "falling"]
                ),
            }

    def get_companies(self) -> list[str]:
        return list(self.companies.keys())

    def get_company(self, company: str) -> dict | None:
        return self.companies.get(company)

    def get_signal(self, company: str) -> float:
        """Noisy hiring signal visible to agent."""
        c = self.companies.get(company)
        if not c:
            return 0.0
        return c["hiring_signal"]

    def get_salary_band(self, company: str) -> tuple:
        c = self.companies.get(company)
        if not c:
            return (0, 0)
        return c["salary_band"]

    def is_actively_hiring(self, company: str) -> bool:
        """Ground truth — only grader uses this."""
        c = self.companies.get(company)
        if not c:
            return False
        return c["true_hiring_state"]

    def get_observable_companies(self) -> list[dict]:
        """Return the noisy observation the agent sees.

        IMPORTANT: Never expose ground truth for role_fit_score or
        culture_match. Apply noise to maintain partial observability.
        """
        result = []
        for name, data in self.companies.items():
            noisy_fit = max(0.0, min(1.0,
                data["role_fit_score"] + self.rng.gauss(0, 0.10)))
            noisy_culture = max(0.0, min(1.0,
                data["culture_match"] + self.rng.gauss(0, 0.10)))
            result.append({
                "name": name,
                "hiring_signal": round(data["hiring_signal"], 2),
                "role_fit_score": round(noisy_fit, 2),
                "culture_match": round(noisy_culture, 2),
                "known_salary_band": data["salary_band"],
                "growth_stage": data["growth_stage"],
            })
        return result

    def get_market_signals(self) -> dict:
        """Market-level signals visible to agent."""
        signals = {}
        for name, data in self.companies.items():
            signals[name] = {
                "hiring_velocity": round(data["hiring_velocity"], 2),
                "recent_layoffs": data["recent_layoffs"],
                "role_demand_trend": data["role_demand_trend"],
            }
        return signals

    def get_market_rate(self) -> float:
        """Average mid-point salary across actively hiring companies."""
        bands = [
            d["salary_band"]
            for d in self.companies.values()
            if d["true_hiring_state"]
        ]
        if not bands:
            return 100000.0
        return sum((lo + hi) / 2 for lo, hi in bands) / len(bands)
