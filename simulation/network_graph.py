import random
import networkx as nx


# People templates for generating network nodes
PEOPLE_TEMPLATES = [
    {"name": "Alice Chen", "company": "TechCorp", "is_active_connector": True},
    {"name": "Bob Martinez", "company": "TechCorp", "is_active_connector": False},
    {"name": "Carol Johnson", "company": "DataFlow", "is_active_connector": True},
    {"name": "David Kim", "company": "DataFlow", "is_active_connector": False},
    {"name": "Eva Patel", "company": "CloudBase", "is_active_connector": False},
    {"name": "Frank Liu", "company": "CloudBase", "is_active_connector": True},
    {"name": "Grace Wang", "company": "AIStartup", "is_active_connector": False},
    {"name": "Henry Zhao", "company": "AIStartup", "is_active_connector": True},
    {"name": "Irene Park", "company": "MegaSoft", "is_active_connector": False},
    {"name": "Jack Brown", "company": "MegaSoft", "is_active_connector": True},
    {"name": "Karen Lee", "company": "NeuralNet", "is_active_connector": False},
    {"name": "Leo Davis", "company": "NeuralNet", "is_active_connector": True},
]


class NetworkGraph:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.graph = nx.Graph()
        self.social_capital = 1.0
        self._build_network()

    def _build_network(self):
        for person in PEOPLE_TEMPLATES:
            warmth = self.rng.uniform(0.1, 0.4)
            noise = self.rng.gauss(0, 0.1)
            self.graph.add_node(
                person["name"],
                company=person["company"],
                warmth=warmth,
                referral_willingness_signal=max(0, min(1, warmth + noise)),
                is_active_connector=person["is_active_connector"],
            )

        # Create edges (relationships) between people
        people = list(self.graph.nodes)
        for i in range(len(people)):
            for j in range(i + 1, len(people)):
                # Same company = higher chance of connection
                same_company = (
                    self.graph.nodes[people[i]]["company"]
                    == self.graph.nodes[people[j]]["company"]
                )
                connect_prob = 0.7 if same_company else 0.25
                if self.rng.random() < connect_prob:
                    weight = self.rng.uniform(0.3, 0.9)
                    self.graph.add_edge(people[i], people[j], weight=weight)

    @property
    def nodes(self):
        return dict(self.graph.nodes)

    def get_warmth(self, person: str) -> float:
        if person not in self.graph:
            return 0.0
        return self.graph.nodes[person]["warmth"]

    def update_warmth(self, person: str, delta: float):
        if person not in self.graph:
            return
        noise_factor = self.rng.uniform(0.8, 1.2)
        current = self.graph.nodes[person]["warmth"]
        new_warmth = max(0.0, min(1.0, current + delta * noise_factor))
        self.graph.nodes[person]["warmth"] = new_warmth
        # Update noisy signal too
        noise = self.rng.gauss(0, 0.1)
        self.graph.nodes[person]["referral_willingness_signal"] = max(
            0, min(1, new_warmth + noise)
        )

    def get_connections(self, person: str) -> list[str]:
        if person not in self.graph:
            return []
        return list(self.graph.neighbors(person))

    def get_path_to_company(self, company: str) -> list[str]:
        """Find shortest path from any known warm contact to someone at target company."""
        company_people = [
            n for n, d in self.graph.nodes(data=True) if d["company"] == company
        ]
        if not company_people:
            return []

        # Find person with highest warmth as starting point
        warm_people = sorted(
            self.graph.nodes(data=True),
            key=lambda x: x[1]["warmth"],
            reverse=True,
        )

        for start_person, _ in warm_people:
            for target in company_people:
                try:
                    path = nx.shortest_path(self.graph, start_person, target)
                    return path
                except nx.NetworkXNoPath:
                    continue
        return []

    def get_people_at_company(self, company: str) -> list[str]:
        return [
            n for n, d in self.graph.nodes(data=True) if d["company"] == company
        ]

    def update_social_capital(self, delta: float):
        self.social_capital = max(0.0, min(1.0, self.social_capital + delta))

    def get_observable_graph(self) -> dict:
        """Return the noisy observation the agent sees.

        IMPORTANT: Never expose true warmth. Only show the noisy
        referral_willingness_signal. This enforces partial observability.
        """
        nodes = {}
        for name, data in self.graph.nodes(data=True):
            nodes[name] = {
                "company": data["company"],
                "warmth_signal": round(
                    data["referral_willingness_signal"], 2
                ),
                "is_active_connector": data["is_active_connector"],
            }
        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append({"from": u, "to": v, "weight": round(data["weight"], 2)})
        return {"nodes": nodes, "edges": edges}

    def snapshot_warmth(self) -> dict:
        """Take a snapshot of current warmth values for later comparison."""
        return {
            name: data["warmth"] for name, data in self.graph.nodes(data=True)
        }
