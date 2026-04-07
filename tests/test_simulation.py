"""Test simulation modules in isolation."""
from simulation.network_graph import NetworkGraph
from simulation.job_market import JobMarket
from simulation.pipeline import PipelineManager


def test_network_graph_seeded():
    g1 = NetworkGraph(seed=42)
    g2 = NetworkGraph(seed=42)
    # Same seed should produce same warmth values
    for name in g1.graph.nodes:
        assert g1.get_warmth(name) == g2.get_warmth(name)


def test_network_warmth_update():
    g = NetworkGraph(seed=42)
    person = list(g.graph.nodes)[0]
    old = g.get_warmth(person)
    g.update_warmth(person, 0.1)
    new = g.get_warmth(person)
    assert new > old


def test_network_social_capital():
    g = NetworkGraph(seed=42)
    assert g.social_capital == 1.0
    g.update_social_capital(-0.3)
    assert g.social_capital == 0.7


def test_network_connections():
    g = NetworkGraph(seed=42)
    person = list(g.graph.nodes)[0]
    connections = g.get_connections(person)
    assert isinstance(connections, list)


def test_network_observable():
    g = NetworkGraph(seed=42)
    obs = g.get_observable_graph()
    assert "nodes" in obs
    assert "edges" in obs
    assert len(obs["nodes"]) == 12


def test_job_market_seeded():
    m1 = JobMarket(seed=42)
    m2 = JobMarket(seed=42)
    for company in m1.get_companies():
        assert m1.get_signal(company) == m2.get_signal(company)


def test_job_market_companies():
    m = JobMarket(seed=42)
    companies = m.get_companies()
    assert len(companies) == 6
    assert "TechCorp" in companies


def test_job_market_signals():
    m = JobMarket(seed=42)
    signal = m.get_signal("TechCorp")
    assert 0.0 <= signal <= 1.0


def test_job_market_ground_truth():
    m = JobMarket(seed=42)
    # TechCorp is set to true_hiring_state=True in template
    assert m.is_actively_hiring("TechCorp") is True


def test_pipeline_add_process():
    p = PipelineManager(seed=42)
    process = p.add_process("TechCorp", current_step=0)
    assert process.company == "TechCorp"
    assert process.stage.value == "APPLIED"


def test_pipeline_advance():
    p = PipelineManager(seed=42)
    p.add_process("TechCorp", current_step=0, referral_used=True)
    success, msg = p.advance_process("TechCorp", current_step=1)
    # With referral, 60% chance — may or may not pass
    assert isinstance(success, bool)


def test_pipeline_expiry():
    p = PipelineManager(seed=42)
    p.add_process("TechCorp", current_step=0)
    expired = p.check_expirations(current_step=10)
    assert "TechCorp" in expired
