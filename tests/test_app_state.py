"""Tests for application state."""

from hospital_demand_analyzer.app.state import AppState


def test_app_state_init_without_explicit_config() -> None:
    state = AppState()
    assert state.config is not None
    assert state.cluster_alto_cutoff > 0
    assert state.cluster_baixo_cutoff >= 0
    assert state.clustering_method
