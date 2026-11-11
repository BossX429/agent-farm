import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def _isolate_profile_cards(monkeypatch, tmp_path):
    """Keep profile cards out of the repo during tests.

    Profiler factory Phase 2 records every swarm run on a
    profile card in profiles/. Without this, any test that
    runs a swarm would write cards into the repo as a side
    effect. Tests that need their own cards set
    AGENT_FARM_PROFILES_DIR themselves.
    """
    monkeypatch.setenv("AGENT_FARM_PROFILES_DIR", str(tmp_path))

