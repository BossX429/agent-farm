"""Shared state for agent-farm: model tiers, colonies, and stats.

This module has no imports from other agent_farm modules, so every
tool module can import it without creating a cycle.
"""

import os
import uuid
import json
import httpx
import asyncio
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

OLLAMA_URL = "http://localhost:11434"

# Model tiers - optimized for 7900 XTX 24GB VRAM
# ALL models must be capable of following tool format.
# Roles use different models on purpose: the code-facing roles
# (worker writes code, learner analyzes it) run the code-tuned
# qwen2.5-coder, while scout/memory/guardian stay on the fast
# qwen3:4b. The synthesizer deliberately uses qwen2.5:14b
# (8.99GB) for accuracy on result synthesis, so it is NOT a
# 4b tier. Different models per role is what lets the profiler
# measure real differences instead of timing noise between
# identical workers.
MODEL_TIERS = {
    "scout": "qwen3:4b",       # 2.5GB - reconnaissance
    "worker": "qwen2.5-coder:7b-v8-review",  # 8.1GB - task execution, code-tuned
    "memory": "qwen3:4b",      # 2.5GB - was 8b but inconsistent tool format
    "guardian": "qwen3:4b",    # 2.5GB - validation, protection
    "learner": "qwen2.5-coder:7b-v8-review",  # 8.1GB - code analysis, code-tuned
    "synthesizer": "qwen2.5:14b",  # 8.99GB - synthesis/summarization (accuracy over speed)
}

# Grounding instruction to prevent hallucination
GROUNDING = "/nothink You are a specialized AI bug. Answer ONLY from given context. If unsure, say 'UNCERTAIN'. Be concise."


@dataclass
class Bug:
    """Individual bug in a colony"""
    id: str
    role: str
    model: str
    status: str = "idle"
    last_task: str = ""
    discoveries: List[str] = field(default_factory=list)


@dataclass 
class SharedConsciousness:
    """Shared state across entangled bugs"""
    discoveries: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    threats: List[str] = field(default_factory=list)
    rag_cache: Dict[str, str] = field(default_factory=dict)


@dataclass
class NanoArmor:
    """Protection layer for bugs"""
    shield: bool = True      # Path traversal protection
    filter: bool = True      # Input sanitization  
    validation: bool = True  # Data verification
    scanner: bool = True     # Threat detection
    cache: bool = True       # Response caching
    formatter: bool = True   # Output formatting
    repair: bool = True      # Self-healing
    sensor: bool = True      # Environment monitoring
    attacks_blocked: int = 0


@dataclass
class Colony:
    """Bug colony with shared consciousness"""
    id: str
    bugs: List[Bug] = field(default_factory=list)
    colony_type: str = "standard"
    consciousness: SharedConsciousness = field(default_factory=SharedConsciousness)
    armor: NanoArmor = field(default_factory=NanoArmor)
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    tasks_completed: int = 0


# Active colonies
colonies: Dict[str, Colony] = {}

# Stats tracking
stats = {
    "colonies_spawned": 0,
    "tasks_executed": 0,
    "discoveries": 0,
    "swarm_runs": 0,
    # NOTE: no parallel_speedup bucket. A previous version "measured" speedup
    # by deriving sequential time from elapsed time (elapsed * tasks / bugs),
    # which is circular -- speedup then just equals tasks / bugs. It was
    # removed rather than replaced with another invented number. If a real
    # speedup is ever claimed, it must come from a timed sequential baseline.
}
