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
# qwen3:4b. The synthesizer deliberately uses qwen3:30b-a3b-q3k
# (14.7GB) for accuracy on result synthesis, so it is NOT a
# 4b tier. It runs AFTER colony work completes, so Ollama evicts
# the finished colony models to make room; it never shares VRAM
# with two 8.1GB coder workers. Different models per role is what
# lets the profiler measure real differences instead of timing
# noise between identical workers.
MODEL_TIERS = {
    "scout": "qwen3:4b",       # 2.5GB - reconnaissance
    "worker": "qwen2.5-coder:7b-v8-review",  # 8.1GB - task execution, code-tuned
    "memory": "qwen3:4b",      # 2.5GB - was 8b but inconsistent tool format
    "guardian": "qwen3:4b",    # 2.5GB - validation, protection
    "learner": "qwen2.5-coder:7b-v8-review",  # 8.1GB - code analysis, code-tuned
    "synthesizer": "qwen3:30b-a3b-q3k",  # 14.7GB - synthesis/summarization (accuracy over speed)
}

# Models that are review-tuned: given a bare task prompt they
# return a review verdict ("CLEAN" / "BUGGY") instead of doing
# the work. qwen2.5-coder:7b-v8-review does this reproducibly:
# the bare prompt "write a python function that adds two numbers"
# returns "CLEAN" every time, while "Return only the code" makes
# it write proper code. These models get an explicit do-the-task
# instruction prepended to the prompt. The instruction is keyed
# by MODEL, not by role: it follows the model wherever it is
# used, and stops applying if a role's model changes to one that
# does not need it. The tool-swarm path does not need it: its
# prompt builders already tell the model exactly what to do
# (pick a tool, return JSON).
MODELS_NEEDING_TASK_INSTRUCTION = frozenset({"qwen2.5-coder:7b-v8-review"})
TASK_INSTRUCTION = (
    "Do the task below. Write out the full answer. "
    "Do not give a one-word verdict, do not review anything, "
    "do not say CLEAN or BUGGY."
)


def prompt_for_model(model: str, prompt: str) -> str:
    """Prepend the do-the-task instruction for review-tuned models.

    A model that does not need the instruction gets the prompt
    back unchanged, so this is safe to call on every prompt.
    """
    if model in MODELS_NEEDING_TASK_INSTRUCTION:
        return f"{TASK_INSTRUCTION}\n\n{prompt}"
    return prompt

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
    consciousness: SharedConsciousness = field(default_factory=SharedConsciousness)
    armor: NanoArmor = field(default_factory=NanoArmor)
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    tasks_completed: int = 0
    # colony_type goes LAST, not after bugs: a field inserted in
    # the middle would silently rebind any caller that constructs
    # Colony(id, bugs, consciousness) positionally, binding the
    # consciousness object to colony_type.
    colony_type: str = "standard"


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
