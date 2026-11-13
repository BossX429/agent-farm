"""Ollama query helpers, previously at the top of server.py."""

import os
import uuid
import json
import httpx
import asyncio
from contextlib import nullcontext
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from .state import MODEL_TIERS, OLLAMA_URL, GROUNDING
from .reliability import MODEL_SEMAPHORE


def strip_think_blocks(text: str) -> str:
    """Remove Qwen3 <think>...</think> reasoning blocks.

    A qwen3 model emits its reasoning inside <think> tags. The
    grounded paths suppress this with "/nothink", but the raw and
    synthesis paths do not use grounding, so callers would receive
    reasoning text in code, tests, reviews, or summaries. An
    unclosed block (model cut off mid-thought) is removed to the
    end of the string. A response with no block is returned
    unchanged.
    """
    if not isinstance(text, str):
        return text
    if "<think>" not in text and "</think>" not in text:
        return text
    out = []
    rest = text
    while True:
        start = rest.find("<think>")
        if start == -1:
            # No opening tag left. A dangling </think> means the
            # text before it is reasoning (the model emitted the
            # close without the open); drop everything up to and
            # including it and keep what follows.
            end = rest.find("</think>")
            if end == -1:
                out.append(rest)
            else:
                out.append(rest[end + len("</think>"):])
            break
        out.append(rest[:start])
        end = rest.find("</think>", start)
        if end == -1:
            # Unclosed block: drop everything from <think> onward.
            break
        rest = rest[end + len("</think>"):]
    return "".join(out).strip()


async def query_ollama(model: str, prompt: str, grounded: bool = True) -> str:
    """Query Ollama with optional grounding instruction"""
    full_prompt = f"{GROUNDING}\n\n{prompt}" if grounded else prompt
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": model, "prompt": full_prompt, "stream": False},
                timeout=120.0
            )
            if resp.status_code == 200:
                return strip_think_blocks(resp.json().get("response", "No response"))
            return f"Error: {resp.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"


def sync_query_ollama(model: str, prompt: str, grounded: bool = True, slot_held: bool = False) -> str:
    """Synchronous wrapper for parallel execution.

    Model calls are gated by the shared MODEL_SEMAPHORE (see
    reliability.py), so independent thread pools in different tools
    cannot each run an unbounded number of simultaneous model
    requests. This is a queue, not a cancellation: a call waits for
    a slot instead of starting immediately.

    slot_held=True means the caller already holds a semaphore
    slot (probes.py does this so it can start its clock after
    the wait, not before). The call then skips acquiring a
    second slot: holding one while waiting for another would
    deadlock once the pool size matches the semaphore size.
    """
    full_prompt = f"{GROUNDING}\n\n{prompt}" if grounded else prompt
    try:
        slot = nullcontext() if slot_held else MODEL_SEMAPHORE
        with slot, httpx.Client() as client:
            resp = client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": model, "prompt": full_prompt, "stream": False},
                timeout=120.0
            )
            if resp.status_code == 200:
                return strip_think_blocks(resp.json().get("response", "No response"))
            return f"Error: {resp.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"


def sync_query_raw(model: str, prompt: str) -> str:
    """Raw query WITHOUT grounding - for deep analysis that needs full thinking"""
    try:
        with MODEL_SEMAPHORE, httpx.Client() as client:
            resp = client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=300.0  # 5 min for complex analysis
            )
            if resp.status_code == 200:
                return strip_think_blocks(resp.json().get("response", "No response"))
            return f"Error: {resp.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"


def synthesize_swarm_results(results: List[Dict], task_context: str = "") -> str:
    """
    Synthesize swarm results into a unified summary using the synthesizer model.
    Uses the synthesizer tier (qwen3:30b-a3b-q3k) for accuracy; no model
    guarantees zero hallucinations, so summaries still need checking.
    
    Args:
        results: List of swarm result dicts (with 'answer' or 'response' keys)
        task_context: Optional context about what the swarm was doing
    
    Returns:
        Synthesized summary string
    """
    model = MODEL_TIERS["synthesizer"]
    
    # Extract answers from results
    findings = []
    for i, r in enumerate(results, 1):
        answer = r.get("answer") or r.get("response") or r.get("result", "")
        if isinstance(answer, dict):
            answer = str(answer)
        task = r.get("task", f"Task {i}")
        role = r.get("role", "bug")
        status = r.get("status", "unknown")
        
        if answer and status not in ["failed", "error"]:
            findings.append(f"[{role}] {task}:\n{answer}")
    
    if not findings:
        return "No valid results to synthesize."
    
    combined = "\n\n---\n\n".join(findings)
    
    context_line = f"\nCONTEXT: {task_context}\n" if task_context else ""
    
    prompt = f"""You are a synthesis expert. Combine the following findings into ONE coherent summary.

STRICT RULES:
1. ONLY use information from the findings below - DO NOT add anything
2. If findings conflict, note the conflict
3. If something is uncertain, say "uncertain" or "unclear"
4. Be concise - bullet points are fine
5. No speculation, no assumptions, no hallucination
{context_line}
FINDINGS:
{combined}

---

SYNTHESIZED SUMMARY:"""
    
    try:
        with MODEL_SEMAPHORE, httpx.Client() as client:
            resp = client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=180.0  # 3 min for synthesis
            )
            if resp.status_code == 200:
                response = resp.json().get("response", "Synthesis failed")
                # Clean up common artifacts, including the <think>
                # block a qwen3 model emits when no "/nothink" is set.
                response = strip_think_blocks(response)
                if response.startswith("SYNTHESIZED SUMMARY:"):
                    response = response[20:].strip()
                return response
            return f"Synthesis error: {resp.status_code}"
    except Exception as e:
        return f"Synthesis error: {str(e)}"
