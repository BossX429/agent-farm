"""Real-model smoke test for agent-farm's chunked tools.

This is NOT a pytest test and it is NOT run in CI. It needs a real
Ollama server running locally with a model already pulled:

    ollama pull qwen3:4b
    ollama serve   # or let Ollama start automatically

    python scripts/run_real_model_smoke.py
    python scripts/run_real_model_smoke.py --model qwen3:0.6b   # small CPU test

It runs one small job through each chunked tool -- write, code
generation, and analysis -- with every model tier pointed at the
chosen model, and prints a JSON summary: status, validation outcome,
and elapsed seconds for each job. Outputs go to a temporary directory
under /tmp, which is inside agent-farm's write allowlist.

Read the results honestly:

* "validated" means structurally complete, NOT correct, good, or
  fast. A validated document can still be mediocre; a validated
  analysis can still be shallow. That is by design -- see
  validate_document_sections / validate_python_code / validate_analyses.
* "written_not_validated" means the file was written but failed its
  check. That is the safety net working, not a pass.
* One run of each tool is a sample size of one. It proves the
  plumbing works against a real model. It does not prove quality,
  speed, or reliability in general, and it says nothing about a
  different model or machine.

On a CPU-only machine with a tiny model, expect each job to take
roughly 1-3 minutes. On a GPU machine with qwen3:4b, expect much less.
If Ollama is not reachable, the tools return error markers instead
of raising, so a run with every job "failed" usually means Ollama
was not running, not that agent-farm crashed.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_farm import state  # noqa: E402
from agent_farm.chunked_tools import (  # noqa: E402
    chunked_analysis,
    chunked_code_gen,
    chunked_write,
)


def _summary(result: dict, elapsed: float) -> dict:
    return {
        "status": result.get("status"),
        "validation_status": result.get("validation_status"),
        "validation": result.get("validation"),
        "elapsed_seconds": round(elapsed, 2),
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3:4b",
                        help="Ollama model for every tier (default: qwen3:4b)")
    parser.add_argument("--out-dir", default="/tmp/agent-farm-smoke",
                        help="Directory for generated outputs (default: /tmp/agent-farm-smoke)")
    args = parser.parse_args()

    for tier in state.MODEL_TIERS:
        state.MODEL_TIERS[tier] = args.model
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = {"model": args.model, "ollama_url": state.OLLAMA_URL, "jobs": {}}

    t0 = time.time()
    write = await chunked_write(
        str(out / "smoke-report.md"),
        "A short guide to making tea", 3)
    summary["jobs"]["chunked_write"] = _summary(write, time.time() - t0)

    t0 = time.time()
    code = await chunked_code_gen(
        str(out / "smoke-utils.py"),
        "Simple utilities for working with a list of numbers", "python", 3)
    summary["jobs"]["chunked_code_gen"] = _summary(code, time.time() - t0)

    t0 = time.time()
    analysis = await chunked_analysis(
        str(out / "smoke-report.md"),
        "Is this guide clear and correct for a beginner making tea?", 3)
    summary["jobs"]["chunked_analysis"] = _summary(analysis, time.time() - t0)

    print(json.dumps(summary, indent=2, default=str))
    print(f"\nOutputs in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
