"""MCP tools for chunked writes, code generation, and analysis."""

import os
import uuid
import json
import httpx
import asyncio
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from .mcp_app import mcp
from .state import MODEL_TIERS, colonies, stats
from .colony import spawn_colony_internal
from .ollama import query_ollama, sync_query_ollama, sync_query_raw, synthesize_swarm_results
from .reliability import (
    MODEL_TASK_TIMEOUT_SECONDS,
    run_parallel,
    validate_python_code,
)


# ============ CHUNKED WRITE PATTERN ============
# Bugs generate small pieces, Python concatenates directly

SECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"}
                }
            }
        }
    },
    "required": ["sections"]
}


def plan_sections(spec: str, num_sections: int = 5) -> List[Dict]:
    """
    Use a bug to plan document sections.
    Returns list of {title, description} for each section.
    """
    from .agent import query_ollama_structured
    
    prompt = f"""Create an outline with exactly {num_sections} sections for this document:

SPECIFICATION: {spec}

Return JSON with a "sections" array. Each section needs:
- title: Short section title
- description: 1-2 sentence description of what this section should cover

Make sections logical and sequential. Keep descriptions under 100 chars."""

    response = query_ollama_structured(
        MODEL_TIERS["learner"],  # Use learner for planning
        prompt,
        SECTION_SCHEMA
    )
    
    if "error" in response:
        # Fallback: generate generic sections
        return [
            {"title": f"Section {i+1}", "description": f"Part {i+1} of the document about: {spec[:50]}"}
            for i in range(num_sections)
        ]
    
    sections = response.get("sections", [])
    if len(sections) < num_sections:
        # Pad with generic sections
        for i in range(len(sections), num_sections):
            sections.append({"title": f"Section {i+1}", "description": "Additional content"})
    
    return sections[:num_sections]


def generate_section_content(section: Dict, spec: str, section_num: int, total: int) -> str:
    """
    Have a worker bug generate content for one section.
    Keeps output under 400 chars to avoid corruption.
    """
    from .agent import query_ollama_sync
    
    prompt = f"""Write section {section_num} of {total} for this document.

DOCUMENT TOPIC: {spec}

SECTION TITLE: {section['title']}
SECTION SHOULD COVER: {section['description']}

RULES:
- Write 3-5 paragraphs for this section ONLY
- Be concise but informative
- Do NOT include other sections
- Do NOT add headers for other sections
- Start with the section title as a header

Write the section content now:"""

    response = query_ollama_sync(MODEL_TIERS["worker"], prompt)
    
    # Clean up response
    content = response.strip()
    
    # Remove any thinking tags
    import re
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL)
    
    return content


def _run_section_generation(sections: List[Dict], spec: str, num_sections: int) -> List[str]:
    """Sync parallel section generation, run via asyncio.to_thread.

    The caller is an async MCP tool: running this blocking parallel
    work in a thread keeps the event loop free for other requests
    while the model calls complete.
    """
    callables = [
        (lambda s=sections[i], n=i + 1: generate_section_content(s, spec, n, num_sections))
        for i in range(num_sections)
    ]
    results = run_parallel(
        callables, max_workers=min(4, num_sections),
        timeout=MODEL_TASK_TIMEOUT_SECONDS,
    )
    contents = []
    for idx, result in enumerate(results):
        if isinstance(result, str):
            contents.append(result)
        elif isinstance(result, dict):
            contents.append(f"[Section {idx+1} generation failed: {result.get('error', 'unknown error')}]")
        else:
            contents.append(f"[Section {idx+1} generation failed: {result}]")
    return contents


@mcp.tool()
async def chunked_write(
    output_path: str,
    spec: str,
    num_sections: int = 5,
    doc_type: str = "markdown"
) -> dict:
    """
    Generate large documents by having bugs write sections in parallel.
    Bypasses the long-content limitation by chunking work.
    
    HOW IT WORKS:
    1. Planner bug creates outline (structured JSON)
    2. Worker bugs generate sections in PARALLEL
    3. Python concatenates directly (NO LLM involved)
    4. heavy_write saves result
    
    output_path: Where to save the document
    spec: What the document should be about
    num_sections: How many sections (default 5, max 10)
    doc_type: 'markdown', 'text', or 'code'
    
    EXAMPLE:
    chunked_write("/tmp/report.md", "Analysis of Python best practices", 5)
    """
    import time
    start = time.time()

    # Clamp sections
    num_sections = max(2, min(10, num_sections))

    # Step 1: Plan sections -- a blocking model call, so run it in a
    # thread instead of blocking the async event loop.
    sections = await asyncio.to_thread(plan_sections, spec, num_sections)

    # Step 2: Generate sections in parallel, also off the event loop.
    section_contents = await asyncio.to_thread(
        _run_section_generation, sections, spec, num_sections
    )

    # Step 3: Concatenate directly (NO LLM)
    if doc_type == "markdown":
        header = f"# {spec}\n\n"
        separator = "\n\n---\n\n"
    elif doc_type == "code":
        header = f"# {spec}\n# Auto-generated by Agent Farm chunked_write\n\n"
        separator = "\n\n"
    else:
        header = f"{spec}\n{'='*len(spec)}\n\n"
        separator = "\n\n"
    
    full_content = header + separator.join(section_contents)
    
    # Step 4: Direct write (bypasses LLM), off the event loop.
    from .tools import tool_write_file
    write_result = await asyncio.to_thread(tool_write_file, output_path, full_content)

    elapsed = time.time() - start

    return {
        "status": "completed" if "error" not in write_result else "failed",
        "output_path": output_path,
        "spec": spec,
        "sections_planned": len(sections),
        "sections_generated": len([s for s in section_contents if s]),
        "total_chars": len(full_content),
        "total_lines": full_content.count('\n') + 1,
        "section_titles": [s["title"] for s in sections],
        "write_result": write_result,
        "elapsed_seconds": round(elapsed, 2)
    }


def _generate_function_sync(func: Dict, spec: str, language: str) -> str:
    """Generate one function (sync, run via asyncio.to_thread)."""
    from .agent import query_ollama_sync

    prompt = f"""Write ONE {language} function:

FUNCTION NAME: {func['title']}
PURPOSE: {func['description']}
CONTEXT: Part of a module for: {spec}

RULES:
- Write ONLY this one function
- Include docstring/comments
- Keep it focused and clean
- Do NOT include imports (those go at top)
- Do NOT include other functions

Write the function now:"""
    response = query_ollama_sync(MODEL_TIERS["worker"], prompt)
    import re
    content = response.strip()
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    content = re.sub(r'```\w*\n?', '', content)
    content = re.sub(r'```', '', content)
    return content.strip()


def _run_function_generation(functions: List[Dict], spec: str, language: str) -> List[str]:
    """Sync parallel function generation, run via asyncio.to_thread."""
    callables = [
        (lambda f=functions[i]: _generate_function_sync(f, spec, language))
        for i in range(len(functions))
    ]
    results = run_parallel(
        callables, max_workers=min(4, len(functions)),
        timeout=MODEL_TASK_TIMEOUT_SECONDS,
    )
    contents = []
    for idx, result in enumerate(results):
        if isinstance(result, str):
            contents.append(result)
        elif isinstance(result, dict):
            contents.append(f"# Function {idx} generation failed: {result.get('error', 'unknown error')}")
        else:
            contents.append(f"# Function {idx} generation failed: {result}")
    return contents


@mcp.tool()
async def chunked_code_gen(
    output_path: str,
    spec: str,
    language: str = "python",
    num_functions: int = 4
) -> dict:
    """
    Generate code files by having bugs write functions in parallel.
    Each bug writes one function, Python assembles the file.
    
    output_path: Where to save the code file
    spec: What the code should do
    language: 'python', 'javascript', 'bash', etc.
    num_functions: How many functions to generate (default 4, max 8)
    
    EXAMPLE:
    chunked_code_gen("/tmp/utils.py", "File utilities: read, write, copy, delete", "python", 4)
    """
    import time
    from .agent import query_ollama_structured
    start = time.time()

    num_functions = max(2, min(8, num_functions))

    # Step 1: Plan functions -- blocking model call, off the event loop.
    plan_prompt = f"""Plan {num_functions} functions for this code:

SPECIFICATION: {spec}
LANGUAGE: {language}

Return JSON with "sections" array where each item has:
- title: function name (e.g., "read_file", "calculate_sum")
- description: what the function does (1 sentence)

Make functions logical and complementary."""

    plan_response = await asyncio.to_thread(
        query_ollama_structured,
        MODEL_TIERS["learner"],
        plan_prompt,
        SECTION_SCHEMA,
    )

    functions = plan_response.get("sections", [])
    if len(functions) < num_functions:
        for i in range(len(functions), num_functions):
            functions.append({"title": f"helper_{i}", "description": "Helper function"})
    functions = functions[:num_functions]

    # Step 2: Generate functions in parallel, off the event loop.
    function_contents = await asyncio.to_thread(
        _run_function_generation, functions, spec, language
    )

    # Step 3: Assemble file (NO LLM)
    if language == "python":
        header = f'''"""
{spec}
Auto-generated by Agent Farm chunked_code_gen
"""

'''
        separator = "\n\n\n"
    elif language in ["javascript", "typescript"]:
        header = f'''/**
 * {spec}
 * Auto-generated by Agent Farm chunked_code_gen
 */

'''
        separator = "\n\n"
    elif language == "bash":
        header = f'''#!/bin/bash
# {spec}
# Auto-generated by Agent Farm chunked_code_gen

'''
        separator = "\n\n"
    else:
        header = f"// {spec}\n// Auto-generated by Agent Farm\n\n"
        separator = "\n\n"
    
    full_content = header + separator.join(function_contents)

    # Step 4: Write directly, off the event loop.
    from .tools import tool_write_file
    write_result = await asyncio.to_thread(tool_write_file, output_path, full_content)

    # Step 5: Mandatory validation, before declaring the code complete.
    # "written" means the file was saved; "validated" additionally
    # means it compiles with no duplicate top-level functions and no
    # failed generation fragments. Validation never runs the code.
    # Non-Python languages are not validated here, so they stay
    # "written_not_validated" rather than being claimed as working.
    if "error" in write_result:
        validation = {
            "syntax_ok": False,
            "errors": ["Write failed; validation not run"],
            "warnings": [],
            "validated": False,
            "function_names": [],
            "function_count": 0,
        }
        status = "failed"
    elif language == "python":
        validation = await asyncio.to_thread(validate_python_code, full_content)
        status = "validated" if validation["validated"] else "written_not_validated"
    else:
        validation = {
            "syntax_ok": None,
            "errors": [],
            "warnings": [f"Validation is only implemented for python, not {language}"],
            "validated": False,
            "function_names": [f["title"] for f in functions],
            "function_count": len(functions),
        }
        status = "written_not_validated"

    elapsed = time.time() - start

    return {
        "status": status,
        "validation_status": "validated" if status == "validated" else (
            "failed" if status == "failed" else "written_not_validated"
        ),
        "validation": validation,
        "output_path": output_path,
        "language": language,
        "spec": spec,
        "functions_planned": len(functions),
        "functions_generated": len([f for f in function_contents if f]),
        "function_names": [f["title"] for f in functions],
        "total_chars": len(full_content),
        "total_lines": full_content.count('\n') + 1,
        "write_result": write_result,
        "elapsed_seconds": round(elapsed, 2)
    }


def _analyze_perspective_sync(persp: Dict, target: str, question: str) -> Dict:
    """One perspective analysis (sync, run via asyncio.to_thread)."""
    from .agent import query_ollama_sync

    prompt = f"""Analyze from the {persp['name']} perspective.

TARGET: {target}
QUESTION: {question}
FOCUS ON: {persp['focus']}

Provide a focused analysis (3-5 key points) from this perspective only.
Be specific and actionable."""
    response = query_ollama_sync(MODEL_TIERS["learner"], prompt)
    import re
    content = response.strip()
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    return {
        "perspective": persp["name"],
        "focus": persp["focus"],
        "analysis": content,
    }


def _run_perspective_analysis(perspectives: List[Dict], target: str, question: str) -> List[Dict]:
    """Sync parallel perspective analysis, run via asyncio.to_thread."""
    callables = [
        (lambda p=p: _analyze_perspective_sync(p, target, question))
        for p in perspectives
    ]
    results = run_parallel(
        callables, max_workers=min(4, len(perspectives)),
        timeout=MODEL_TASK_TIMEOUT_SECONDS,
    )
    analyses = []
    for result in results:
        if isinstance(result, dict) and "analysis" in result:
            analyses.append(result)
        elif isinstance(result, dict):
            analyses.append({"perspective": "Error", "analysis": result.get("error", "unknown error")})
        else:
            analyses.append({"perspective": "Error", "analysis": str(result)})
    return analyses


@mcp.tool()
async def chunked_analysis(
    target: str,
    question: str,
    num_perspectives: int = 4
) -> dict:
    """
    Analyze something from multiple perspectives in parallel.
    Each bug analyzes from a different angle, results are synthesized.
    
    target: What to analyze (file path, concept, code, etc.)
    question: The analysis question
    num_perspectives: How many different angles (default 4)
    
    EXAMPLE:
    chunked_analysis("/home/kyle/repos/project", "What are the main architectural patterns?", 4)
    """
    import time
    start = time.time()

    num_perspectives = max(2, min(6, num_perspectives))
    
    # Define analysis perspectives
    perspectives = [
        {"name": "Structure", "focus": "organization, layout, components"},
        {"name": "Patterns", "focus": "design patterns, conventions, idioms"},
        {"name": "Quality", "focus": "code quality, best practices, issues"},
        {"name": "Performance", "focus": "efficiency, bottlenecks, optimization"},
        {"name": "Security", "focus": "vulnerabilities, risks, safety"},
        {"name": "Maintainability", "focus": "readability, modularity, documentation"},
    ][:num_perspectives]
    
    # Generate analyses in parallel, off the event loop.
    analyses = await asyncio.to_thread(
        _run_perspective_analysis, perspectives, target, question
    )

    # Synthesize -- also a blocking model call, off the event loop.
    synthesis = await asyncio.to_thread(
        synthesize_swarm_results,
        [{"answer": a["analysis"], "role": a["perspective"]} for a in analyses],
        task_context=f"Multi-perspective analysis of {target}: {question}",
    )
    
    elapsed = time.time() - start
    
    return {
        "target": target,
        "question": question,
        "perspectives": len(analyses),
        "analyses": analyses,
        "synthesis": synthesis,
        "elapsed_seconds": round(elapsed, 2)
    }
