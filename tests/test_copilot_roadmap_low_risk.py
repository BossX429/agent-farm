"""Tests for Copilot roadmap low-risk items #1, #3, #4.

#1 Completion validation: chunked_write and chunked_analysis now
report validated / written_not_validated, structurally only.
#3 Metrics: per-tool calls, successes, durations, validation counts.
#4 Observability: farm_stats reports metrics and the action-log
summary, without adding a new MCP tool.

No model calls are made here: validation helpers are pure, and
metrics are recorded directly.
"""

import asyncio

from agent_farm.reliability import validate_document_sections, validate_analyses
from agent_farm.metrics import record_tool_run, get_metrics, reset_metrics
from agent_farm.colony_tools import farm_stats, _action_log_summary


def test_validate_document_sections_all_good():
    result = validate_document_sections(
        ["Intro", "Body"], ["Intro text that is long enough.", "Body text that is long enough."]
    )
    assert result["validated"] is True
    assert result["errors"] == []
    assert result["sections_failed"] == 0


def test_validate_document_sections_empty_and_failed():
    result = validate_document_sections(
        ["A", "B", "C"],
        ["Good content that is long enough.", "", "[Section 3 generation failed: timeout]"],
    )
    assert result["validated"] is False
    assert result["sections_empty"] == 1
    assert result["sections_failed"] == 1
    assert any("empty" in e for e in result["errors"])


def test_validate_document_sections_wrong_count():
    result = validate_document_sections(["A", "B"], ["Only one section here, long enough."])
    assert result["validated"] is False
    assert "Expected 2 sections, got 1" in result["errors"]


def test_validate_document_sections_does_not_judge_quality():
    # A complete but nonsense document still validates: validation is
    # structural completion, not correctness.
    result = validate_document_sections(
        ["A"], ["This is complete nonsense text about nothing at all, but it is long."]
    )
    assert result["validated"] is True


def test_validate_analyses_all_good():
    result = validate_analyses(2, [
        {"perspective": "Structure", "analysis": "A solid analysis with enough detail here."},
        {"perspective": "Security", "analysis": "Another solid analysis with enough detail."},
    ])
    assert result["validated"] is True


def test_validate_analyses_error_placeholder():
    result = validate_analyses(2, [
        {"perspective": "Structure", "analysis": "A solid analysis with enough detail here."},
        {"perspective": "Error", "analysis": "timeout"},
    ])
    assert result["validated"] is False
    assert result["perspectives_failed"] == 1


def test_metrics_record_and_average():
    reset_metrics()
    record_tool_run("chunked_write", 2.0, True, items=5, failed_items=0, validation_status="validated")
    record_tool_run("chunked_write", 4.0, False, items=5, failed_items=1, validation_status="written_not_validated")
    metrics = get_metrics()
    entry = metrics["chunked_write"]
    assert entry["calls"] == 2
    assert entry["successes"] == 1
    assert entry["failures"] == 1
    assert entry["avg_seconds"] == 3.0
    assert entry["items"] == 10
    assert entry["failed_items"] == 1
    assert entry["validations"]["validated"] == 1
    assert entry["validations"]["written_not_validated"] == 1
    reset_metrics()


def test_metrics_no_token_cost_invented():
    reset_metrics()
    record_tool_run("chunked_write", 1.0, True, items=2)
    metrics = get_metrics()
    # Token cost is deliberately not tracked: Ollama does not report
    # it uniformly, so no token field should appear.
    assert "token_cost" not in metrics["chunked_write"]
    assert "tokens" not in metrics["chunked_write"]
    reset_metrics()


def test_farm_stats_includes_metrics_and_observability():
    reset_metrics()
    record_tool_run("chunked_analysis", 1.5, True, items=4, validation_status="validated")
    result = asyncio.run(farm_stats())
    assert "metrics" in result
    assert result["metrics"]["chunked_analysis"]["calls"] == 1
    assert "observability" in result
    assert "action_log" in result["observability"]
    assert result["observability"]["action_log"]["enabled"] in (True, False)
    reset_metrics()


def test_action_log_summary_off_by_default():
    # With no AGENT_FARM_ACTION_LOG env and no explicit enable, the
    # summary must report disabled, not crash or invent records.
    from agent_farm.action_log import disable_action_log
    import os
    os.environ.pop("AGENT_FARM_ACTION_LOG", None)
    disable_action_log()
    summary = _action_log_summary()
    assert summary["enabled"] is False
    assert summary["records"] == 0
