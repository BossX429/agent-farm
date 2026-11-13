"""Tests for mechanical verification of review findings.

The bar is the live proof from PR #24: the synthesizer's v2
review produced three findings, all false positives, and
Sourcery produced three real ones. Verification must reject
all three false positives and keep all three real findings.
A check that cannot run (no file content, no code tokens, a
function defined nowhere in the review) must keep the finding:
verification only rejects on positive evidence that a finding
is wrong, never on the absence of evidence.
"""

from agent_farm.review_verify import (
    extract_code_tokens,
    verify_finding,
    verify_findings,
)


def _file_with(content_by_line):
    """Build a file whose given 1-based lines hold the given
    text; every other line is a filler comment."""
    top = max(content_by_line)
    lines = ["# filler" for _ in range(top)]
    for line_no, text in content_by_line.items():
        lines[line_no - 1] = text
    return "\n".join(lines) + "\n"


# The reviewed files, cut down to the parts the findings are
# about. colony_profiles.py defines record_colony_run with the
# correct parameter and records correctness only when the run
# succeeded; probes.py calls it; profiles.py defines
# record_profile_run with seven parameters.
_COLONY = _file_with(
    {
        1: "def record_colony_run(",
        2: "    colony_type, task_type, elapsed, success,",
        3: "    failure_kind=None, correct=None,",
        4: "):",
        5: '    """Record one colony run on its card."""',
        6: "    card = _load_colony_card(colony_type)",
        7: "    correct = correct if success else None",
        8: "    if correct is not None:",
        9: '        card["checked"] += 1',
        10: "        if correct:",
        11: '            card["correct"] += 1',
        12: "    return _save_colony_card(card)",
        13: "",
        14: "def colony_type_pick(task_type):",
        15: "    best = None",
        16: "    best_quality = -1.0",
        17: "    for colony_type, card in _cards().items():",
        18: "        entry = card['per_task_type'][task_type]",
        19: "        if entry['checked'] >= 5:",
        20: "            quality = entry['correct'] / entry['checked']",
        21: "        else:",
        22: "            quality = entry['successes'] / entry['samples']",
        23: "        if quality > best_quality:",
        24: "            best = colony_type",
        25: "            best_quality = quality",
        26: "    return best",
    }
)
_PROBES = _file_with(
    {
        1: "from .colony_profiles import record_colony_run",
        2: "",
        3: "def run_probe(colony_type, task_type):",
        4: '    record_colony_run("heavy", "write", 1.0, True, None, True)',
        5: "    return True",
    }
)
_PROFILES = _file_with(
    {
        1: "def record_profile_run(",
        2: "    role, model, task_type, elapsed, success,",
        3: "    failure_kind=None, correct=None,",
        4: "):",
        5: '    """Record one profile run on its card."""',
        6: "    card = _load_card(role, model)",
        7: "    return _save_card(card)",
    }
)
_TEST_COLONY = _file_with(
    {
        130: "def test_pick_prefers_measured_correctness_over_unmeasured():",
        131: '    # Heavy has been measured and is right for "write"',
        132: "    # (correctness 1.0), but its success rate is lower than",
        133: "    # fast's. Fast has no correctness data yet, so it competes",
        134: "    # on success. A success-only picker picks fast; the",
        135: "    # correctness picker must pick heavy.",
        136: '    _trust_card("fast")',
        137: '    _trust_card("heavy")',
        138: "    for _ in range(4):",
        139: '        assert record_colony_run("fast", "write", 1.0, True)',
        140: '    assert record_colony_run("fast", "write", 1.0, False, "error")',
        141: "    for _ in range(5):",
        142: '        assert record_colony_run("heavy", "write", 1.0, True, None, True)',
        143: "    for _ in range(5):",
        144: '        assert record_colony_run("heavy", "write", 1.0, False, "error")',
        145: '    assert colony_type_pick("write") == "heavy"',
    }
)
_TEST_PROFILER = _file_with(
    {
        609: "def test_failed_run_does_not_count_as_checked():",
        610: "    # A failed response must not count as a checked answer on a",
        611: "    # role card either, even if a caller passes a correct value",
        612: "    # alongside success=False.",
        613: "    from agent_farm.profiles import _load_card, record_profile_run",
        614: "",
        615: '    assert record_profile_run("worker", "qwen3:4b", "write", 1.0, False, "error", True)',
        616: '    card = _load_card("worker", "qwen3:4b")',
        617: '    assert card["checked"] == 0',
    }
)

_CONTENTS = {
    "agent_farm/colony_profiles.py": _COLONY,
    "agent_farm/probes.py": _PROBES,
    "agent_farm/profiles.py": _PROFILES,
    "tests/test_colony_profiles.py": _TEST_COLONY,
    "tests/test_profiler_phase2.py": _TEST_PROFILER,
}


def _finding(severity, title, file, line, body):
    return {
        "severity": severity,
        "title": title,
        "file": file,
        "line": line,
        "body": body,
    }


# The synthesizer's three v2 false positives on PR #24, in its
# own words. All three must be rejected.
_FP_MISSING_PARAM = _finding(
    "HIGH",
    "Missing implementation for new function parameter",
    "agent_farm/probes.py",
    4,
    "The code adds a new parameter `correct` to the "
    "`record_colony_run` function call, but there is no "
    "indication that the implementation of `record_colony_run` "
    "was updated to accept this new parameter. This would cause "
    "a TypeError when the code runs.",
)
_FP_KEY_ERROR = _finding(
    "HIGH",
    "Test assumes key exists without checking",
    "tests/test_colony_profiles.py",
    138,
    "The test assumes `normalized[\"per_task_type\"][\"write\"]` "
    "exists, but the test data has `per_task_type` as an empty "
    "dict. If the code doesn't populate `per_task_type` with a "
    '"write" entry, the test will fail with a KeyError.',
)
_FP_TOO_MANY_ARGS = _finding(
    "MEDIUM",
    "Test passes invalid parameter to record_profile_run",
    "tests/test_profiler_phase2.py",
    612,
    "The test passes 7 arguments to record_profile_run, but the "
    "function likely accepts only 5 or 6 arguments (based on "
    "other test usage patterns). The extra 'True' parameter will "
    "cause a TypeError when the test runs.",
)

# Sourcery's three real findings on PR #24. All three must be kept.
_REAL_FAILED_CORRECTNESS = _finding(
    "MEDIUM",
    "Failed answers inflate correctness",
    "agent_farm/colony_profiles.py",
    8,
    "When a caller passes a non-None `correct` value together "
    "with `success=False`, `record_colony_run` increments "
    "`checked` and possibly `correct` whenever `correct` is "
    "non-None, even when `success` is false. Only record "
    "correctness when `success` is true, or reject inconsistent "
    "arguments.",
)
_REAL_OUTRANK = _finding(
    "MEDIUM",
    "Wrong answers can outrank unmeasured types",
    "agent_farm/colony_profiles.py",
    20,
    "When a colony type has measured `correctness` of 0.0 on a "
    "task type, `colony_type_pick` ranks it below unmeasured "
    "types. A type that is always wrong should rank below every "
    "type with a real rate.",
)
_REAL_PICK_IGNORED = _finding(
    "MEDIUM",
    "Correctness ranking can be ignored",
    "tests/test_colony_profiles.py",
    145,
    "The pick test records runs with equal success and speed, so "
    "a picker that ignores `correctness` still passes. Make fast "
    "faster and higher-success so only a correctness-aware "
    "`colony_type_pick` chooses heavy for \"write\".",
)


def test_verify_rejects_all_three_v2_false_positives():
    kept, rejected = verify_findings(
        [_FP_MISSING_PARAM, _FP_KEY_ERROR, _FP_TOO_MANY_ARGS],
        _CONTENTS,
    )
    assert kept == []
    assert len(rejected) == 3
    reasons = " ".join(reason for _f, reason in rejected)
    assert "does accept parameter" in reasons
    assert "normalized" in reasons
    assert "blank or a comment" in reasons


def test_verify_keeps_all_three_real_sourcery_findings():
    kept, rejected = verify_findings(
        [_REAL_FAILED_CORRECTNESS, _REAL_OUTRANK, _REAL_PICK_IGNORED],
        _CONTENTS,
    )
    assert rejected == []
    assert [f["title"] for f in kept] == [
        "Failed answers inflate correctness",
        "Wrong answers can outrank unmeasured types",
        "Correctness ranking can be ignored",
    ]


def test_verify_keeps_finding_with_no_file_content():
    # Verification only rejects on positive evidence. A finding
    # whose file was not fetched (deleted, binary) is kept.
    finding = _finding("HIGH", "Something", "gone.py", 10, "Body.")
    kept, rejected = verify_findings([finding], {})
    assert kept == [finding]
    assert rejected == []


def test_verify_rejects_line_past_end_of_file():
    finding = _finding("HIGH", "Bad line", "agent_farm/probes.py", 999, "Body.")
    kept, rejected = verify_findings([finding], _CONTENTS)
    assert kept == []
    assert "past the end of the file" in rejected[0][1]


def test_verify_keeps_finding_with_no_code_tokens():
    # A finding that names no code cannot be checked by the
    # token or signature checks, so it is kept.
    finding = _finding(
        "LOW",
        "Unclear wording",
        "agent_farm/probes.py",
        4,
        "This comment is confusing and should be rewritten.",
    )
    kept, rejected = verify_findings([finding], _CONTENTS)
    assert kept == [finding]
    assert rejected == []


def test_signature_check_keeps_when_function_not_in_review():
    # The finding is about a real call on the cited line, but the
    # function's definition lives in a file that is NOT in this
    # review, so the signature check cannot run and the finding
    # is kept. (A finding that names code appearing nowhere in
    # the file is rejected by the token check instead.)
    finding = _finding(
        "HIGH",
        "Missing parameter",
        "agent_farm/probes.py",
        4,
        "The call passes a new parameter `correct` to "
        "`record_colony_run`, but it was not updated to accept "
        "this new parameter.",
    )
    contents = {
        path: text
        for path, text in _CONTENTS.items()
        if path != "agent_farm/colony_profiles.py"
    }
    kept, rejected = verify_findings([finding], contents)
    assert kept == [finding]
    assert rejected == []


def test_arg_count_rejects_when_call_fits_signature():
    # The finding claims the call passes 8 arguments, but the
    # real call on line 615 passes 7 and the signature accepts
    # 7, so the claim is wrong. The check counts the real call,
    # not the claimed number.
    finding = _finding(
        "HIGH",
        "Too many arguments",
        "tests/test_profiler_phase2.py",
        615,
        "The test passes 8 arguments to record_profile_run, but "
        "the function accepts only 7.",
    )
    kept, rejected = verify_findings([finding], _CONTENTS)
    assert kept == []
    assert "accepts 7 parameters" in rejected[0][1]
    assert "passes 7" in rejected[0][1]


def test_extract_code_tokens_picks_up_snake_case():
    finding = _finding(
        "MEDIUM",
        "Title",
        "agent_farm/probes.py",
        4,
        "The record_colony_run call passes `correct` and the "
        'card["checked"] count is wrong.',
    )
    tokens = extract_code_tokens(finding)
    assert "record_colony_run" in tokens
    assert "correct" in tokens
    assert "card" in tokens
    assert "checked" in tokens


def test_verify_finding_returns_reason_for_rejection():
    keep, reason = verify_finding(_FP_MISSING_PARAM, _CONTENTS)
    assert keep is False
    assert "record_colony_run" in reason
    assert "correct" in reason
