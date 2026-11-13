"""Mechanical verification of PR review findings.

A local review model will invent findings about code that does not
exist: a parameter the function does accept, a dict access on a
line that is a for loop, a key the file does initialize. The
model cannot be trusted to check itself, so every finding is
checked against the actual file before it is reported.

verify_findings(findings, file_contents) runs five deterministic
checks on each finding and drops the ones that fail:

1. Line check -- the cited line exists in the file and is not
   blank or a comment. A finding that points at a line that is
   not there, or at nothing, is not grounded.
2. Token check -- the code the finding names (backticked
   identifiers, x["key"] accesses, dotted names) must appear in
   a window around the cited line (15 lines either side). The
   synthesizer once cited line 144 for a dict access; line 144
   was a for loop, and the named code was nowhere near it.
3. Signature check -- when a finding claims a function is called
   with a parameter it does not accept, parse the files and read
   the function's real signature. The definition may live in
   another file in the same review (the finding cites the call
   site), so every reviewed file is searched. If the parameter
   is accepted, the finding is wrong and is dropped. If the
   function is not in any reviewed file, or a file is not
   Python, the check cannot run and the finding is kept: a check
   that cannot run must not reject.
4. Argument-count check -- when a finding claims a call passes
   too many arguments, count the arguments on the cited line and
   compare them with the function's real parameter list. If the
   call fits, the finding is wrong and is dropped. A multi-line
   call does not parse from one line alone, so the check cannot
   run there and the finding is kept.
5. Quote check -- any code a finding QUOTES must appear in the
   file or the diff. A finding that quotes an assertion or an
   expression that is nowhere to be found hallucinated it. A
   lone identifier is a token, not a quote, and is checked by
   the token check.

A finding with no extractable code tokens cannot be checked by
the token or signature checks, so it is kept. Verification only
ever rejects a finding on positive evidence that it is wrong
(the code it names is absent, or the signature contradicts it),
never on the absence of evidence.
"""

import ast
import re

LINE_WINDOW = 15

_BACKTICK_RE = re.compile(r"`([^`]+)`")
_SUBSCRIPT_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\[[\"']([^\"']+)[\"']\]")
_DOTTED_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*)\b")
# snake_case identifiers in plain text are code, not English:
# English words do not contain underscores.
_SNAKE_RE = re.compile(r"\b([a-z][a-z0-9]*_[a-z0-9_]+)\b")


def extract_code_tokens(finding):
    """Code-like tokens a finding names: backticked identifiers,
    subscript bases and keys, dotted names. Plain English words
    are not tokens; only things that look like code count."""
    text = " ".join(
        str(finding.get(key, ""))
        for key in ("title", "body")
    )
    tokens = set()
    for span in _BACKTICK_RE.findall(text):
        # A backticked span may hold a call or an access; split
        # it into its code parts.
        tokens.update(_split_code_span(span))
    for base, key in _SUBSCRIPT_RE.findall(text):
        tokens.add(base)
        tokens.add(key)
    for dotted in _DOTTED_RE.findall(text):
        tokens.add(dotted)
        tokens.add(dotted.split(".")[0])
    for snake in _SNAKE_RE.findall(text):
        tokens.add(snake)
    return {t for t in tokens if t and len(t) >= 2}


def _split_code_span(span):
    """Split one backticked span into identifier-like parts."""
    parts = set()
    for piece in re.split(r"[^A-Za-z0-9_]+", span):
        if len(piece) >= 2:
            parts.add(piece)
    # Keep a dotted name whole too, so "a.b" matches "a.b".
    if "." in span:
        parts.add(span.strip("()"))
    return parts


def _line_is_code(line):
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def _extract_code_quotes(finding):
    """Code a finding quotes: backticked spans that hold more
    than one identifier (a space, an operator, a call, an
    assertion). A lone identifier is a token, not a quote, and
    is checked by the token check. A quote is a claim that the
    code reads exactly that way, so it must appear in the file
    or the diff."""
    text = " ".join(
        str(finding.get(key, ""))
        for key in ("title", "body")
    )
    quotes = set()
    for span in _BACKTICK_RE.findall(text):
        stripped = span.strip()
        # A quote holds code structure: a call, a keyword-led
        # statement, or a comparison with an identifier on
        # either side. A bare `name = literal` assignment is an
        # example value, not a quote of the file, and a
        # `name=value` pair describes arguments, not code, so
        # neither counts.
        if len(stripped) < 4:
            continue
        has_call = re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*\(", stripped) is not None
        has_keyword = stripped.split(None, 1)[0] in (
            "assert", "if", "return", "def", "for", "while",
            "raise", "import", "elif", "else",
        )
        has_comparison = re.search(
            r"[A-Za-z_][A-Za-z0-9_]*\s*(==|!=|<=|>=)\s*\S", stripped
        ) is not None or re.search(
            r"\S\s*(==|!=|<=|>=)\s*[A-Za-z_]", stripped
        ) is not None
        if has_call or has_keyword or has_comparison:
            quotes.add(stripped)
    return quotes


def _normalize_code(text):
    """Collapse all whitespace to single spaces, so a quote
    matches the file even when the model reflows it."""
    return re.sub(r"\s+", " ", text).strip()


def _check_quotes(finding, file_content, diff_text=""):
    """Every code quote in the finding must appear in the file
    or in the diff. A finding that quotes code that is nowhere
    to be found hallucinated it. Returns a rejection reason,
    or None."""
    quotes = _extract_code_quotes(finding)
    if not quotes:
        return None
    haystacks = [_normalize_code(file_content)]
    if diff_text:
        haystacks.append(_normalize_code(diff_text))
    for quote in sorted(quotes):
        norm = _normalize_code(quote)
        if not any(norm in hay for hay in haystacks):
            return f"quoted code not found in the file or the diff: {quote[:80]}"
    return None


def _check_line(finding, lines):
    """The cited line exists and is code. Returns a rejection
    reason, or None when the check passes or cannot run."""
    line_no = finding.get("line")
    if not isinstance(line_no, int) or line_no < 1:
        return "cited line is not a positive line number"
    if line_no > len(lines):
        return f"cited line {line_no} is past the end of the file ({len(lines)} lines)"
    if not _line_is_code(lines[line_no - 1]):
        return f"cited line {line_no} is blank or a comment, not code"
    return None


def _check_tokens(finding, lines, diff_text=""):
    """A code token the finding names must appear within a window
    around the cited line. When the finding names an access
    (x["key"] or x.y), the access base is the token that must
    appear: a bare word like "write" shows up everywhere, but
    the base names the exact thing the finding is about. Returns
    a rejection reason, or None."""
    tokens = extract_code_tokens(finding)
    if not tokens:
        return None
    text = " ".join(
        str(finding.get(key, ""))
        for key in ("title", "body")
    )
    # Access bases are the specific tokens; fall back to any
    # token when the finding names no access.
    access_bases = {base for base, _key in _SUBSCRIPT_RE.findall(text)}
    access_bases.update(
        dotted.split(".")[0] for dotted in _DOTTED_RE.findall(text)
    )
    required = access_bases or tokens
    line_no = finding.get("line")
    if not isinstance(line_no, int) or line_no < 1:
        return None
    start = max(0, line_no - 1 - LINE_WINDOW)
    end = min(len(lines), line_no + LINE_WINDOW)
    window = "\n".join(lines[start:end])
    if any(token in window for token in required):
        return None
    # A finding about removed code names code that lives only in
    # the diff, not in the new file. If any required token is in
    # the diff, the token check passes.
    if diff_text and any(token in diff_text for token in required):
        return None
    return (
        f"none of the named code ({sorted(required)[:5]}) appears "
        f"within {LINE_WINDOW} lines of cited line {line_no}"
    )


def _function_params(source, name):
    """The parameter names of a function defined in source, or
    None when the file is not Python or the function is not
    defined there."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name != name:
                continue
            args = node.args
            params = [a.arg for a in args.posonlyargs]
            params += [a.arg for a in args.args]
            params += [a.arg for a in args.kwonlyargs]
            if args.vararg is not None:
                params.append(args.vararg.arg)
            if args.kwarg is not None:
                params.append(args.kwarg.arg)
            return params
    return None


def _check_signature(finding, file_contents):
    """When a finding claims a function is called with a
    parameter it does not accept, read the real signature. The
    definition may live in another file in the same review (the
    finding cites the call site, not the definition), so every
    reviewed file is searched. Only the literal word "parameter"
    triggers this check: a finding that says "reject inconsistent
    arguments" is not claiming a parameter is missing, it is
    asking for input validation. The parameter is a token named
    near the word "parameter"; if that token is in the function's
    parameter list, the finding is wrong. Returns a rejection
    reason, or None."""
    text = " ".join(
        str(finding.get(key, ""))
        for key in ("title", "body")
    )
    if "parameter" not in text.lower():
        return None
    tokens = extract_code_tokens(finding)
    if not tokens:
        return None
    # The parameter is a token named near the word "parameter".
    param_candidates = set()
    lower = text.lower()
    for match in re.finditer(r"parameter", lower):
        window = text[max(0, match.start() - 50):match.end() + 50]
        for token in tokens:
            if token in window:
                param_candidates.add(token)
    if not param_candidates:
        return None
    for func_name in tokens:
        for source in file_contents.values():
            params = _function_params(source, func_name)
            if params is None:
                continue
            for token in param_candidates:
                if token != func_name and token in params:
                    return (
                        f"{func_name} does accept parameter "
                        f"'{token}': the finding claims otherwise"
                    )
    return None


def _call_arg_count(line, func_name):
    """How many arguments a call to func_name on this line passes,
    or None when the line does not parse or holds no such call.
    A multi-line call does not parse from one line alone, so it
    returns None and the check cannot run. The line is dedented
    first: a call inside a function or a test is indented, and
    ast.parse rejects leading indentation at module level."""
    import textwrap

    try:
        tree = ast.parse(textwrap.dedent(line))
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == func_name:
            return len(node.args) + len(node.keywords)
        if isinstance(func, ast.Attribute) and func.attr == func_name:
            return len(node.args) + len(node.keywords)
    return None


def _check_arg_count(finding, lines, file_contents):
    """When a finding claims a call passes too many arguments,
    count the arguments on the cited line and compare them with
    the function's real parameter list. The function is any token
    in the finding that is defined in a reviewed file; the cited
    line must hold a call to it. If the call fits the signature,
    the finding is wrong. Returns a rejection reason, or None."""
    text = " ".join(
        str(finding.get(key, ""))
        for key in ("title", "body")
    )
    if "argument" not in text.lower():
        return None
    line_no = finding.get("line")
    if not isinstance(line_no, int) or line_no < 1 or line_no > len(lines):
        return None
    tokens = extract_code_tokens(finding)
    for func_name in tokens:
        given = _call_arg_count(lines[line_no - 1], func_name)
        if given is None:
            continue
        for source in file_contents.values():
            params = _function_params(source, func_name)
            if params is None:
                continue
            if given <= len(params):
                return (
                    f"{func_name} accepts {len(params)} parameters "
                    f"and the call passes {given}: the finding "
                    f"claims too many arguments"
                )
    return None


def verify_finding(finding, file_contents, diff_texts=None):
    """Check one finding against the files in its review. The
    line, token, and quote checks use the file the finding
    cites; the signature check searches every reviewed file,
    because the definition may live elsewhere; the quote check
    also searches the diff, because a finding may quote a line
    the diff removed. Returns (keep, reason): keep is True when
    the finding survives, and reason names why it was dropped,
    or None when kept."""
    contents = file_contents or {}
    diffs = diff_texts or {}
    content = contents.get(finding.get("file", ""), "")
    if not content:
        return True, None
    lines = content.splitlines()
    reason = _check_line(finding, lines)
    if reason is not None:
        return False, reason
    diff_text = diffs.get(finding.get("file", ""), "")
    reason = _check_tokens(finding, lines, diff_text)
    if reason is not None:
        return False, reason
    reason = _check_quotes(finding, content, diff_text)
    if reason is not None:
        return False, reason
    reason = _check_signature(finding, contents)
    if reason is not None:
        return False, reason
    reason = _check_arg_count(finding, lines, contents)
    if reason is not None:
        return False, reason
    return True, None


def verify_findings(findings, file_contents, diff_texts=None):
    """Verify every finding against the files in its review.
    Returns (kept, rejected), where rejected is a list of
    (finding, reason) pairs. A finding whose file has no content
    is kept: verification only rejects on positive evidence."""
    kept = []
    rejected = []
    for finding in findings:
        keep, reason = verify_finding(finding, file_contents, diff_texts)
        if keep:
            kept.append(finding)
        else:
            rejected.append((finding, reason))
    return kept, rejected
