"""Shared JSON-extraction helper for parsing LLM replies.

Both the Reasoner and the Actioner need to pull a single balanced JSON
object out of an LLM's free-form reply; this lives in one place so both
use the identical (correct) algorithm instead of each maintaining their
own regex.
"""

from __future__ import annotations


def extract_json_object(text: str) -> str | None:
    """Extract the first balanced ``{...}`` object from free-form text.

    Unlike a greedy ``\\{.*\\}`` regex (which spans from the first '{' to
    the very LAST '}' anywhere in the text -- splicing unrelated content
    together whenever a model emits more than one brace-looking chunk, e.g.
    a JSON object followed by an explanation containing more braces), this
    scans character-by-character and tracks brace depth (respecting quoted
    strings, so braces inside string literals don't confuse the count) to
    find exactly the first complete, balanced object.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def repair_truncated_json(text: str) -> str | None:
    """Best-effort repair of a JSON object that was cut off mid-generation
    (e.g. the model hit its context/token limit before finishing).

    This is a LAST RESORT used only after :func:`extract_json_object` finds
    no complete, balanced object. It closes an unterminated string (if the
    cutoff happened inside one), strips a trailing dangling comma, and then
    appends enough closing ``}``/``]`` to balance whatever was left open.
    The repaired text may drop the tail of a string value, but the
    resulting JSON is syntactically valid, which is much more useful than
    discarding an otherwise-good plan just because a long `expected_outcome`
    string got cut off by a context-window ceiling.

    Returns ``None`` if there's no opening ``{`` at all (nothing to repair).
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    stack: list[str] = []
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
            stack.append("}")
        elif ch == "[":
            depth += 1
            stack.append("]")
        elif ch in "}]":
            if stack:
                stack.pop()
            depth -= 1

    if depth == 0 and not in_string:
        # Already balanced -- extract_json_object should have caught this;
        # nothing to repair.
        return None

    repaired = text[start:]
    if in_string:
        repaired += '"'
    repaired = repaired.rstrip()
    repaired = repaired.rstrip(",")
    repaired += "".join(reversed(stack))
    return repaired
