"""Parse the project's models.txt roster into reasoner/actioner model lists.

models.txt (in the project root) separates locally-available Ollama models
into two roles:

    [reasoners]
    qwen3.5:9b
    ...

    [actioners]
    ornith:9b
    ...

This module just parses that file; experiments/run_grid_search.py uses it to
build every reasoner x actioner combination to benchmark.
"""

from __future__ import annotations

from pathlib import Path


def load_model_roster(path: str | Path = "models.txt") -> dict[str, list[str]]:
    """Parse models.txt into ``{"reasoners": [...], "actioners": [...]}``.

    Blank lines and lines starting with ``#`` are ignored. A ``[section]``
    header switches which list subsequent model lines are appended to.
    Unknown section names are ignored (so extra commentary sections don't
    break parsing) and duplicate model ids are preserved -- if the person
    lists a model twice by mistake we treat it as intentional rather than
    silently deduping.

    Raises
    ------
    FileNotFoundError
        If ``path`` doesn't exist.
    ValueError
        If neither a ``[reasoners]`` nor ``[actioners]`` section was found
        at all (most likely the file wasn't edited yet, or is malformed).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model roster not found at {path}. Expected a models.txt with "
            "[reasoners] and [actioners] sections in the project root."
        )

    roster: dict[str, list[str]] = {"reasoners": [], "actioners": []}
    current_section: str | None = None

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().lower()
            current_section = section if section in roster else None
            continue
        if current_section is not None:
            roster[current_section].append(line)

    if not roster["reasoners"] and not roster["actioners"]:
        raise ValueError(
            f"No [reasoners] or [actioners] entries found in {path}. "
            "Check the file has section headers like '[reasoners]' followed "
            "by one model id per line."
        )

    return roster


if __name__ == "__main__":
    import json
    import sys

    roster_path = sys.argv[1] if len(sys.argv) > 1 else "models.txt"
    roster = load_model_roster(roster_path)
    print(json.dumps(roster, indent=2))
    print(f"\n{len(roster['reasoners'])} reasoner(s) x {len(roster['actioners'])} actioner(s) "
          f"= {len(roster['reasoners']) * len(roster['actioners'])} combinations")
