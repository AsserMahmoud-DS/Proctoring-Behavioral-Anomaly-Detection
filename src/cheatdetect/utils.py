from pathlib import Path 

def _find_project_root() -> Path:
    """Walk upward from cwd until we find pyproject.toml."""
    start = Path.cwd().resolve()
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root (pyproject.toml not found)")