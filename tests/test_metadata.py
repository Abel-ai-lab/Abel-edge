"""Package metadata consistency tests."""

from pathlib import Path
import tomllib

import abel_edge


ROOT = Path(__file__).resolve().parent.parent


def test_runtime_version_matches_project_metadata() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert abel_edge.__version__ == data["project"]["version"]
