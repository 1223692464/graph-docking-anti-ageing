"""Resolve portable copies against the repository, not the original home path."""
import os
from pathlib import Path

ROOT = Path(os.environ.get('GRAPH_DOCKING_PROJECT_ROOT', Path(__file__).resolve().parents[1])).resolve()


def project_path(relative):
    return str(ROOT / relative)
