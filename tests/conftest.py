"""Test configuration for ensuring project modules are importable.

Pytest imports this file before collecting tests.  We add the ``src``
directory to ``sys.path`` here so that the lightweight :mod:`cvxopt` stub
bundled with the project is discovered before tests import it.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add the ``src`` directory (one level above this file) to ``sys.path`` so
# tests can import project-local modules without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
