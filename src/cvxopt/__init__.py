"""Minimal stand-in for the :mod:`cvxopt` package used in tests.

This stub provides the :func:`matrix` helper and a :mod:`solvers` namespace
with a ``qp`` function so the library can be imported without the real
`cvxopt` dependency being installed.  The real optimisation routine is
monkeypatched in the unit tests.
"""

from __future__ import annotations

from typing import Any
import numpy as np


def matrix(data: Any) -> np.ndarray:
    """Return ``data`` as a NumPy ``ndarray``.

    The real ``cvxopt.matrix`` returns a specialised matrix type.  For the
    purposes of the tests a regular NumPy array is sufficient and keeps the
    dependency optional.
    """
    return np.array(data, dtype=float)


class _SolversModule:
    """Container mimicking :mod:`cvxopt.solvers`."""

    def __init__(self) -> None:
        # The options dictionary is accessed by the production code to disable
        # progress output.  It is left writable for test monkeypatching.
        self.options: dict[str, Any] = {}

    def qp(
        self, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:  # pragma: no cover - default
        """Placeholder quadratic-programming solver.

        The test-suite replaces this method with a fake implementation.  Raising
        ``NotImplementedError`` here helps surface accidental uses without the
        monkeypatch.
        """
        raise NotImplementedError("cvxopt solver not available")


# Expose a single instance similar to the real package structure
solvers = _SolversModule()

__all__ = ["matrix", "solvers"]
