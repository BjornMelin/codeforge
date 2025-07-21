# coding=utf-8
"""Package initializer for CodeForge AI.

This module sets up the package with version info and key imports.
"""

from typing import Final

__version__: Final[str] = "0.1.0"

from .config import Settings
from .debate import debate_subgraph
from .main import run_autonomy_workflow
from .router import route_model
from .state import State
from .tools import graphrag_plus

__all__ = [
    "Settings",
    "debate_subgraph",
    "run_autonomy_workflow",
    "route_model",
    "State",
    "graphrag_plus",
]
