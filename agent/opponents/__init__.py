"""
Agent opponents package for tournament simulation.

This package contains different opponent archetypes for testing the PokerMind agent
in tournament scenarios.
"""

# Import all opponent types for easy access
from .The_TAG import TightAggressivePlayer
from .The_LAG import LooseAggressivePlayer
from .The_Nit import NitPlayer

__all__ = ["TightAggressivePlayer", "LooseAggressivePlayer", "NitPlayer"]
