"""
Professional-grade strategic toolkit for Project PokerMind.

This package contains state-of-the-art poker analysis tools designed
for multi-player Texas Hold'em environments.
"""

# Import all toolkit modules for easy access
try:
    from .equity_calculator import EquityCalculator
    from .range_modeler import RangeModeler
    from .board_analyzer import BoardAnalyzer
    from .post_game_analyzer import PostGameAnalyzer

    # Import specific functions from gto_tools
    from .gto_tools import (
        count_combos,
        calculate_mdf,
        calculate_pot_equity_needed,
        analyze_blockers,
        calculate_spr,
        get_spr_strategy_recommendation,
        calculate_continuation_bet_frequency,
        calculate_optimal_bet_size,
        calculate_bluff_to_value_ratio,
    )

    # Import utility functions from helpers
    from .helpers import (
        vectorize_cards,
        calculate_pot_odds,
        estimate_preflop_hand_strength,
        parse_card,
        validate_card_format,
    )

except ImportError as e:
    # Fallback for missing dependencies
    print(f"Warning: Some toolkit modules could not be imported: {e}")

__version__ = "1.0.0"
__all__ = [
    "EquityCalculator",
    "RangeModeler",
    "BoardAnalyzer",
    "PostGameAnalyzer",
    "count_combos",
    "calculate_mdf",
    "calculate_pot_equity_needed",
    "analyze_blockers",
    "calculate_spr",
    "vectorize_cards",
    "calculate_pot_odds",
    "estimate_preflop_hand_strength",
]
