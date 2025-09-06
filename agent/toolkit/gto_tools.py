"""
Professional GTO Mathematics Tools for Project PokerMind.

This module implements core GTO mathematics functions used in professional
poker solvers and analysis tools.

Based on established game theory optimal principles and solver methodology.
"""

import logging
import math
from typing import Dict, Any, List, Optional, Tuple
from collections import Counter


def count_combos(hand_range: str) -> int:
    """
    Count the number of combinations in a hand range.
    
    Args:
        hand_range: Standard range notation (e.g., "AA,KK,AKs")
        
    Returns:
        Total number of combinations
    """
    # Import here to avoid circular imports
    from .range_modeler import RangeModeler
    
    range_modeler = RangeModeler()
    parsed_hands = range_modeler.parse_range_notation(hand_range)
    
    return len(parsed_hands)


def calculate_mdf(bet_size: float, pot_size: float) -> float:
    """
    Calculate Minimum Defense Frequency against a bet.
    
    MDF represents the minimum frequency a player must continue
    (call/raise) to prevent opponent from profitably bluffing.
    
    Args:
        bet_size: Size of the bet
        pot_size: Size of pot before the bet
        
    Returns:
        MDF as a decimal (0.0 to 1.0)
    """
    if bet_size <= 0 or pot_size <= 0:
        return 0.0
        
    total_pot_after_bet = pot_size + bet_size
    mdf = pot_size / total_pot_after_bet
    
    return min(1.0, max(0.0, mdf))


def calculate_pot_equity_needed(
    pot_size: float, 
    bet_to_call: float, 
    implied_odds_multiplier: float = 1.0
) -> float:
    """
    Calculate the minimum equity needed to make a profitable call.
    
    Args:
        pot_size: Current pot size
        bet_to_call: Amount we need to call
        implied_odds_multiplier: Factor for implied odds (>1.0 for positive implied odds)
        
    Returns:
        Minimum equity needed as decimal (0.0 to 1.0)
    """
    if bet_to_call <= 0:
        return 0.0
        
    if pot_size < 0:
        pot_size = 0
        
    # Basic pot odds calculation
    pot_odds_equity = bet_to_call / (pot_size + bet_to_call)
    
    # Adjust for implied odds
    adjusted_equity = pot_odds_equity / implied_odds_multiplier
    
    return min(0.99, max(0.01, adjusted_equity))


def analyze_blockers(
    our_cards: List[str], 
    board_cards: List[str],
    target_hands: List[str]
) -> Dict[str, Any]:
    """
    Analyze blocking effects of our cards on opponent ranges.
    
    Blockers reduce the number of combinations opponent can have
    of specific strong hands, affecting bluffing and value betting decisions.
    
    Args:
        our_cards: Our hole cards
        board_cards: Community cards
        target_hands: List of hand types to analyze blocking effects on
        
    Returns:
        Dict with blocking analysis
    """
    if not our_cards:
        return {"blocking_effect": 0.0, "analysis": "No cards to analyze"}
    
    # Get all cards that are "blocked" (unavailable to opponent)
    blocked_cards = set(our_cards + board_cards)
    
    blocking_effects = {}
    
    for target_hand in target_hands:
        effect = _calculate_single_hand_blocking(blocked_cards, target_hand)
        blocking_effects[target_hand] = effect
        
    # Calculate overall blocking strength
    overall_effect = sum(blocking_effects.values()) / len(blocking_effects) if blocking_effects else 0.0
    
    return {
        "blocking_effect": overall_effect,
        "individual_effects": blocking_effects,
        "blocked_cards": list(blocked_cards),
        "recommendation": _get_blocking_recommendation(overall_effect)
    }


def _calculate_single_hand_blocking(blocked_cards: set, target_hand: str) -> float:
    """Calculate blocking effect on a single target hand type."""
    # Simplified blocking calculation
    # In a full implementation, this would analyze actual combinations
    
    # Convert target hand to representative cards for blocking analysis
    target_cards = _hand_type_to_representative_cards(target_hand)
    
    # Count how many target cards we block
    blocked_target_cards = len([card for card in target_cards if any(card.startswith(c[0]) for c in blocked_cards)])
    
    # Calculate blocking percentage
    if not target_cards:
        return 0.0
        
    blocking_percentage = blocked_target_cards / len(target_cards)
    
    return blocking_percentage


def _hand_type_to_representative_cards(hand_type: str) -> List[str]:
    """Convert hand type string to representative cards for blocking analysis."""
    hand_type = hand_type.lower()
    
    if "flush" in hand_type:
        return ["A", "K", "Q", "J", "T"]  # High cards that make strong flushes
    elif "straight" in hand_type:
        return ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5"]
    elif "full" in hand_type or "set" in hand_type or "trips" in hand_type:
        return ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
    elif "pair" in hand_type:
        return ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
    else:
        return ["A", "K", "Q", "J", "T"]  # Default to high cards


def _get_blocking_recommendation(blocking_effect: float) -> str:
    """Get strategic recommendation based on blocking effect."""
    if blocking_effect > 0.3:
        return "Strong blocking - good bluff candidate"
    elif blocking_effect > 0.15:
        return "Moderate blocking - consider bluffing"
    else:
        return "Weak blocking - avoid bluffing"


def calculate_spr(effective_stack: float, pot_size: float) -> float:
    """
    Calculate Stack-to-Pot Ratio (SPR).
    
    SPR is a crucial concept for determining optimal post-flop strategy.
    Lower SPR favors stronger hands, higher SPR allows more complex play.
    
    Args:
        effective_stack: Effective stack size (smaller of the two stacks)
        pot_size: Current pot size
        
    Returns:
        SPR value
    """
    if pot_size <= 0:
        return float('inf')
        
    if effective_stack <= 0:
        return 0.0
        
    return effective_stack / pot_size


def get_spr_strategy_recommendation(spr: float, hand_strength: str) -> str:
    """
    Get strategy recommendation based on SPR and hand strength.
    
    Args:
        spr: Stack-to-Pot Ratio
        hand_strength: Categorical hand strength ("strong", "medium", "weak")
        
    Returns:
        Strategic recommendation string
    """
    hand_strength = hand_strength.lower()
    
    if spr <= 2:  # Low SPR
        if hand_strength == "strong":
            return "Get all-in - low SPR favors strong hands"
        elif hand_strength == "medium":
            return "Play cautiously - consider pot control"
        else:
            return "Fold to significant action - low SPR unfavorable for weak hands"
            
    elif spr <= 6:  # Medium SPR
        if hand_strength == "strong":
            return "Build pot aggressively - good spot for value"
        elif hand_strength == "medium":
            return "Play fit-or-fold - medium SPR allows selective aggression"
        else:
            return "Play tight - look for good bluffing spots"
            
    else:  # High SPR
        if hand_strength == "strong":
            return "Build pot carefully - deep stacks allow complex play"
        elif hand_strength == "medium":
            return "Play speculative - high SPR rewards drawing hands"
        else:
            return "Play tight or bluff selectively - high SPR rewards skill"


def calculate_continuation_bet_frequency(
    position: str,
    board_texture: str, 
    opponent_type: str = "balanced"
) -> float:
    """
    Calculate optimal continuation bet frequency based on GTO principles.
    
    Args:
        position: Our position ("IP" for in position, "OOP" for out of position)
        board_texture: Board texture ("dry", "wet", "coordinated")
        opponent_type: Opponent type ("tight", "loose", "balanced")
        
    Returns:
        Recommended c-bet frequency (0.0 to 1.0)
    """
    # Base frequency adjustments
    base_frequency = 0.65  # Standard GTO c-bet frequency
    
    # Position adjustment
    if position.upper() == "IP":
        position_adjustment = 0.05  # Slightly higher IP
    else:
        position_adjustment = -0.05  # Slightly lower OOP
        
    # Board texture adjustment
    texture_adjustments = {
        "dry": 0.10,        # C-bet more on dry boards
        "wet": -0.15,       # C-bet less on wet boards  
        "coordinated": -0.20  # C-bet much less on coordinated boards
    }
    
    texture_adjustment = texture_adjustments.get(board_texture.lower(), 0.0)
    
    # Opponent type adjustment
    opponent_adjustments = {
        "tight": 0.08,      # C-bet more vs tight opponents
        "loose": -0.08,     # C-bet less vs loose opponents
        "balanced": 0.0     # No adjustment vs balanced opponents
    }
    
    opponent_adjustment = opponent_adjustments.get(opponent_type.lower(), 0.0)
    
    # Calculate final frequency
    final_frequency = base_frequency + position_adjustment + texture_adjustment + opponent_adjustment
    
    # Clamp to reasonable bounds
    return max(0.15, min(0.90, final_frequency))


def calculate_optimal_bet_size(
    pot_size: float,
    hand_strength: str,
    board_texture: str,
    opponent_tendencies: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Calculate GTO-based optimal bet sizing.
    
    Args:
        pot_size: Current pot size
        hand_strength: Our hand strength category
        board_texture: Board texture description
        opponent_tendencies: Dict with opponent fold frequencies, etc.
        
    Returns:
        Dict with bet size recommendation and reasoning
    """
    if opponent_tendencies is None:
        opponent_tendencies = {}
        
    # Base bet sizes as percentage of pot
    base_sizes = {
        "value": 0.75,      # 75% pot for value
        "bluff": 0.65,      # 65% pot for bluffs
        "protection": 0.50,  # 50% pot for protection
        "thin_value": 0.40   # 40% pot for thin value
    }
    
    # Determine bet category based on hand strength
    if hand_strength.lower() in ["strong", "very_strong"]:
        bet_category = "value"
    elif hand_strength.lower() == "weak":
        bet_category = "bluff" 
    elif hand_strength.lower() == "medium":
        bet_category = "protection"
    else:
        bet_category = "thin_value"
        
    base_size_percentage = base_sizes[bet_category]
    
    # Adjust for board texture
    if board_texture.lower() == "wet":
        size_adjustment = 0.15  # Bet larger on wet boards
    elif board_texture.lower() == "dry":
        size_adjustment = -0.10  # Can bet smaller on dry boards
    else:
        size_adjustment = 0.0
        
    # Adjust for opponent tendencies
    opponent_fold_freq = opponent_tendencies.get("fold_to_cbet", 0.4)
    
    if opponent_fold_freq > 0.6:  # High fold frequency
        size_adjustment += 0.10  # Can bet larger
    elif opponent_fold_freq < 0.3:  # Low fold frequency
        size_adjustment -= 0.05  # Should bet smaller
        
    final_size_percentage = base_size_percentage + size_adjustment
    final_size_percentage = max(0.25, min(1.5, final_size_percentage))  # Reasonable bounds
    
    recommended_bet = pot_size * final_size_percentage
    
    return {
        "bet_size": recommended_bet,
        "pot_percentage": final_size_percentage,
        "category": bet_category,
        "reasoning": f"{bet_category} bet on {board_texture} board vs opponent with {opponent_fold_freq:.1%} fold frequency"
    }


def calculate_bluff_to_value_ratio(pot_size: float, bet_size: float) -> Dict[str, Any]:
    """
    Calculate optimal bluff-to-value ratio based on GTO principles.
    
    The bluff-to-value ratio ensures opponent is indifferent to calling,
    making our strategy unexploitable.
    
    Args:
        pot_size: Current pot size
        bet_size: Size of our bet
        
    Returns:
        Dict with ratio analysis and recommendations
    """
    if bet_size <= 0 or pot_size < 0:
        return {"error": "Invalid pot or bet size"}
        
    # Calculate pot odds opponent is getting
    pot_odds = bet_size / (pot_size + bet_size)
    
    # Calculate required bluff-to-value ratio
    # Formula: Bluffs / (Bluffs + Value) = Pot Odds
    # Solving for Bluffs/Value ratio: (Pot Odds) / (1 - Pot Odds)
    
    if pot_odds >= 0.99:  # Avoid division by near-zero
        bluff_to_value_ratio = 99.0
    else:
        bluff_to_value_ratio = pot_odds / (1 - pot_odds)
    
    return {
        "bluff_to_value_ratio": bluff_to_value_ratio,
        "pot_odds_offered": pot_odds,
        "interpretation": f"For every value bet, include {bluff_to_value_ratio:.2f} bluffs",
        "example": f"If betting 10 value hands, include {int(bluff_to_value_ratio * 10)} bluff hands"
    }


def calculate_implied_odds(
    pot_size: float,
    bet_to_call: float,
    our_stack: float,
    opponent_stack: float,
    win_probability: float,
    implied_bet_probability: float = 0.7
) -> Dict[str, Any]:
    """
    Calculate implied odds considering future betting rounds.
    
    Implied odds account for additional chips we can win if we hit our draw,
    making draws profitable even when pot odds alone are insufficient.
    
    This is a key enhancement for Pillar 4 that enables more sophisticated
    draw evaluation in the Grandmaster's decision logic.
    
    Args:
        pot_size: Current pot size
        bet_to_call: Amount we need to call
        our_stack: Our remaining chips
        opponent_stack: Opponent's remaining chips  
        win_probability: Probability we win if we hit our draw (0.0-1.0)
        implied_bet_probability: Probability opponent will pay us off if we hit
        
    Returns:
        Dict with implied odds analysis and recommendation
    """
    if bet_to_call <= 0 or pot_size < 0 or win_probability <= 0:
        return {"error": "Invalid inputs for implied odds calculation"}
    
    # Calculate direct pot odds
    total_pot_after_call = pot_size + bet_to_call * 2
    direct_pot_odds = bet_to_call / total_pot_after_call
    
    # Calculate additional chips we can win (implied odds)
    max_additional_win = min(our_stack - bet_to_call, opponent_stack)
    expected_additional_win = max_additional_win * implied_bet_probability
    
    # Calculate total expected winnings if we hit
    total_expected_win = total_pot_after_call + expected_additional_win
    
    # Calculate implied odds
    implied_odds = bet_to_call / total_expected_win if total_expected_win > 0 else 1.0
    
    # Calculate break-even probability needed
    break_even_probability = implied_odds
    
    # Determine if call is profitable
    is_profitable = win_probability > break_even_probability
    
    # Calculate expected value
    expected_value = (win_probability * total_expected_win) - bet_to_call
    
    # Risk assessment
    stack_risk_ratio = bet_to_call / (our_stack + bet_to_call)
    
    return {
        "implied_odds": implied_odds,
        "direct_pot_odds": direct_pot_odds,
        "break_even_probability": break_even_probability,
        "win_probability": win_probability,
        "expected_value": expected_value,
        "is_profitable": is_profitable,
        "profitability_margin": win_probability - break_even_probability,
        "stack_risk_ratio": stack_risk_ratio,
        "max_additional_win": max_additional_win,
        "expected_additional_win": expected_additional_win,
        "recommendation": _get_implied_odds_recommendation(
            is_profitable, expected_value, stack_risk_ratio, win_probability
        ),
        "analysis_type": "implied_odds_advanced"
    }


def _get_implied_odds_recommendation(
    is_profitable: bool,
    expected_value: float, 
    stack_risk_ratio: float,
    win_probability: float
) -> str:
    """Generate strategic recommendation based on implied odds analysis."""
    if not is_profitable:
        if expected_value < -10:
            return "Clear fold - negative implied odds with high cost"
        else:
            return "Fold - implied odds insufficient despite future betting potential"
    
    if stack_risk_ratio > 0.5:
        return "Caution - profitable but high stack risk, consider fold in tournaments"
    elif expected_value > 20:
        return "Strong call - excellent implied odds with high expected value"
    elif win_probability > 0.3:
        return "Good call - solid implied odds with reasonable win probability"
    else:
        return "Marginal call - profitable but low win probability"


def calculate_reverse_implied_odds(
    pot_size: float,
    bet_to_call: float,
    our_hand_strength: float,
    board_danger: float,
    position: str = "unknown"
) -> Dict[str, Any]:
    """
    Calculate reverse implied odds - money we might lose when we hit but still lose.
    
    Reverse implied odds occur when we make our hand but opponent has an even stronger hand,
    causing us to lose additional chips we might have saved by folding.
    
    Args:
        pot_size: Current pot size
        bet_to_call: Amount to call
        our_hand_strength: Our hand strength if we hit (0.0-1.0)
        board_danger: Board danger level (0.0-1.0)
        position: Our position for strategic adjustment
        
    Returns:
        Dict with reverse implied odds analysis
    """
    # Calculate base reverse implied odds
    reverse_implied_factor = board_danger * (1.0 - our_hand_strength)
    
    # Position adjustment (worse position = higher reverse implied odds)
    position_multiplier = 1.2 if position == "out_of_position" else 1.0
    reverse_implied_factor *= position_multiplier
    
    # Estimate additional losses when we hit but lose
    potential_additional_loss = pot_size * reverse_implied_factor
    
    # Adjust expected value for reverse implied odds
    reverse_implied_cost = potential_additional_loss * 0.3  # Probability of hitting but losing
    
    return {
        "reverse_implied_factor": reverse_implied_factor,
        "potential_additional_loss": potential_additional_loss, 
        "reverse_implied_cost": reverse_implied_cost,
        "board_danger_adjustment": board_danger,
        "position_multiplier": position_multiplier,
        "recommendation": "Consider folding strong draws on dangerous boards" if reverse_implied_factor > 0.4 else "Reverse implied odds manageable"
    }