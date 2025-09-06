"""
Utility functions for Project PokerMind.

This module contains common utility functions used throughout the agent,
including card vectorization, pot odds calculation, and other helpers.
"""

import numpy as np
from typing import List, Tuple, Union
import logging

logger = logging.getLogger(__name__)

# Card representation constants
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['h', 'd', 'c', 's']  # hearts, diamonds, clubs, spades

# Mapping for quick lookups
RANK_TO_INDEX = {rank: i for i, rank in enumerate(RANKS)}
SUIT_TO_INDEX = {suit: i for i, suit in enumerate(SUITS)}


def vectorize_cards(hole_cards: List[str], community_cards: List[str] = None) -> np.ndarray:
    """
    Convert hole cards and community cards into a 104-element binary vector.
    
    The vector structure is:
    - Elements 0-51: Hole card representation (52 cards, 1 if card is in hole)
    - Elements 52-103: Community card representation (52 cards, 1 if card is in community)
    
    This allows the model to distinguish between hole cards and community cards.
    
    Args:
        hole_cards: List of hole card strings (e.g., ['Ah', 'Kd'])
        community_cards: List of community card strings (e.g., ['Ts', '9c', '8h'])
    
    Returns:
        np.ndarray: 104-element binary vector representing the cards
        
    Example:
        >>> vectorize_cards(['Ah', 'Kd'], ['Ts', '9c', '8h'])
        array([0, 0, 0, ..., 1, 0, 0])  # 104 elements
    """
    if community_cards is None:
        community_cards = []
    
    # Initialize 104-element vector (52 for hole + 52 for community)
    vector = np.zeros(104, dtype=np.float32)
    
    # Process hole cards (first 52 elements)
    for card in hole_cards:
        if len(card) >= 2:
            rank = card[0]
            suit = card[1].lower()
            
            if rank in RANK_TO_INDEX and suit in SUIT_TO_INDEX:
                card_index = RANK_TO_INDEX[rank] * 4 + SUIT_TO_INDEX[suit]
                vector[card_index] = 1.0
            else:
                logger.warning(f"Invalid card format in hole cards: {card}")
    
    # Process community cards (elements 52-103)
    for card in community_cards:
        if len(card) >= 2:
            rank = card[0]
            suit = card[1].lower()
            
            if rank in RANK_TO_INDEX and suit in SUIT_TO_INDEX:
                card_index = RANK_TO_INDEX[rank] * 4 + SUIT_TO_INDEX[suit]
                vector[52 + card_index] = 1.0  # Offset by 52 for community section
            else:
                logger.warning(f"Invalid card format in community cards: {card}")
    
    return vector


def calculate_pot_odds(pot_size: int, call_amount: int) -> float:
    """
    Calculate pot odds as the required equity percentage to make calling profitable.
    
    Pot odds = call_amount / (pot_size + call_amount)
    This gives us the minimum equity needed for a profitable call.
    
    Args:
        pot_size: Current size of the pot
        call_amount: Amount needed to call the bet
    
    Returns:
        float: Required equity as a decimal (e.g., 0.25 = 25%)
        
    Example:
        >>> calculate_pot_odds(100, 50)  # Call $50 into $100 pot
        0.3333333333333333  # Need 33.33% equity to call profitably
    """
    if call_amount <= 0:
        return 0.0  # No cost to call = any equity is profitable
    
    total_pot_after_call = pot_size + call_amount
    
    if total_pot_after_call <= 0:
        return 1.0  # Edge case: need 100% equity if no pot
    
    required_equity = call_amount / total_pot_after_call
    return required_equity


def parse_card(card_str: str) -> Tuple[str, str]:
    """
    Parse a card string into rank and suit components.
    
    Args:
        card_str: Card string like 'Ah', 'Kd', 'Ts', etc.
    
    Returns:
        Tuple[str, str]: (rank, suit) or ('', '') if invalid
        
    Example:
        >>> parse_card('Ah')
        ('A', 'h')
        >>> parse_card('Kd')
        ('K', 'd')
    """
    if len(card_str) >= 2:
        return card_str[0], card_str[1].lower()
    return '', ''


def cards_to_readable_string(cards: List[str]) -> str:
    """
    Convert a list of cards to a human-readable string.
    
    Args:
        cards: List of card strings
    
    Returns:
        str: Readable representation of cards
        
    Example:
        >>> cards_to_readable_string(['Ah', 'Kd', 'Qs'])
        'A♥ K♦ Q♠'
    """
    suit_symbols = {'h': '♥', 'd': '♦', 'c': '♣', 's': '♠'}
    
    readable_cards = []
    for card in cards:
        if len(card) >= 2:
            rank = card[0]
            suit = card[1].lower()
            if suit in suit_symbols:
                readable_cards.append(f"{rank}{suit_symbols[suit]}")
            else:
                readable_cards.append(card)
        else:
            readable_cards.append(card)
    
    return ' '.join(readable_cards)


def validate_card_format(card: str) -> bool:
    """
    Validate that a card string has the correct format.
    
    Args:
        card: Card string to validate
    
    Returns:
        bool: True if card format is valid
        
    Example:
        >>> validate_card_format('Ah')
        True
        >>> validate_card_format('Zx')
        False
    """
    if len(card) < 2:
        return False
    
    rank = card[0]
    suit = card[1].lower()
    
    return rank in RANKS and suit in SUITS


def get_card_strength_value(card: str) -> int:
    """
    Get the numerical strength value of a card (for simple comparisons).
    
    Args:
        card: Card string like 'Ah', 'Kd', etc.
    
    Returns:
        int: Numerical value (2=2, 3=3, ..., T=10, J=11, Q=12, K=13, A=14)
        
    Example:
        >>> get_card_strength_value('Ah')
        14
        >>> get_card_strength_value('2c')
        2
    """
    if len(card) < 1:
        return 0
    
    rank = card[0]
    if rank in RANK_TO_INDEX:
        return RANK_TO_INDEX[rank] + 2  # Convert 0-12 index to 2-14 value
    return 0


def estimate_preflop_hand_strength(hole_cards: List[str]) -> float:
    """
    Estimate the strength of a preflop hand on a scale of 0.0 to 1.0.
    
    This is a simple heuristic for hand strength estimation before the flop.
    
    Args:
        hole_cards: List of two hole cards
    
    Returns:
        float: Hand strength estimate (0.0 = weakest, 1.0 = strongest)
        
    Example:
        >>> estimate_preflop_hand_strength(['Ah', 'Ad'])  # Pocket Aces
        0.95
        >>> estimate_preflop_hand_strength(['2h', '7c'])  # Weak hand
        0.1
    """
    if len(hole_cards) != 2:
        return 0.1  # Invalid hand
    
    card1_value = get_card_strength_value(hole_cards[0])
    card2_value = get_card_strength_value(hole_cards[1])
    
    if card1_value == 0 or card2_value == 0:
        return 0.1  # Invalid cards
    
    # Basic hand evaluation
    is_pair = card1_value == card2_value
    is_suited = hole_cards[0][1].lower() == hole_cards[1][1].lower()
    
    high_card = max(card1_value, card2_value)
    low_card = min(card1_value, card2_value)
    gap = high_card - low_card
    
    # Base strength
    if is_pair:
        # Pocket pairs - stronger for higher pairs
        base_strength = 0.5 + (high_card - 2) * 0.035  # AA=~0.92, 22=~0.5
    else:
        # High card combinations
        base_strength = 0.1 + (high_card - 2) * 0.02 + (low_card - 2) * 0.01
        
        # Suited bonus
        if is_suited:
            base_strength += 0.05
        
        # Connected cards bonus
        if gap == 1:  # Connected (KQ, 98, etc.)
            base_strength += 0.03
        elif gap == 2:  # One gap (KJ, 97, etc.)
            base_strength += 0.01
        
        # Premium combinations bonus
        if high_card >= 14 and low_card >= 13:  # AK
            base_strength += 0.2
        elif high_card >= 13 and low_card >= 11:  # AQ, KQ, etc.
            base_strength += 0.12
    
    # Ensure bounds
    return min(max(base_strength, 0.0), 1.0)


# Constants for testing and validation
EXAMPLE_HOLE_CARDS = ['Ah', 'Kd']
EXAMPLE_COMMUNITY_CARDS = ['Ts', '9c', '8h', '7s', '6d']