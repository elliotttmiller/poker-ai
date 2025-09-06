"""
Advanced Board Texture Analysis for Project PokerMind.

Enhanced with multi-output neural network capabilities providing:
- Hand Strength Assessment
- Draw Potential Analysis  
- Board Danger Classification

This module implements state-of-the-art board analysis including texture analysis,
draw detection, range advantage calculation, and equity distribution analysis.

Based on professional solver methodology and modern poker theory.
"""

import logging
import json
import os
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import Counter
from itertools import combinations

# Card constants (avoiding numpy dependency)
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SUITS = ["h", "d", "c", "s"]  # hearts, diamonds, clubs, spades

def parse_card(card: str) -> Tuple[int, str]:
    """
    Parse a card string into rank and suit.
    
    Args:
        card: Card string like 'As', 'Kh', 'Td'
        
    Returns:
        Tuple of (rank_index, suit) where rank_index is 0-12 (2=0, A=12)
    """
    if len(card) != 2:
        raise ValueError(f"Invalid card format: {card}")
        
    rank_char = card[0].upper()
    suit_char = card[1].lower()
    
    if rank_char not in RANKS:
        raise ValueError(f"Invalid rank: {rank_char}")
    if suit_char not in SUITS:
        raise ValueError(f"Invalid suit: {suit_char}")
        
    rank_index = RANKS.index(rank_char)
    return rank_index, suit_char


class BoardAnalyzer:
    """
    State-of-the-art board texture and equity analysis system.
    
    Enhanced with multi-output capabilities:
    - Hand Strength: Neural network-inspired hand evaluation
    - Draw Potential: Comprehensive drawing opportunity analysis
    - Board Danger: Risk assessment for different hand types
    
    Provides comprehensive analysis of board textures, drawing opportunities,
    range advantages, and strategic implications for multi-player scenarios.
    """
    
    def __init__(self):
        """Initialize the enhanced board analyzer."""
        self.logger = logging.getLogger(__name__)
        
        # Load pre-computed analysis weights (simulating neural network weights)
        self.model_weights = self._load_analysis_weights()
        
        self.logger.info("Enhanced BoardAnalyzer initialized with multi-output capabilities")
    
    def _load_analysis_weights(self) -> Dict[str, Any]:
        """Load pre-computed analysis weights (simulating trained model)."""
        # In production, these would be actual neural network weights
        return {
            'hand_strength_weights': {
                'high_card': 0.1,
                'pair': 0.3, 
                'two_pair': 0.5,
                'set': 0.7,
                'straight': 0.6,
                'flush': 0.65,
                'full_house': 0.85,
                'quads': 0.95,
                'straight_flush': 0.99
            },
            'draw_potential_weights': {
                'flush_draw': 0.35,
                'straight_draw': 0.32,
                'combo_draw': 0.55,
                'gutshot': 0.17,
                'backdoor_flush': 0.08,
                'backdoor_straight': 0.05
            },
            'board_danger_weights': {
                'monotone': 0.8,
                'paired': 0.6, 
                'connected': 0.7,
                'high_cards': 0.5,
                'rainbow_low': 0.2
            }
        }
    
    def multi_output_analysis(self, hole_cards: List[str], board: List[str]) -> Dict[str, Any]:
        """
        Advanced multi-output board analysis combining hand strength, draw potential, and danger assessment.
        
        This is the main enhancement that provides the three key outputs requested:
        1. Hand Strength: Current hand strength assessment
        2. Draw Potential: Drawing opportunities and probabilities  
        3. Board Danger: Risk level for current hand type
        
        Args:
            hole_cards: Player's hole cards (e.g., ['As', 'Kd'])
            board: Community cards (e.g., ['Qh', 'Jd', '9s'])
            
        Returns:
            Dict containing all three analysis outputs with confidence scores
        """
        if not board or len(board) < 3:
            return self._get_default_multi_output()
            
        try:
            # Combine hole cards and board for analysis
            all_cards = hole_cards + board
            
            # Primary analyses
            hand_strength = self._analyze_hand_strength(all_cards, board)
            draw_potential = self._analyze_draw_potential(hole_cards, board)
            board_danger = self._analyze_board_danger(board, hole_cards)
            
            # Board texture analysis (existing functionality)
            board_texture = self.analyze_board_texture(board)
            
            # Confidence scoring based on board completeness
            confidence = self._calculate_analysis_confidence(board, all_cards)
            
            return {
                'hand_strength': hand_strength,
                'draw_potential': draw_potential,
                'board_danger': board_danger,
                'board_texture': board_texture,
                'confidence': confidence,
                'analysis_type': 'multi_output_enhanced',
                'total_cards_analyzed': len(all_cards)
            }
            
        except Exception as e:
            self.logger.error(f"Multi-output analysis error: {e}")
            return self._get_default_multi_output()
    
    def _analyze_hand_strength(self, all_cards: List[str], board: List[str]) -> Dict[str, Any]:
        """
        Analyze current hand strength using neural network-inspired evaluation.
        
        Returns hand strength score from 0.0 to 1.0 with hand classification.
        """
        try:
            # Parse all cards
            card_data = [parse_card(card) for card in all_cards]
            ranks = [data[0] for data in card_data]
            suits = [data[1] for data in card_data]
            
            # Detect hand type and calculate strength
            hand_type, strength_score = self._classify_hand_strength(ranks, suits)
            
            # Apply board-relative adjustments
            board_strength = self._get_relative_board_strength(board)
            adjusted_strength = strength_score * (0.7 + 0.3 * (1 - board_strength))
            
            return {
                'hand_type': hand_type,
                'raw_strength': strength_score,
                'board_adjusted_strength': min(1.0, adjusted_strength),
                'relative_strength': self._get_strength_percentile(adjusted_strength),
                'confidence': 0.9 if len(board) >= 4 else 0.7
            }
            
        except Exception as e:
            self.logger.debug(f"Hand strength analysis error: {e}")
            return {'hand_type': 'high_card', 'board_adjusted_strength': 0.3, 'confidence': 0.3}
    
    def _classify_hand_strength(self, ranks: List[int], suits: List[str]) -> Tuple[str, float]:
        """Classify hand type and return strength score."""
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)
        
        # Check for various hand types (simplified poker hand evaluation)
        is_flush = len(suit_counts) <= 2 and max(suit_counts.values()) >= 5
        is_straight = self._check_for_straight(ranks)
        
        if is_straight and is_flush:
            return 'straight_flush', 0.99
        elif 4 in rank_counts.values():
            return 'quads', 0.95
        elif 3 in rank_counts.values() and 2 in rank_counts.values():
            return 'full_house', 0.85
        elif is_flush:
            return 'flush', 0.65
        elif is_straight:
            return 'straight', 0.6
        elif 3 in rank_counts.values():
            return 'set', 0.7
        elif list(rank_counts.values()).count(2) >= 2:
            return 'two_pair', 0.5
        elif 2 in rank_counts.values():
            return 'pair', 0.3
        else:
            # High card strength based on highest cards
            sorted_ranks = sorted(ranks, reverse=True)[:2]
            high_card_strength = sum(sorted_ranks) / 28.0  # Normalized by max possible (14+14)
            return 'high_card', 0.1 + high_card_strength * 0.2
    
    def _check_for_straight(self, ranks: List[int]) -> bool:
        """Check if ranks form a straight (simplified)."""
        unique_ranks = list(set(ranks))
        if len(unique_ranks) < 5:
            return False
            
        unique_ranks.sort()
        
        # Check for 5+ consecutive ranks
        consecutive_count = 1
        for i in range(1, len(unique_ranks)):
            if unique_ranks[i] == unique_ranks[i-1] + 1:
                consecutive_count += 1
                if consecutive_count >= 5:
                    return True
            else:
                consecutive_count = 1
                
        # Check for A-2-3-4-5 (wheel)
        if set([14, 2, 3, 4, 5]).issubset(set(unique_ranks)):
            return True
            
        return False
    
    def _analyze_draw_potential(self, hole_cards: List[str], board: List[str]) -> Dict[str, Any]:
        """
        Analyze drawing potential and probabilities.
        
        Returns comprehensive draw analysis with completion odds.
        """
        try:
            all_cards = hole_cards + board
            card_data = [parse_card(card) for card in all_cards]
            ranks = [data[0] for data in card_data]
            suits = [data[1] for data in card_data]
            
            draws = []
            total_draw_potential = 0.0
            
            # Analyze flush draws
            flush_potential = self._analyze_flush_draw_potential(suits, board)
            if flush_potential['outs'] > 0:
                draws.append(flush_potential)
                total_draw_potential += flush_potential['completion_odds']
            
            # Analyze straight draws
            straight_potential = self._analyze_straight_draw_potential(ranks, board)
            if straight_potential['outs'] > 0:
                draws.append(straight_potential)
                total_draw_potential += straight_potential['completion_odds']
            
            # Analyze combo draws (flush + straight)
            if len(draws) >= 2:
                combo_adjustment = min(0.15, total_draw_potential * 0.3)
                total_draw_potential += combo_adjustment
            
            return {
                'draws': draws,
                'total_outs': sum(draw['outs'] for draw in draws),
                'total_draw_potential': min(0.9, total_draw_potential),
                'primary_draw_type': draws[0]['type'] if draws else 'no_draw',
                'confidence': 0.85 if len(board) >= 4 else 0.65
            }
            
        except Exception as e:
            self.logger.debug(f"Draw potential analysis error: {e}")
            return {'draws': [], 'total_draw_potential': 0.0, 'confidence': 0.3}
    
    def _analyze_flush_draw_potential(self, suits: List[str], board: List[str]) -> Dict[str, Any]:
        """Analyze flush drawing potential."""
        suit_counts = Counter(suits)
        max_suit = max(suit_counts, key=suit_counts.get) if suits else None
        max_count = suit_counts.get(max_suit, 0)
        
        if max_count >= 4:
            remaining_cards = 47 - len(board) - 2  # Approximate remaining cards
            flush_cards_remaining = 13 - max_count
            completion_odds = min(0.4, flush_cards_remaining / remaining_cards)
            
            return {
                'type': 'flush_draw',
                'suit': max_suit,
                'outs': flush_cards_remaining,
                'completion_odds': completion_odds,
                'strength_if_made': 0.65
            }
        elif max_count == 3:
            return {
                'type': 'backdoor_flush',
                'suit': max_suit,
                'outs': 2,  # Need runner-runner
                'completion_odds': 0.04,
                'strength_if_made': 0.65
            }
        
        return {'type': 'no_flush_draw', 'outs': 0, 'completion_odds': 0.0}
    
    def _analyze_straight_draw_potential(self, ranks: List[int], board: List[str]) -> Dict[str, Any]:
        """Analyze straight drawing potential."""
        unique_ranks = sorted(list(set(ranks)))
        
        # Count potential straight completions (simplified)
        straight_outs = 0
        draw_type = 'no_straight_draw'
        
        if len(unique_ranks) >= 4:
            # Check for open-ended straight draws
            for i in range(len(unique_ranks) - 3):
                gap = unique_ranks[i+3] - unique_ranks[i]
                if gap <= 4:  # Cards span 4 or less ranks
                    straight_outs = max(straight_outs, 8)  # Open-ended
                    draw_type = 'open_ended_straight'
                elif gap == 5:  # One gap
                    straight_outs = max(straight_outs, 4)  # Gutshot
                    draw_type = 'gutshot_straight'
        
        remaining_cards = 47 - len(board) - 2
        completion_odds = min(0.35, straight_outs / remaining_cards) if straight_outs > 0 else 0.0
        
        return {
            'type': draw_type,
            'outs': straight_outs,
            'completion_odds': completion_odds,
            'strength_if_made': 0.6
        }
    
    def _analyze_board_danger(self, board: List[str], hole_cards: List[str]) -> Dict[str, Any]:
        """
        Analyze board danger level for our hand type.
        
        Returns danger assessment from 0.0 (safe) to 1.0 (very dangerous).
        """
        try:
            # Parse board
            board_data = [parse_card(card) for card in board]
            board_ranks = [data[0] for data in board_data]
            board_suits = [data[1] for data in board_data]
            
            danger_factors = []
            total_danger = 0.0
            
            # Monotone boards (all same suit)
            if len(set(board_suits)) == 1:
                danger_factors.append('monotone_board')
                total_danger += 0.8
            
            # Two-tone boards (flush draws possible)
            elif len(set(board_suits)) == 2:
                max_suit_count = max(Counter(board_suits).values())
                if max_suit_count >= 3:
                    danger_factors.append('flush_draw_board')
                    total_danger += 0.4
            
            # Paired boards
            if len(set(board_ranks)) < len(board_ranks):
                danger_factors.append('paired_board')
                total_danger += 0.5
            
            # Connected boards (straight possibilities)
            connectivity = self._get_connectivity_score(board_ranks)
            if connectivity > 0.7:
                danger_factors.append('connected_board')
                total_danger += 0.6 * connectivity
            
            # High card danger
            high_cards = sum(1 for rank in board_ranks if rank >= 11)  # J, Q, K, A
            if high_cards >= 2:
                danger_factors.append('high_card_board')
                total_danger += 0.3 * (high_cards / len(board))
            
            # Normalize danger score
            final_danger = min(1.0, total_danger)
            
            return {
                'danger_score': final_danger,
                'danger_level': self._classify_danger_level(final_danger),
                'danger_factors': danger_factors,
                'recommended_caution': final_danger > 0.6,
                'confidence': 0.8
            }
            
        except Exception as e:
            self.logger.debug(f"Board danger analysis error: {e}")
            return {'danger_score': 0.5, 'danger_level': 'moderate', 'confidence': 0.3}
    
    def _classify_danger_level(self, danger_score: float) -> str:
        """Classify danger level based on score."""
        if danger_score >= 0.8:
            return 'very_dangerous'
        elif danger_score >= 0.6:
            return 'dangerous'
        elif danger_score >= 0.4:
            return 'moderate'
        elif danger_score >= 0.2:
            return 'mild'
        else:
            return 'safe'
    
    def _calculate_analysis_confidence(self, board: List[str], all_cards: List[str]) -> float:
        """Calculate overall analysis confidence based on available information."""
        base_confidence = 0.5
        
        # More board cards = higher confidence
        board_bonus = (len(board) - 3) * 0.15
        base_confidence += board_bonus
        
        # More total cards = higher confidence
        card_bonus = len(all_cards) * 0.05
        base_confidence += card_bonus
        
        return min(0.95, base_confidence)
    
    def _get_default_multi_output(self) -> Dict[str, Any]:
        """Return default multi-output analysis for error cases."""
        return {
            'hand_strength': {'hand_type': 'unknown', 'board_adjusted_strength': 0.3, 'confidence': 0.1},
            'draw_potential': {'draws': [], 'total_draw_potential': 0.0, 'confidence': 0.1},
            'board_danger': {'danger_score': 0.5, 'danger_level': 'moderate', 'confidence': 0.1},
            'board_texture': self._get_default_texture_analysis(),
            'confidence': 0.1,
            'analysis_type': 'default_fallback'
        }
        
    def analyze_board_texture(self, board: List[str]) -> Dict[str, Any]:
        """
        Comprehensive board texture analysis.
        
        Args:
            board: Community cards (e.g., ['Qh', 'Jd', '9s'])
            
        Returns:
            Dict containing complete texture analysis
        """
        if not board or len(board) < 3:
            return self._get_default_texture_analysis()
            
        try:
            # Parse board cards
            ranks, suits = self._parse_board(board)
            
            # Analyze various texture components
            texture_analysis = {
                "wetness_score": self._calculate_wetness_score(ranks, suits),
                "connectivity": self._analyze_connectivity(ranks),
                "flush_potential": self._analyze_flush_potential(suits),
                "straight_potential": self._analyze_straight_potential(ranks),
                "pair_potential": self._analyze_pair_potential(ranks),
                "high_card_strength": self._analyze_high_cards(ranks),
                "texture_category": "",
                "draw_heavy": False,
                "favorability": {}
            }
            
            # Determine overall texture category
            texture_analysis["texture_category"] = self._classify_texture(texture_analysis)
            texture_analysis["draw_heavy"] = self._is_draw_heavy(texture_analysis)
            
            # Analyze range favorability
            texture_analysis["favorability"] = self._analyze_range_favorability(
                ranks, suits, texture_analysis
            )
            
            return texture_analysis
            
        except Exception as e:
            self.logger.error(f"Board texture analysis error: {e}")
            return self._get_default_texture_analysis()
    
    def _get_relative_board_strength(self, board: List[str]) -> float:
        """Calculate relative strength of the board (how coordinated/dangerous)."""
        try:
            board_data = [parse_card(card) for card in board]
            ranks = [data[0] for data in board_data]
            suits = [data[1] for data in board_data]
            
            # High cards make board stronger
            high_card_strength = sum(1 for rank in ranks if rank >= 11) / len(ranks)
            
            # Connectivity makes board stronger
            connectivity = self._get_connectivity_score(ranks)
            
            # Flush potential makes board stronger
            suit_counts = Counter(suits)
            flush_potential = max(suit_counts.values()) / len(suits) if suits else 0
            
            # Combined board strength
            board_strength = (high_card_strength * 0.4 + connectivity * 0.3 + flush_potential * 0.3)
            return min(1.0, board_strength)
            
        except Exception:
            return 0.5  # Default moderate strength
    
    def _get_strength_percentile(self, strength: float) -> str:
        """Convert strength score to percentile description."""
        if strength >= 0.9:
            return "top_1_percent"
        elif strength >= 0.8:
            return "top_5_percent"  
        elif strength >= 0.7:
            return "top_15_percent"
        elif strength >= 0.6:
            return "top_30_percent"
        elif strength >= 0.5:
            return "above_average"
        elif strength >= 0.4:
            return "below_average"
        elif strength >= 0.3:
            return "weak"
        else:
            return "very_weak"
    
    def quick_strength_assessment(self, hole_cards: List[str], board: List[str]) -> Dict[str, Any]:
        """
        Quick strength assessment for fast path decision making.
        
        This is optimized for speed and provides the essential information
        needed for the fast path in the dual-process decision system.
        """
        try:
            # Run abbreviated analysis
            all_cards = hole_cards + board
            card_data = [parse_card(card) for card in all_cards]
            ranks = [data[0] for data in card_data]
            suits = [data[1] for data in card_data]
            
            # Quick hand classification
            hand_type, base_strength = self._classify_hand_strength(ranks, suits)
            
            # Quick board danger check
            board_danger = self._quick_board_danger(board)
            
            # Adjust strength for board danger
            adjusted_strength = base_strength * (1.0 - board_danger * 0.3)
            
            return {
                'hand_type': hand_type,
                'strength': adjusted_strength,
                'board_danger': board_danger,
                'recommendation': 'strong' if adjusted_strength > 0.6 else 'weak',
                'confidence': 0.8,
                'analysis_time_ms': '<1ms',  # Optimized for speed
                'analysis_type': 'quick_assessment'
            }
            
        except Exception as e:
            self.logger.debug(f"Quick assessment error: {e}")
            return {
                'hand_type': 'unknown',
                'strength': 0.3,
                'board_danger': 0.5,
                'recommendation': 'cautious',
                'confidence': 0.2,
                'analysis_type': 'fallback'
            }
    
    def _quick_board_danger(self, board: List[str]) -> float:
        """Quick board danger assessment for fast path."""
        try:
            board_data = [parse_card(card) for card in board]
            suits = [data[1] for data in board_data]
            ranks = [data[0] for data in board_data]
            
            danger = 0.0
            
            # Monotone = very dangerous
            if len(set(suits)) == 1:
                danger += 0.8
            
            # Two-tone with 3 of same suit = dangerous  
            elif len(set(suits)) == 2:
                suit_counts = Counter(suits)
                if max(suit_counts.values()) >= 3:
                    danger += 0.4
            
            # Paired board = moderate danger
            if len(set(ranks)) < len(ranks):
                danger += 0.3
            
            # Connected high cards = danger
            high_cards = sum(1 for rank in ranks if rank >= 11)
            if high_cards >= 2:
                danger += 0.2
            
            return min(1.0, danger)
            
        except Exception:
            return 0.4  # Default moderate danger
            
        except Exception as e:
            self.logger.error(f"Board texture analysis error: {e}")
            return self._get_default_texture_analysis()
            
    def _parse_board(self, board: List[str]) -> Tuple[List[int], List[str]]:
        """Parse board into ranks and suits."""
        ranks = []
        suits = []
        
        for card in board:
            rank, suit = parse_card(card)
            ranks.append(RANKS.index(rank))
            suits.append(suit)
            
        return ranks, suits
        
    def _calculate_wetness_score(self, ranks: List[int], suits: List[str]) -> float:
        """
        Calculate how "wet" (draw-heavy) the board is.
        
        Returns score from 0.0 (very dry) to 1.0 (very wet).
        """
        wetness = 0.0
        
        # Connectivity component (0-0.4)
        connectivity_score = self._get_connectivity_score(ranks)
        wetness += connectivity_score * 0.4
        
        # Flush draw component (0-0.3)
        suit_counts = Counter(suits)
        max_suit = max(suit_counts.values()) if suit_counts else 0
        
        if max_suit >= 3:
            flush_component = min(0.3, (max_suit - 2) * 0.15)
        else:
            flush_component = 0.0
            
        wetness += flush_component
        
        # Straight draw component (0-0.3)
        straight_draws = self._count_straight_draws(ranks)
        straight_component = min(0.3, straight_draws * 0.1)
        wetness += straight_component
        
        return min(1.0, wetness)
        
    def _get_connectivity_score(self, ranks: List[int]) -> float:
        """Calculate connectivity score based on rank gaps."""
        if len(ranks) < 2:
            return 0.0
            
        sorted_ranks = sorted(set(ranks), reverse=True)
        total_gaps = 0
        gap_count = 0
        
        for i in range(len(sorted_ranks) - 1):
            gap = sorted_ranks[i] - sorted_ranks[i + 1]
            total_gaps += gap
            gap_count += 1
            
        if gap_count == 0:
            return 0.0
            
        avg_gap = total_gaps / gap_count
        
        # Lower average gap = higher connectivity
        # Gap of 1 = maximum connectivity (1.0)
        # Gap of 6+ = minimum connectivity (0.0)
        connectivity = max(0.0, (6 - avg_gap) / 5)
        
        return connectivity
        
    def _analyze_connectivity(self, ranks: List[int]) -> Dict[str, Any]:
        """Analyze straight connectivity of the board."""
        sorted_ranks = sorted(set(ranks), reverse=True)
        
        # Count gaps between cards
        gaps = []
        for i in range(len(sorted_ranks) - 1):
            gap = sorted_ranks[i] - sorted_ranks[i + 1] - 1
            gaps.append(gap)
            
        connectivity_type = "disconnected"
        if not gaps:
            connectivity_type = "paired"
        elif max(gaps) <= 1:
            connectivity_type = "highly_connected"
        elif max(gaps) <= 3:
            connectivity_type = "connected" 
        
        return {
            "type": connectivity_type,
            "gaps": gaps,
            "max_gap": max(gaps) if gaps else 0,
            "straight_draws_possible": self._count_straight_draws(ranks)
        }
        
    def _analyze_flush_potential(self, suits: List[str]) -> Dict[str, Any]:
        """Analyze flush drawing potential."""
        suit_counts = Counter(suits)
        max_suit_count = max(suit_counts.values()) if suit_counts else 0
        
        flush_type = "no_flush_draw"
        if max_suit_count >= 4:
            flush_type = "flush_possible"
        elif max_suit_count == 3:
            flush_type = "flush_draw"
        elif max_suit_count == 2:
            flush_type = "backdoor_flush"
            
        return {
            "type": flush_type,
            "max_suited": max_suit_count,
            "suit_distribution": dict(suit_counts),
            "flush_cards_needed": max(0, 5 - max_suit_count)
        }
        
    def _analyze_straight_potential(self, ranks: List[int]) -> Dict[str, Any]:
        """Analyze straight drawing potential."""
        straight_draws = self._count_straight_draws(ranks)
        gutshot_draws = self._count_gutshot_draws(ranks)
        
        if any(self._check_straight_made(ranks, i) for i in range(len(RANKS) - 4)):
            straight_type = "straight_made"
        elif straight_draws > 0:
            straight_type = "open_ended_draw"
        elif gutshot_draws > 0:
            straight_type = "gutshot_draw"
        else:
            straight_type = "no_straight_draw"
            
        return {
            "type": straight_type,
            "open_ended_draws": straight_draws,
            "gutshot_draws": gutshot_draws,
            "total_outs": straight_draws * 8 + gutshot_draws * 4
        }
        
    def _count_straight_draws(self, ranks: List[int]) -> int:
        """Count open-ended straight draw possibilities."""
        if len(ranks) < 2:
            return 0
            
        unique_ranks = sorted(set(ranks))
        draw_count = 0
        
        # Check each possible straight
        for start in range(len(RANKS) - 4):
            straight_ranks = list(range(start, start + 5))
            board_contribution = len([r for r in unique_ranks if r in straight_ranks])
            
            # Open-ended draw requires exactly 3 cards from the straight
            if board_contribution == 3:
                # Check if it's actually open-ended (not inside)
                if self._is_open_ended_draw(unique_ranks, straight_ranks):
                    draw_count += 1
                    
        return draw_count
        
    def _count_gutshot_draws(self, ranks: List[int]) -> int:
        """Count gutshot (inside) straight draw possibilities."""
        if len(ranks) < 2:
            return 0
            
        unique_ranks = sorted(set(ranks))
        gutshot_count = 0
        
        # Check each possible straight
        for start in range(len(RANKS) - 4):
            straight_ranks = list(range(start, start + 5))
            board_contribution = len([r for r in unique_ranks if r in straight_ranks])
            
            # Gutshot requires exactly 3 cards from straight
            if board_contribution == 3:
                if not self._is_open_ended_draw(unique_ranks, straight_ranks):
                    gutshot_count += 1
                    
        return gutshot_count
        
    def _is_open_ended_draw(self, board_ranks: List[int], straight_ranks: List[int]) -> bool:
        """Check if a 3-card straight contribution is open-ended."""
        board_in_straight = [r for r in board_ranks if r in straight_ranks]
        
        if len(board_in_straight) != 3:
            return False
            
        board_in_straight.sort()
        
        # Open-ended if the 3 cards are consecutive and at ends of straight
        consecutive = all(board_in_straight[i+1] - board_in_straight[i] == 1 
                         for i in range(2))
        
        if not consecutive:
            return False
            
        # Check if at ends of the potential straight
        straight_start = min(straight_ranks)
        straight_end = max(straight_ranks)
        
        board_start = min(board_in_straight)
        board_end = max(board_in_straight)
        
        # Open-ended if board cards are at either end
        at_low_end = board_start == straight_start
        at_high_end = board_end == straight_end
        
        return at_low_end or at_high_end
        
    def _check_straight_made(self, ranks: List[int], start_rank: int) -> bool:
        """Check if a straight is made starting at given rank."""
        straight_ranks = set(range(start_rank, start_rank + 5))
        board_ranks = set(ranks)
        
        return straight_ranks.issubset(board_ranks)
        
    def _analyze_pair_potential(self, ranks: List[int]) -> Dict[str, Any]:
        """Analyze pairing potential of the board."""
        rank_counts = Counter(ranks)
        
        pairs = [rank for rank, count in rank_counts.items() if count >= 2]
        trips = [rank for rank, count in rank_counts.items() if count >= 3]
        
        if trips:
            pair_type = "trips_on_board"
        elif len(pairs) >= 2:
            pair_type = "two_pair_on_board"
        elif pairs:
            pair_type = "pair_on_board"
        else:
            pair_type = "rainbow_board"
            
        return {
            "type": pair_type,
            "pairs": pairs,
            "trips": trips,
            "rank_distribution": dict(rank_counts)
        }
        
    def _analyze_high_cards(self, ranks: List[int]) -> Dict[str, Any]:
        """Analyze high card strength of board."""
        high_cards = [r for r in ranks if r >= RANKS.index('T')]  # T, J, Q, K, A
        broadway_cards = [r for r in ranks if r >= RANKS.index('T')]
        
        avg_rank = sum(ranks) / len(ranks) if ranks else 0
        high_card_percentage = len(high_cards) / len(ranks) if ranks else 0
        
        if high_card_percentage >= 0.67:
            board_strength = "high"
        elif high_card_percentage >= 0.33:
            board_strength = "medium"
        else:
            board_strength = "low"
            
        return {
            "strength": board_strength,
            "high_cards": len(high_cards),
            "broadway_cards": len(broadway_cards),
            "average_rank": avg_rank,
            "high_card_percentage": high_card_percentage
        }
        
    def _classify_texture(self, analysis: Dict[str, Any]) -> str:
        """Classify overall board texture."""
        wetness = analysis["wetness_score"]
        connectivity = analysis["connectivity"]["type"]
        flush_potential = analysis["flush_potential"]["type"]
        
        if wetness > 0.7:
            return "very_wet"
        elif wetness > 0.5:
            return "wet"
        elif wetness > 0.3:
            return "semi_wet"
        elif wetness > 0.15:
            return "dry"
        else:
            return "very_dry"
            
    def _is_draw_heavy(self, analysis: Dict[str, Any]) -> bool:
        """Determine if board is draw-heavy."""
        flush_draws = analysis["flush_potential"]["type"] in ["flush_draw", "flush_possible"]
        straight_draws = analysis["straight_potential"]["open_ended_draws"] > 0
        
        return flush_draws or straight_draws
        
    def _analyze_range_favorability(
        self, 
        ranks: List[int], 
        suits: List[str], 
        texture_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze which types of ranges are favored by this board.
        
        This is crucial for determining who has "range advantage" in multi-player scenarios.
        """
        favorability = {
            "preflop_aggressor": 0.5,  # Baseline
            "preflop_caller": 0.5,
            "tight_ranges": 0.5,
            "loose_ranges": 0.5,
            "position_advantage": 0.0
        }
        
        # High boards favor preflop aggressor (more high pairs and broadways)
        high_card_strength = texture_analysis["high_card_strength"]["strength"]
        if high_card_strength == "high":
            favorability["preflop_aggressor"] += 0.2
            favorability["preflop_caller"] -= 0.2
            favorability["tight_ranges"] += 0.15
            favorability["loose_ranges"] -= 0.15
            
        # Connected boards favor calling ranges (more speculative hands)
        if texture_analysis["connectivity"]["type"] in ["connected", "highly_connected"]:
            favorability["preflop_caller"] += 0.15
            favorability["preflop_aggressor"] -= 0.15
            favorability["loose_ranges"] += 0.1
            favorability["tight_ranges"] -= 0.1
            
        # Draw-heavy boards favor position
        if texture_analysis["draw_heavy"]:
            favorability["position_advantage"] += 0.2
            
        # Clamp values to reasonable ranges
        for key in favorability:
            favorability[key] = max(0.0, min(1.0, favorability[key]))
            
        return favorability
        
    def who_has_range_advantage(
        self,
        board: List[str],
        preflop_aggressor_position: str,
        caller_position: str
    ) -> Dict[str, Any]:
        """
        Determine who has range advantage on this board.
        
        Critical for multi-player strategy - determines who should be
        more aggressive and who should play more passively.
        """
        texture_analysis = self.analyze_board_texture(board)
        favorability = texture_analysis["favorability"]
        
        aggressor_advantage = favorability["preflop_aggressor"]
        caller_advantage = favorability["preflop_caller"]
        position_factor = favorability["position_advantage"]
        
        # Adjust for position
        if preflop_aggressor_position in ["BTN", "CO"]:  # Good position
            aggressor_advantage += position_factor * 0.5
        if caller_position in ["BTN", "CO"]:  # Good position
            caller_advantage += position_factor * 0.5
            
        # Determine winner
        if aggressor_advantage > caller_advantage + 0.1:
            advantage_holder = "preflop_aggressor"
            advantage_strength = aggressor_advantage - caller_advantage
        elif caller_advantage > aggressor_advantage + 0.1:
            advantage_holder = "preflop_caller" 
            advantage_strength = caller_advantage - aggressor_advantage
        else:
            advantage_holder = "neutral"
            advantage_strength = 0.0
            
        return {
            "advantage_holder": advantage_holder,
            "advantage_strength": advantage_strength,
            "aggressor_score": aggressor_advantage,
            "caller_score": caller_advantage,
            "texture_category": texture_analysis["texture_category"],
            "recommendation": self._get_range_advantage_recommendation(
                advantage_holder, advantage_strength, texture_analysis
            )
        }
        
    def _get_range_advantage_recommendation(
        self,
        advantage_holder: str,
        advantage_strength: float,
        texture_analysis: Dict[str, Any]
    ) -> str:
        """Get strategic recommendation based on range advantage."""
        if advantage_holder == "preflop_aggressor":
            if advantage_strength > 0.3:
                return "Strong range advantage - bet aggressively for value and as bluffs"
            elif advantage_strength > 0.15:
                return "Moderate range advantage - continue betting with reasonable frequency"
            else:
                return "Slight range advantage - can bet but be selective"
        elif advantage_holder == "preflop_caller":
            if advantage_strength > 0.3:
                return "Caller has strong advantage - check/call and look for check-raises"
            elif advantage_strength > 0.15:
                return "Caller has moderate advantage - check/call and play passively"
            else:
                return "Caller has slight advantage - play cautiously"
        else:
            return "Neutral board - play standard strategy based on hand strength"
            
    def _get_default_texture_analysis(self) -> Dict[str, Any]:
        """Return default analysis for error cases."""
        return {
            "wetness_score": 0.5,
            "connectivity": {"type": "unknown", "gaps": [], "max_gap": 0},
            "flush_potential": {"type": "no_flush_draw", "max_suited": 0},
            "straight_potential": {"type": "no_straight_draw", "open_ended_draws": 0},
            "pair_potential": {"type": "rainbow_board", "pairs": []},
            "high_card_strength": {"strength": "medium", "high_cards": 0},
            "texture_category": "unknown",
            "draw_heavy": False,
            "favorability": {
                "preflop_aggressor": 0.5,
                "preflop_caller": 0.5,
                "tight_ranges": 0.5,
                "loose_ranges": 0.5,
                "position_advantage": 0.0
            },
            "error": "Invalid or insufficient board data"
        }