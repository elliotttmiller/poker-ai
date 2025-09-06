"""
Opponent Modeler Module for Project PokerMind.

This module tracks and analyzes opponent behavior to build comprehensive
player profiles for exploitative play.
"""

import logging
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
from dataclasses import dataclass
import statistics


@dataclass
class PlayerStats:
    """Statistics for a single opponent."""

    name: str
    hands_played: int = 0

    # Preflop stats
    vpip: float = 0.0  # Voluntarily Put In Pot
    pfr: float = 0.0  # Pre-Flop Raise

    # Postflop stats
    cbet: float = 0.0  # Continuation Bet
    fold_to_cbet: float = 0.0
    wtsd: float = 0.0  # Went To Showdown

    # Aggression
    aggression_factor: float = 1.0
    total_bets_raises: int = 0
    total_calls: int = 0

    # Tendencies
    avg_bet_size_ratio: float = 0.5  # Bet size as ratio of pot
    bluff_frequency: float = 0.2

    # Recent actions (sliding window)
    recent_actions: deque = None

    def __post_init__(self):
        if self.recent_actions is None:
            self.recent_actions = deque(maxlen=50)


class OpponentModeler:
    """
    Opponent modeling system for tracking and analyzing player behavior.

    Builds comprehensive profiles of opponents to enable exploitative play
    while maintaining efficient real-time updates.
    """

    def __init__(self, history_window: int = 100):
        """
        Initialize the opponent modeler.

        Args:
            history_window: Number of hands to track for each player
        """
        self.logger = logging.getLogger(__name__)
        self.history_window = history_window

        # Player tracking
        self.player_stats: Dict[str, PlayerStats] = {}
        self.action_histories: Dict[str, List[Dict]] = defaultdict(list)

        # Game context
        self.current_street = "preflop"
        self.current_aggressor = None

        self.logger.info("Opponent Modeler initialized")

    def get_opponent_analysis(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get analysis of all opponents in the current game state.

        Enhanced in Phase 5 to provide confidence scoring for weighted blending.

        Args:
            game_state: Current game state

        Returns:
            Dict containing opponent analysis with enhanced confidence metrics
        """
        try:
            seats = game_state.get("seats", [])
            our_seat_id = game_state.get("our_seat_id")

            opponent_analysis = {}
            exploitable_opportunities = []
            analysis_confidence_scores = []

            for seat in seats:
                if seat.get("seat_id") == our_seat_id:
                    continue  # Skip ourselves

                player_name = seat.get("name", f"Player_{seat.get('seat_id')}")
                player_analysis = self._analyze_player(player_name, game_state)

                if player_analysis:
                    opponent_analysis[player_name] = player_analysis
                    analysis_confidence_scores.append(
                        player_analysis.get("confidence", 0.5)
                    )

                    # Check for exploitable patterns
                    exploits = self._identify_exploits(player_name, player_analysis)
                    exploitable_opportunities.extend(exploits)

            # Calculate overall confidence for opponent modeling
            overall_confidence = (
                statistics.mean(analysis_confidence_scores)
                if analysis_confidence_scores
                else 0.0
            )

            return {
                "opponents": opponent_analysis,
                "exploit_opportunities": exploitable_opportunities,
                "table_dynamics": self._analyze_table_dynamics(opponent_analysis),
                "recommended_adjustments": self._get_strategy_adjustments(
                    opponent_analysis, game_state
                ),
                "confidence": overall_confidence,
                "confidence_breakdown": {
                    "sample_size_confidence": self._calculate_sample_size_confidence(),
                    "data_freshness_confidence": self._calculate_data_freshness_confidence(),
                    "pattern_consistency_confidence": self._calculate_pattern_consistency_confidence(),
                },
            }

        except Exception as e:
            self.logger.warning(f"Opponent analysis error: {e}")
            return {
                "opponents": {},
                "exploit_opportunities": [],
                "confidence": 0.0,
                "confidence_breakdown": {},
            }

    def _analyze_player(
        self, player_name: str, game_state: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze a specific player's tendencies.

        Enhanced in Phase 5 to include confidence scoring.
        """
        if player_name not in self.player_stats:
            return None  # Not enough data

        stats = self.player_stats[player_name]

        if stats.hands_played < 5:
            return None  # Need more hands for meaningful analysis

        # Categorize player type
        player_type = self._categorize_player_type(stats)

        # Calculate exploitability metrics
        exploitability = self._calculate_exploitability(stats)

        # Calculate confidence for this player analysis
        player_confidence = self._calculate_player_confidence(stats)

        return {
            "type": player_type,
            "classification": player_type,  # For compatibility with synthesizer
            "stats": {
                "vpip": stats.vpip,
                "pfr": stats.pfr,
                "aggression_factor": stats.aggression_factor,
                "cbet": stats.cbet,
                "wtsd": stats.wtsd,
                "hands_played": stats.hands_played,
            },
            "exploitability": exploitability,
            "recent_pattern": self._get_recent_pattern(stats),
            "recommended_strategy": self._get_recommended_strategy(player_type, stats),
            "confidence": player_confidence,
            "confidence_breakdown": {
                "sample_size": min(
                    1.0, stats.hands_played / 50.0
                ),  # More hands = more confidence
                "consistency": self._calculate_consistency_score(stats),
                "recency": self._calculate_recency_score(stats),
            },
        }

    def _categorize_player_type(self, stats: PlayerStats) -> str:
        """Categorize player based on their statistics."""
        vpip = stats.vpip
        pfr = stats.pfr
        af = stats.aggression_factor

        # Tight/Loose classification
        if vpip < 0.15:
            tightness = "tight"
        elif vpip > 0.35:
            tightness = "loose"
        else:
            tightness = "normal"

        # Passive/Aggressive classification
        if af < 1.0:
            aggression = "passive"
        elif af > 3.0:
            aggression = "aggressive"
        else:
            aggression = "normal"

        # Combined classification
        return f"{tightness}_{aggression}"

    def _calculate_exploitability(self, stats: PlayerStats) -> Dict[str, float]:
        """Calculate how exploitable a player is in different situations."""
        exploitability = {}

        # Fold equity (how often they fold to aggression)
        fold_tendency = 1.0 - (stats.total_calls / max(stats.hands_played, 1))
        exploitability["fold_equity"] = min(fold_tendency, 1.0)

        # Bluff catching (how often they call with weak hands)
        exploitability["bluff_catcher"] = stats.wtsd / max(stats.hands_played, 1)

        # Bet folding (how often they fold to bets after showing interest)
        exploitability["bet_fold"] = stats.fold_to_cbet

        # Predictability (consistency in bet sizing)
        exploitability["predictability"] = 1.0 - abs(stats.avg_bet_size_ratio - 0.5) * 2

        return exploitability

    def _identify_exploits(
        self, player_name: str, analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify specific exploitable patterns for a player."""
        exploits = []
        stats = analysis.get("stats", {})
        exploitability = analysis.get("exploitability", {})

        # High fold equity - can bluff more
        if exploitability.get("fold_equity", 0) > 0.7:
            exploits.append(
                {
                    "type": "bluff_opportunity",
                    "player": player_name,
                    "confidence": exploitability["fold_equity"],
                    "description": f"{player_name} folds too often to aggression",
                }
            )

        # Calling station - value bet more
        if stats.get("wtsd", 0) > 0.4:
            exploits.append(
                {
                    "type": "value_bet_opportunity",
                    "player": player_name,
                    "confidence": stats["wtsd"],
                    "description": f"{player_name} calls too wide - value bet thin",
                }
            )

        # Tight player - steal blinds more
        if stats.get("vpip", 0) < 0.15:
            exploits.append(
                {
                    "type": "steal_opportunity",
                    "player": player_name,
                    "confidence": 1.0 - stats["vpip"],
                    "description": f"{player_name} is very tight - steal more",
                }
            )

        return exploits

    def _analyze_table_dynamics(
        self, opponent_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze overall table dynamics."""
        if not opponent_analysis:
            return {}

        # Calculate table averages
        total_players = len(opponent_analysis)
        avg_vpip = statistics.mean(
            [p["stats"]["vpip"] for p in opponent_analysis.values()]
        )
        avg_aggression = statistics.mean(
            [p["stats"]["aggression_factor"] for p in opponent_analysis.values()]
        )

        # Classify table type
        if avg_vpip < 0.2:
            table_type = "tight"
        elif avg_vpip > 0.35:
            table_type = "loose"
        else:
            table_type = "normal"

        if avg_aggression > 2.5:
            table_type += "_aggressive"
        elif avg_aggression < 1.5:
            table_type += "_passive"

        return {
            "type": table_type,
            "avg_vpip": avg_vpip,
            "avg_aggression": avg_aggression,
            "total_opponents": total_players,
            "exploitable_count": sum(
                1
                for p in opponent_analysis.values()
                if p.get("exploitability", {}).get("fold_equity", 0) > 0.6
            ),
        }

    def _get_strategy_adjustments(
        self, opponent_analysis: Dict[str, Any], game_state: Dict[str, Any]
    ) -> List[str]:
        """Get recommended strategy adjustments based on opponent analysis."""
        adjustments = []

        if not opponent_analysis:
            return adjustments

        # Analyze fold equity opportunities
        high_fold_equity_count = sum(
            1
            for p in opponent_analysis.values()
            if p.get("exploitability", {}).get("fold_equity", 0) > 0.7
        )

        if high_fold_equity_count >= len(opponent_analysis) * 0.6:
            adjustments.append("increase_bluff_frequency")

        # Analyze calling station opportunities
        calling_stations = sum(
            1
            for p in opponent_analysis.values()
            if p.get("stats", {}).get("wtsd", 0) > 0.4
        )

        if calling_stations >= len(opponent_analysis) * 0.5:
            adjustments.append("increase_value_betting")

        # Position-based adjustments
        if game_state.get("street") == "preflop":
            tight_players = sum(
                1
                for p in opponent_analysis.values()
                if p.get("stats", {}).get("vpip", 0) < 0.15
            )

            if tight_players >= len(opponent_analysis) * 0.7:
                adjustments.append("increase_steal_frequency")

        return adjustments

    def _get_recent_pattern(self, stats: PlayerStats) -> str:
        """Analyze recent playing pattern."""
        if len(stats.recent_actions) < 5:
            return "insufficient_data"

        recent = list(stats.recent_actions)[-10:]
        aggressive_actions = sum(
            1 for action in recent if action.get("type") in ["bet", "raise"]
        )

        if aggressive_actions >= len(recent) * 0.7:
            return "aggressive_streak"
        elif aggressive_actions <= len(recent) * 0.2:
            return "passive_streak"
        else:
            return "balanced"

    def _get_recommended_strategy(
        self, player_type: str, stats: PlayerStats
    ) -> List[str]:
        """Get recommended strategy against this player type."""
        strategies = []

        type_parts = player_type.split("_")
        tightness = type_parts[0]
        aggression = type_parts[1] if len(type_parts) > 1 else "normal"

        if tightness == "tight":
            strategies.extend(["steal_blinds", "fold_to_resistance"])
        elif tightness == "loose":
            strategies.extend(["value_bet_thin", "avoid_bluffs"])

        if aggression == "aggressive":
            strategies.extend(["trap_with_strong_hands", "fold_marginal"])
        elif aggression == "passive":
            strategies.extend(["bet_for_value", "bluff_more"])

        return strategies

    def get_basic_stats(self, player_name: str) -> Optional[Dict[str, Any]]:
        """
        Get basic statistics for a player.

        Args:
            player_name: Name of the player

        Returns:
            Dict containing basic stats, or None if player not found
        """
        if player_name not in self.player_stats:
            return None

        stats = self.player_stats[player_name]

        return {
            "hands_played": stats.hands_played,
            "vpip": stats.vpip,
            "pfr": stats.pfr,
            "aggression_factor": stats.aggression_factor,
            "cbet": stats.cbet,
            "fold_to_cbet": stats.fold_to_cbet,
            "wtsd": stats.wtsd,
            "avg_bet_size_ratio": stats.avg_bet_size_ratio,
            "bluff_frequency": stats.bluff_frequency,
            "total_bets_raises": stats.total_bets_raises,
            "total_calls": stats.total_calls,
        }

    # Update methods called by the cognitive core
    def update(
        self,
        player_name: str,
        action: str,
        amount: int = 0,
        street: str = "preflop",
        pot_size: int = 0,
    ):
        """
        Simple update method as requested in directive.

        Args:
            player_name: Name of the player
            action: Action taken ('call', 'raise', 'fold', etc.)
            amount: Amount bet/raised (0 for fold/check)
            street: Current street ('preflop', 'flop', 'turn', 'river')
            pot_size: Current pot size
        """
        if not player_name:
            return

        if player_name not in self.player_stats:
            self.player_stats[player_name] = PlayerStats(name=player_name)

        stats = self.player_stats[player_name]

        # Record the action
        action_record = {
            "type": action,
            "amount": amount,
            "street": street,
            "pot_size": pot_size,
        }

        stats.recent_actions.append(action_record)

        # Update statistics based on action
        self._update_stats_from_action(stats, action_record)

    def get_profile(self, player_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a player's profile with simple classification as requested in directive.

        Args:
            player_name: Name of the player

        Returns:
            Dict containing player profile and classification, or None if insufficient data
        """
        if player_name not in self.player_stats:
            return None

        stats = self.player_stats[player_name]

        if stats.hands_played < 3:
            return None  # Need minimum hands for meaningful profile

        # Simple classification heuristic as requested
        classification = "unknown"

        if stats.vpip > 0.4:
            classification = "loose"
        elif stats.vpip < 0.2:
            classification = "tight"
        else:
            classification = "normal"

        if stats.pfr < 0.1:
            classification += "_passive"
        elif stats.pfr > 0.25:
            classification += "_aggressive"

        return {
            "name": player_name,
            "classification": classification,
            "hands_played": stats.hands_played,
            "vpip": stats.vpip,  # Voluntarily Put money In Pot
            "pfr": stats.pfr,  # Pre-Flop Raise
            "aggression_factor": stats.aggression_factor,
            "recent_pattern": self._get_recent_pattern(stats),
        }

    def update_from_action(self, action: Dict[str, Any], round_state: Dict[str, Any]):
        """Update opponent model based on observed action."""
        player_name = action.get("player", {}).get("name")
        if not player_name:
            return

        if player_name not in self.player_stats:
            self.player_stats[player_name] = PlayerStats(name=player_name)

        stats = self.player_stats[player_name]

        # Record the action
        action_record = {
            "type": action.get("action"),
            "amount": action.get("amount", 0),
            "street": round_state.get("street", "preflop"),
            "pot_size": round_state.get("pot", {}).get("main", {}).get("amount", 0),
        }

        stats.recent_actions.append(action_record)

        # Update statistics based on action
        self._update_stats_from_action(stats, action_record)

    def _update_stats_from_action(self, stats: PlayerStats, action: Dict[str, Any]):
        """Update player statistics based on a single action."""
        action_type = action.get("type")
        street = action.get("street", "preflop")

        # Update hand count - each call to update represents a new hand in this simplified model
        if street == "preflop":
            stats.hands_played += 1

        # Update VPIP (any voluntary money put in pot)
        if street == "preflop" and action_type in ["call", "raise", "bet"]:
            # Count voluntary actions from recent actions
            voluntary_hands = sum(
                1
                for a in stats.recent_actions
                if a.get("street") == "preflop"
                and a.get("type") in ["call", "raise", "bet"]
            )
            stats.vpip = voluntary_hands / max(stats.hands_played, 1)
        elif street == "preflop":
            # Recalculate VPIP to account for fold actions too
            voluntary_hands = sum(
                1
                for a in stats.recent_actions
                if a.get("street") == "preflop"
                and a.get("type") in ["call", "raise", "bet"]
            )
            stats.vpip = voluntary_hands / max(stats.hands_played, 1)

        # Update PFR (preflop raises)
        if street == "preflop" and action_type in ["raise", "bet"]:
            raise_hands = sum(
                1
                for a in stats.recent_actions
                if a.get("street") == "preflop" and a.get("type") in ["raise", "bet"]
            )
            stats.pfr = raise_hands / max(stats.hands_played, 1)

        # Update aggression factor
        if action_type in ["bet", "raise"]:
            stats.total_bets_raises += 1
        elif action_type == "call":
            stats.total_calls += 1

        if stats.total_calls > 0:
            stats.aggression_factor = stats.total_bets_raises / stats.total_calls

    def update_from_result(
        self, winners: list, hand_info: list, round_state: Dict[str, Any]
    ):
        """Update opponent model based on showdown result."""
        # TODO: Implement showdown analysis
        # This would update WTSD, showdown tendencies, etc.
        pass

    def reset(self):
        """Reset opponent model for a new session."""
        self.player_stats.clear()
        self.action_histories.clear()

    # Phase 5 Enhancement: Confidence Calculation Methods
    def _calculate_player_confidence(self, stats: PlayerStats) -> float:
        """Calculate confidence score for a player's analysis."""
        # Sample size confidence
        sample_confidence = min(1.0, stats.hands_played / 50.0)

        # Consistency confidence
        consistency_confidence = self._calculate_consistency_score(stats)

        # Recency confidence
        recency_confidence = self._calculate_recency_score(stats)

        # Combined confidence
        confidence = (
            0.4 * sample_confidence
            + 0.3 * consistency_confidence
            + 0.3 * recency_confidence
        )
        return min(1.0, max(0.0, confidence))

    def _calculate_sample_size_confidence(self) -> float:
        """Calculate confidence based on overall sample size."""
        if not self.player_stats:
            return 0.0

        avg_hands = statistics.mean(
            [stats.hands_played for stats in self.player_stats.values()]
        )
        return min(1.0, avg_hands / 30.0)  # Confident with 30+ hands average

    def _calculate_data_freshness_confidence(self) -> float:
        """Calculate confidence based on data freshness."""
        # For now, assume all data is fresh
        # In a real implementation, this would consider time since last action
        return 0.8

    def _calculate_pattern_consistency_confidence(self) -> float:
        """Calculate confidence based on pattern consistency."""
        if not self.player_stats:
            return 0.0

        consistency_scores = [
            self._calculate_consistency_score(stats)
            for stats in self.player_stats.values()
        ]
        return statistics.mean(consistency_scores) if consistency_scores else 0.0

    def _calculate_consistency_score(self, stats: PlayerStats) -> float:
        """Calculate how consistent a player's behavior is."""
        if len(stats.recent_actions) < 10:
            return 0.5  # Medium confidence for small samples

        # Analyze consistency in recent actions
        recent = list(stats.recent_actions)[-20:]
        action_types = [action.get("type", "unknown") for action in recent]

        # Calculate variance in action types
        action_counts = {}
        for action_type in action_types:
            action_counts[action_type] = action_counts.get(action_type, 0) + 1

        # If player has very mixed actions, consistency is lower
        if len(action_counts) > 4:  # Too many different actions
            return 0.3
        elif len(action_counts) <= 2:  # Very consistent
            return 0.9
        else:
            return 0.6

    def _calculate_recency_score(self, stats: PlayerStats) -> float:
        """Calculate recency score based on recent activity."""
        if len(stats.recent_actions) < 5:
            return 0.3

        # More recent actions = higher confidence
        recent_actions = len(stats.recent_actions)
        return min(1.0, recent_actions / 20.0)
        self.logger.info("Opponent model reset")
