"""
Post-Game Self-Improvement Analyzer for Project PokerMind.

This module implements autonomous analysis of multi-player poker sessions
to identify leaks and generate improvement recommendations.

Based on professional poker analysis methodology and GTO principles.
"""

import logging
import json
import statistics
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict, Counter

from .helpers import estimate_preflop_hand_strength
from .gto_tools import calculate_mdf, calculate_pot_equity_needed
from .board_analyzer import BoardAnalyzer


class PostGameAnalyzer:
    """
    Autonomous self-improvement analysis system for multi-player sessions.

    Analyzes decision patterns, identifies statistical deviations from GTO,
    and generates actionable improvement recommendations.
    """

    def __init__(self):
        """Initialize the post-game analyzer."""
        self.logger = logging.getLogger(__name__)
        self.board_analyzer = BoardAnalyzer()

        # Statistical tracking
        self.session_stats = {
            "hands_analyzed": 0,
            "total_decisions": 0,
            "gto_deviations": [],
            "position_stats": defaultdict(dict),
            "opponent_interaction_stats": defaultdict(dict),
            "board_texture_performance": defaultdict(dict),
        }

    def find_my_leaks(self, session_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze session logs to identify leaks and deviations from optimal play.

        Args:
            session_logs: List of decision packets from multi-player games

        Returns:
            Dict containing comprehensive leak analysis
        """
        if not session_logs:
            return {"error": "No session data provided"}

        try:
            # Initialize analysis results
            leak_analysis = {
                "session_overview": self._analyze_session_overview(session_logs),
                "positional_leaks": self._analyze_positional_play(session_logs),
                "multi_way_pot_leaks": self._analyze_multi_way_performance(session_logs),
                "board_texture_leaks": self._analyze_board_texture_performance(session_logs),
                "betting_pattern_leaks": self._analyze_betting_patterns(session_logs),
                "opponent_adaptation_leaks": self._analyze_opponent_adaptation(session_logs),
                "statistical_deviations": self._calculate_statistical_deviations(session_logs),
                "priority_improvements": [],
                "confidence": 0.0,
            }

            # Identify priority improvement areas
            leak_analysis["priority_improvements"] = self._prioritize_improvements(leak_analysis)

            # Calculate overall analysis confidence
            leak_analysis["confidence"] = self._calculate_analysis_confidence(session_logs)

            return leak_analysis

        except Exception as e:
            self.logger.error(f"Leak analysis error: {e}")
            return {"error": f"Analysis failed: {str(e)}"}

    def _analyze_session_overview(self, session_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall session statistics."""
        total_hands = len(session_logs)

        # Extract key decision metrics
        actions_taken = [
            log.get("final_action", {}).get("action", "unknown") for log in session_logs
        ]
        action_distribution = dict(Counter(actions_taken))

        # Calculate confidence scores
        confidence_scores = [log.get("confidence_score", 0.5) for log in session_logs]
        avg_confidence = statistics.mean(confidence_scores) if confidence_scores else 0.5

        # Analyze streets played
        streets_played = [log.get("street", "preflop") for log in session_logs]
        street_distribution = dict(Counter(streets_played))

        return {
            "total_hands": total_hands,
            "action_distribution": action_distribution,
            "average_confidence": avg_confidence,
            "street_distribution": street_distribution,
            "confidence_range": {
                "min": min(confidence_scores) if confidence_scores else 0,
                "max": max(confidence_scores) if confidence_scores else 0,
                "std_dev": statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0,
            },
        }

    def _analyze_positional_play(self, session_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze positional play for leaks."""
        positional_stats = defaultdict(
            lambda: {
                "hands_played": 0,
                "actions": defaultdict(int),
                "avg_confidence": 0.0,
                "confidence_scores": [],
            }
        )

        for log in session_logs:
            # Extract position information
            position = self._extract_position(log)
            if not position:
                continue

            action = log.get("final_action", {}).get("action", "unknown")
            confidence = log.get("confidence_score", 0.5)

            positional_stats[position]["hands_played"] += 1
            positional_stats[position]["actions"][action] += 1
            positional_stats[position]["confidence_scores"].append(confidence)

        # Calculate averages and identify leaks
        position_leaks = {}
        for position, stats in positional_stats.items():
            if stats["hands_played"] < 5:  # Skip positions with insufficient data
                continue

            avg_confidence = statistics.mean(stats["confidence_scores"])

            # Calculate action frequencies
            total_actions = sum(stats["actions"].values())
            action_frequencies = {
                action: count / total_actions for action, count in stats["actions"].items()
            }

            # Identify potential leaks
            leaks = []

            # Check for over-folding in late position
            if position in ["BTN", "CO"] and action_frequencies.get("fold", 0) > 0.6:
                leaks.append("Potentially over-folding in late position")

            # Check for under-folding in early position
            if position in ["UTG", "UTG+1"] and action_frequencies.get("fold", 0) < 0.4:
                leaks.append("Potentially under-folding in early position")

            # Check for low confidence in position
            if avg_confidence < 0.6:
                leaks.append("Low confidence decisions in this position")

            position_leaks[position] = {
                "hands_played": stats["hands_played"],
                "action_frequencies": action_frequencies,
                "average_confidence": avg_confidence,
                "identified_leaks": leaks,
            }

        return position_leaks

    def _analyze_multi_way_performance(self, session_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance in multi-way pots."""
        multi_way_stats = {
            "heads_up_pots": {"count": 0, "actions": defaultdict(int)},
            "three_way_pots": {"count": 0, "actions": defaultdict(int)},
            "four_plus_way_pots": {"count": 0, "actions": defaultdict(int)},
        }

        for log in session_logs:
            # Determine number of opponents
            num_opponents = self._count_active_opponents(log)
            action = log.get("final_action", {}).get("action", "unknown")

            if num_opponents <= 1:
                multi_way_stats["heads_up_pots"]["count"] += 1
                multi_way_stats["heads_up_pots"]["actions"][action] += 1
            elif num_opponents == 2:
                multi_way_stats["three_way_pots"]["count"] += 1
                multi_way_stats["three_way_pots"]["actions"][action] += 1
            else:
                multi_way_stats["four_plus_way_pots"]["count"] += 1
                multi_way_stats["four_plus_way_pots"]["actions"][action] += 1

        # Identify multi-way specific leaks
        leaks = []

        # Check if playing too loose in multi-way pots
        for pot_type, stats in multi_way_stats.items():
            if stats["count"] < 3:  # Skip if insufficient data
                continue

            total_actions = sum(stats["actions"].values())
            if total_actions == 0:
                continue

            call_frequency = stats["actions"]["call"] / total_actions
            raise_frequency = stats["actions"]["raise"] / total_actions
            fold_frequency = stats["actions"]["fold"] / total_actions

            if pot_type == "four_plus_way_pots":
                # Should be more conservative in multi-way pots
                if call_frequency > 0.4:
                    leaks.append(f"Potentially calling too frequently in {pot_type}")
                if raise_frequency > 0.2:
                    leaks.append(f"Potentially raising too frequently in {pot_type}")

        return {"pot_type_distribution": multi_way_stats, "identified_leaks": leaks}

    def _analyze_board_texture_performance(
        self, session_logs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze performance on different board textures."""
        texture_stats = defaultdict(
            lambda: {"hands_played": 0, "actions": defaultdict(int), "confidence_scores": []}
        )

        for log in session_logs:
            community_cards = log.get("community_cards", [])
            if len(community_cards) < 3:  # Only analyze flop+ situations
                continue

            # Analyze board texture
            texture_analysis = self.board_analyzer.analyze_board_texture(community_cards)
            texture_category = texture_analysis.get("texture_category", "unknown")

            action = log.get("final_action", {}).get("action", "unknown")
            confidence = log.get("confidence_score", 0.5)

            texture_stats[texture_category]["hands_played"] += 1
            texture_stats[texture_category]["actions"][action] += 1
            texture_stats[texture_category]["confidence_scores"].append(confidence)

        # Identify texture-specific leaks
        texture_leaks = {}

        for texture, stats in texture_stats.items():
            if stats["hands_played"] < 3:
                continue

            total_actions = sum(stats["actions"].values())
            if total_actions == 0:
                continue

            action_frequencies = {
                action: count / total_actions for action, count in stats["actions"].items()
            }

            avg_confidence = statistics.mean(stats["confidence_scores"])

            leaks = []

            # Texture-specific leak detection
            if texture in ["very_wet", "wet"]:
                # Should be more cautious on wet boards
                if action_frequencies.get("raise", 0) > 0.3:
                    leaks.append("Potentially over-aggressive on wet boards")
            elif texture in ["very_dry", "dry"]:
                # Can be more aggressive on dry boards
                if action_frequencies.get("fold", 0) > 0.5:
                    leaks.append("Potentially under-aggressive on dry boards")

            texture_leaks[texture] = {
                "hands_played": stats["hands_played"],
                "action_frequencies": action_frequencies,
                "average_confidence": avg_confidence,
                "identified_leaks": leaks,
            }

        return texture_leaks

    def _analyze_betting_patterns(self, session_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze betting patterns and sizing."""
        betting_stats = {
            "bet_sizes": [],
            "raise_sizes": [],
            "call_frequencies": defaultdict(int),
            "fold_frequencies": defaultdict(int),
        }

        for log in session_logs:
            final_action = log.get("final_action", {})
            action = final_action.get("action", "unknown")
            amount = final_action.get("amount", 0)
            pot_size = log.get("pot_size", 100)

            if action == "raise" and amount > 0 and pot_size > 0:
                bet_ratio = amount / pot_size
                betting_stats["raise_sizes"].append(bet_ratio)
            elif action in ["bet"] and amount > 0 and pot_size > 0:
                bet_ratio = amount / pot_size
                betting_stats["bet_sizes"].append(bet_ratio)

        # Analyze patterns
        leaks = []

        if betting_stats["bet_sizes"]:
            avg_bet_size = statistics.mean(betting_stats["bet_sizes"])
            bet_size_variance = (
                statistics.stdev(betting_stats["bet_sizes"])
                if len(betting_stats["bet_sizes"]) > 1
                else 0
            )

            # Check for sizing tells
            if bet_size_variance < 0.1:
                leaks.append("Very consistent bet sizing - may be exploitable")
            if avg_bet_size > 1.2:
                leaks.append("Average bet sizes may be too large")
            elif avg_bet_size < 0.4:
                leaks.append("Average bet sizes may be too small")

        return {"betting_statistics": betting_stats, "identified_leaks": leaks}

    def _analyze_opponent_adaptation(self, session_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how well we adapted to opponent tendencies."""
        opponent_interactions = defaultdict(
            lambda: {
                "encounters": 0,
                "actions_vs_opponent": defaultdict(int),
                "opponent_stats_observed": {},
            }
        )

        for log in session_logs:
            # Extract opponent information
            opponent_analysis = log.get("opponent_model", {})
            opponents = opponent_analysis.get("opponents", {})

            final_action = log.get("final_action", {}).get("action", "unknown")

            for opponent_name, opponent_data in opponents.items():
                opponent_interactions[opponent_name]["encounters"] += 1
                opponent_interactions[opponent_name]["actions_vs_opponent"][final_action] += 1

                # Store latest stats for this opponent
                if "stats" in opponent_data:
                    opponent_interactions[opponent_name]["opponent_stats_observed"] = opponent_data[
                        "stats"
                    ]

        # Identify adaptation leaks
        adaptation_leaks = []

        for opponent, data in opponent_interactions.items():
            if data["encounters"] < 5:  # Skip opponents with little interaction
                continue

            opponent_stats = data["opponent_stats_observed"]
            if not opponent_stats:
                continue

            # Check for adaptation to specific opponent types
            vpip = opponent_stats.get("vpip", 0.25)
            pfr = opponent_stats.get("pfr", 0.15)

            action_total = sum(data["actions_vs_opponent"].values())
            if action_total == 0:
                continue

            fold_frequency = data["actions_vs_opponent"]["fold"] / action_total

            # Check if we're adapting properly to tight opponents
            if vpip < 0.15 and fold_frequency < 0.6:
                adaptation_leaks.append(f"Not folding enough vs tight opponent {opponent}")

            # Check if we're adapting properly to loose opponents
            if vpip > 0.4 and fold_frequency > 0.4:
                adaptation_leaks.append(f"Folding too much vs loose opponent {opponent}")

        return {
            "opponent_interaction_summary": dict(opponent_interactions),
            "identified_leaks": adaptation_leaks,
        }

    def _calculate_statistical_deviations(
        self, session_logs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate statistical deviations from expected GTO play."""
        deviations = {
            "preflop_vpip": None,
            "preflop_pfr": None,
            "cbet_frequency": None,
            "fold_to_cbet_frequency": None,
            "aggression_factor": None,
        }

        # Calculate our own statistics
        preflop_voluntary_actions = 0
        preflop_raises = 0
        preflop_hands = 0

        cbets_made = 0
        cbet_opportunities = 0

        for log in session_logs:
            street = log.get("street", "preflop")
            action = log.get("final_action", {}).get("action", "unknown")

            if street == "preflop":
                preflop_hands += 1
                if action in ["call", "raise"]:
                    preflop_voluntary_actions += 1
                if action == "raise":
                    preflop_raises += 1

            # TODO: Add more sophisticated c-bet tracking

        # Calculate statistics
        if preflop_hands > 0:
            our_vpip = preflop_voluntary_actions / preflop_hands
            our_pfr = preflop_raises / preflop_hands

            # Compare to GTO expectations (rough approximations)
            expected_vpip = 0.25  # Varies by position
            expected_pfr = 0.18  # Varies by position

            deviations["preflop_vpip"] = {
                "actual": our_vpip,
                "expected": expected_vpip,
                "deviation": our_vpip - expected_vpip,
            }

            deviations["preflop_pfr"] = {
                "actual": our_pfr,
                "expected": expected_pfr,
                "deviation": our_pfr - expected_pfr,
            }

        return deviations

    def _prioritize_improvements(self, leak_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prioritize improvement areas based on impact and frequency."""
        improvements = []

        # Extract leaks from all categories
        all_leaks = []

        # Positional leaks
        for position, data in leak_analysis.get("positional_leaks", {}).items():
            for leak in data.get("identified_leaks", []):
                all_leaks.append(
                    {
                        "category": "positional",
                        "description": f"{position}: {leak}",
                        "frequency": data.get("hands_played", 0),
                        "priority_score": data.get("hands_played", 0) * 2,  # Position is important
                    }
                )

        # Multi-way pot leaks
        for leak in leak_analysis.get("multi_way_pot_leaks", {}).get("identified_leaks", []):
            all_leaks.append(
                {
                    "category": "multi_way",
                    "description": leak,
                    "frequency": 10,  # Assume moderate frequency
                    "priority_score": 15,  # Multi-way play is crucial
                }
            )

        # Board texture leaks
        for texture, data in leak_analysis.get("board_texture_leaks", {}).items():
            for leak in data.get("identified_leaks", []):
                all_leaks.append(
                    {
                        "category": "board_texture",
                        "description": f"{texture}: {leak}",
                        "frequency": data.get("hands_played", 0),
                        "priority_score": data.get("hands_played", 0),
                    }
                )

        # Betting pattern leaks
        for leak in leak_analysis.get("betting_pattern_leaks", {}).get("identified_leaks", []):
            all_leaks.append(
                {
                    "category": "betting_patterns",
                    "description": leak,
                    "frequency": 5,  # Assume moderate frequency
                    "priority_score": 12,  # Sizing leaks are important
                }
            )

        # Sort by priority score
        all_leaks.sort(key=lambda x: x["priority_score"], reverse=True)

        return all_leaks[:5]  # Return top 5 priority improvements

    def _calculate_analysis_confidence(self, session_logs: List[Dict[str, Any]]) -> float:
        """Calculate confidence in the analysis based on sample size."""
        sample_size = len(session_logs)

        # Confidence increases with sample size but plateaus
        if sample_size < 10:
            return 0.3
        elif sample_size < 50:
            return 0.3 + (sample_size - 10) / 40 * 0.4  # 0.3 to 0.7
        elif sample_size < 200:
            return 0.7 + (sample_size - 50) / 150 * 0.2  # 0.7 to 0.9
        else:
            return 0.9

    def _extract_position(self, log: Dict[str, Any]) -> Optional[str]:
        """Extract position from log entry."""
        # TODO: Implement proper position extraction from game state
        seats = log.get("seats", [])
        our_seat_id = log.get("our_seat_id")

        # For now, return a placeholder
        if our_seat_id and seats:
            return f"SEAT_{our_seat_id}"

        return "UNKNOWN"

    def _count_active_opponents(self, log: Dict[str, Any]) -> int:
        """Count number of active opponents."""
        seats = log.get("seats", [])
        our_seat_id = log.get("our_seat_id")

        # Count seats other than ours
        opponent_count = len([seat for seat in seats if seat.get("seat_id") != our_seat_id])

        return max(0, opponent_count)

    def generate_improvement_report(
        self, leak_analysis: Dict[str, Any], session_date: str = None
    ) -> str:
        """
        Generate a comprehensive improvement report.

        Args:
            leak_analysis: Results from find_my_leaks()
            session_date: Date of session analysis

        Returns:
            Formatted report string
        """
        if session_date is None:
            session_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report_lines = []

        # Header
        report_lines.append("=" * 60)
        report_lines.append("POKERMIND SELF-IMPROVEMENT ANALYSIS REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Analysis Date: {session_date}")
        report_lines.append(f"Analysis Confidence: {leak_analysis.get('confidence', 0.0):.1%}")
        report_lines.append("")

        # Session Overview
        overview = leak_analysis.get("session_overview", {})
        if overview:
            report_lines.append("SESSION OVERVIEW")
            report_lines.append("-" * 20)
            report_lines.append(f"Total Hands Analyzed: {overview.get('total_hands', 0)}")
            report_lines.append(
                f"Average Decision Confidence: {overview.get('average_confidence', 0.0):.1%}"
            )

            action_dist = overview.get("action_distribution", {})
            if action_dist:
                report_lines.append("Action Distribution:")
                for action, count in action_dist.items():
                    percentage = count / overview.get("total_hands", 1) * 100
                    report_lines.append(f"  {action.capitalize()}: {count} ({percentage:.1f}%)")
            report_lines.append("")

        # Priority Improvements
        priority_improvements = leak_analysis.get("priority_improvements", [])
        if priority_improvements:
            report_lines.append("PRIORITY IMPROVEMENTS")
            report_lines.append("-" * 25)
            for i, improvement in enumerate(priority_improvements, 1):
                report_lines.append(
                    f"{i}. [{improvement['category'].upper()}] {improvement['description']}"
                )
                report_lines.append(f"   Priority Score: {improvement['priority_score']}")
            report_lines.append("")

        # Statistical Deviations
        stats = leak_analysis.get("statistical_deviations", {})
        if any(v is not None for v in stats.values()):
            report_lines.append("STATISTICAL DEVIATIONS FROM GTO")
            report_lines.append("-" * 35)

            for stat_name, stat_data in stats.items():
                if stat_data is not None:
                    actual = stat_data.get("actual", 0)
                    expected = stat_data.get("expected", 0)
                    deviation = stat_data.get("deviation", 0)

                    report_lines.append(f"{stat_name.replace('_', ' ').title()}:")
                    report_lines.append(f"  Actual: {actual:.1%}, Expected: {expected:.1%}")
                    report_lines.append(f"  Deviation: {deviation:+.1%}")
            report_lines.append("")

        # Recommendations
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("-" * 15)

        if priority_improvements:
            report_lines.append("Focus on the following areas for maximum improvement:")
            report_lines.append("")

            for improvement in priority_improvements[:3]:  # Top 3
                category = improvement["category"]

                if category == "positional":
                    report_lines.append("• POSITIONAL PLAY:")
                    report_lines.append("  - Review preflop ranges for each position")
                    report_lines.append("  - Practice position-appropriate aggression levels")

                elif category == "multi_way":
                    report_lines.append("• MULTI-WAY POT STRATEGY:")
                    report_lines.append("  - Tighten ranges in multi-way scenarios")
                    report_lines.append("  - Focus on value betting over bluffing")

                elif category == "board_texture":
                    report_lines.append("• BOARD TEXTURE ADAPTATION:")
                    report_lines.append(
                        "  - Study continuation betting frequencies on different textures"
                    )
                    report_lines.append("  - Improve range advantage recognition")

                report_lines.append("")
        else:
            report_lines.append(
                "No major leaks identified. Continue current strategy and focus on:"
            )
            report_lines.append("• Maintaining consistent decision quality")
            report_lines.append("• Refining opponent-specific adaptations")
            report_lines.append("• Expanding multi-player strategy knowledge")

        report_lines.append("=" * 60)

        return "\n".join(report_lines)
