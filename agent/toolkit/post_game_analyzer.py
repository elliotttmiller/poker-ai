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

    def generate_full_report(
        self,
        session_logs: List[Dict[str, Any]],
        tournament_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive world-class agent-centric analytics report.

        This method implements the world-class analytics engine focused exclusively
        on the PokerMindAgent's performance with insight-driven metrics.

        Args:
            session_logs: List of decision packets from games/tournaments
            tournament_results: Optional list of tournament outcome data

        Returns:
            Comprehensive analytics dictionary with all specified metrics
        """
        if not session_logs:
            return {"error": "No session data provided for analysis"}

        try:
            full_report = {
                "report_timestamp": datetime.now().isoformat(),
                "sample_size": {
                    "total_hands": len(session_logs),
                    "tournaments": len(tournament_results) if tournament_results else 0,
                },
                # Overall Performance Metrics
                "overall_performance": self._calculate_overall_performance(
                    session_logs, tournament_results
                ),
                # Strategic Profile of Our Agent
                "strategic_profile": self._calculate_strategic_profile(session_logs),
                # Situational Performance Analysis
                "situational_performance": {
                    "by_tournament_stage": self._analyze_performance_by_tournament_stage(
                        session_logs
                    ),
                    "by_position": self._analyze_performance_by_position(session_logs),
                    "vs_opponent_archetype": self._analyze_performance_vs_opponent_types(
                        session_logs
                    ),
                },
                # Decision Quality Insights
                "decision_quality_insights": {
                    "cognitive_path_usage": self._analyze_cognitive_path_usage(session_logs),
                    "gto_vs_exploitation": self._analyze_gto_vs_exploitation(session_logs),
                    "confidence_analysis": self._analyze_confidence_patterns(session_logs),
                },
                # Advanced Analytics
                "advanced_metrics": {
                    "risk_management": self._analyze_risk_management(session_logs),
                    "adaptation_efficiency": self._analyze_adaptation_efficiency(session_logs),
                    "tournament_survival": self._analyze_tournament_survival(
                        session_logs, tournament_results
                    ),
                },
                # Recommendations and Insights
                "insights_and_recommendations": self._generate_strategic_insights(
                    session_logs, tournament_results
                ),
            }

            return full_report

        except Exception as e:
            self.logger.error(f"Error generating full report: {e}")
            return {"error": f"Report generation failed: {str(e)}"}

    def _calculate_overall_performance(
        self, session_logs: List[Dict[str, Any]], tournament_results: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Calculate overall performance metrics: ROI, ITM %, Average Finish, BB/100."""
        performance = {
            "roi": 0.0,
            "itm_percentage": 0.0,
            "average_finish": 0.0,
            "bb_per_100": 0.0,
            "total_winnings": 0.0,
            "total_buyins": 0.0,
        }

        if tournament_results:
            total_winnings = sum(result.get("winnings", 0) for result in tournament_results)
            total_buyins = sum(result.get("buyin", 0) for result in tournament_results)
            itm_count = sum(
                1
                for result in tournament_results
                if result.get("finish_position", 999) <= result.get("itm_positions", 0)
            )

            performance["total_winnings"] = total_winnings
            performance["total_buyins"] = total_buyins
            performance["roi"] = (
                ((total_winnings - total_buyins) / total_buyins * 100) if total_buyins > 0 else 0.0
            )
            performance["itm_percentage"] = (
                (itm_count / len(tournament_results) * 100) if tournament_results else 0.0
            )
            performance["average_finish"] = (
                statistics.mean([result.get("finish_position", 0) for result in tournament_results])
                if tournament_results
                else 0.0
            )

        # Calculate BB/100 from session logs
        total_bb_won = 0
        total_hands = len(session_logs)

        for log in session_logs:
            pot_won = log.get("pot_won", 0)
            big_blind = log.get("big_blind", 1)
            if big_blind > 0:
                total_bb_won += pot_won / big_blind

        performance["bb_per_100"] = (total_bb_won / total_hands * 100) if total_hands > 0 else 0.0

        return performance

    def _calculate_strategic_profile(self, session_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate agent's strategic profile: VPIP, PFR, 3-Bet %, Aggression Factor."""
        profile = {
            "vpip": 0.0,
            "pfr": 0.0,
            "three_bet_percentage": 0.0,
            "aggression_factor": 0.0,
            "cbet_frequency": 0.0,
        }

        preflop_hands = 0
        voluntary_actions = 0
        preflop_raises = 0
        three_bets = 0

        aggressive_actions = 0  # bets and raises
        passive_actions = 0  # calls

        cbet_opportunities = 0
        cbets_made = 0

        for log in session_logs:
            street = log.get("street", "preflop")
            action = log.get("final_action", {}).get("action", "fold")

            if street == "preflop":
                preflop_hands += 1
                if action in ["call", "raise"]:
                    voluntary_actions += 1
                if action == "raise":
                    preflop_raises += 1

                # Check for 3-bet (simplified logic)
                betting_history = log.get("betting_history", [])
                if len(betting_history) >= 3 and action == "raise":
                    three_bets += 1

            # Track aggression
            if action in ["bet", "raise"]:
                aggressive_actions += 1
            elif action == "call":
                passive_actions += 1

            # Track continuation betting (simplified)
            if street == "flop" and action in ["bet", "raise"]:
                cbet_opportunities += 1
                if action in ["bet", "raise"]:
                    cbets_made += 1

        # Calculate statistics
        if preflop_hands > 0:
            profile["vpip"] = voluntary_actions / preflop_hands
            profile["pfr"] = preflop_raises / preflop_hands
            profile["three_bet_percentage"] = three_bets / preflop_hands

        if passive_actions > 0:
            profile["aggression_factor"] = aggressive_actions / passive_actions

        if cbet_opportunities > 0:
            profile["cbet_frequency"] = cbets_made / cbet_opportunities

        return profile

    def _analyze_performance_by_tournament_stage(
        self, session_logs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze performance by tournament stage: Early (M > 20), Middle (10 < M <= 20), Late (M <= 10)."""
        stages = {
            "early": {"hands": 0, "bb_won": 0, "win_rate": 0.0},
            "middle": {"hands": 0, "bb_won": 0, "win_rate": 0.0},
            "late": {"hands": 0, "bb_won": 0, "win_rate": 0.0},
        }

        for log in session_logs:
            # Calculate M-ratio (simplified)
            our_stack = log.get("our_stack", 1000)
            big_blind = log.get("big_blind", 1)
            small_blind = log.get("small_blind", big_blind / 2)
            ante = log.get("ante", 0)

            m_ratio = (
                our_stack / (small_blind + big_blind + ante)
                if (small_blind + big_blind + ante) > 0
                else 20
            )

            # Determine stage
            if m_ratio > 20:
                stage = "early"
            elif m_ratio > 10:
                stage = "middle"
            else:
                stage = "late"

            stages[stage]["hands"] += 1

            # Track BB won/lost
            pot_won = log.get("pot_won", 0)
            if big_blind > 0:
                stages[stage]["bb_won"] += pot_won / big_blind

        # Calculate win rates
        for stage_name, stage_data in stages.items():
            if stage_data["hands"] > 0:
                stage_data["win_rate"] = stage_data["bb_won"] / stage_data["hands"] * 100

        return stages

    def _analyze_performance_by_position(
        self, session_logs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze performance by position: Early, Late, Blinds."""
        positions = {
            "early": {
                "hands": 0,
                "bb_won": 0,
                "win_rate": 0.0,
                "vpip": 0.0,
                "voluntary_actions": 0,
            },
            "late": {"hands": 0, "bb_won": 0, "win_rate": 0.0, "vpip": 0.0, "voluntary_actions": 0},
            "blinds": {
                "hands": 0,
                "bb_won": 0,
                "win_rate": 0.0,
                "vpip": 0.0,
                "voluntary_actions": 0,
            },
        }

        for log in session_logs:
            position = self._categorize_position(log.get("position", "unknown"))
            if position == "unknown":
                continue

            positions[position]["hands"] += 1

            # Track performance
            pot_won = log.get("pot_won", 0)
            big_blind = log.get("big_blind", 1)
            if big_blind > 0:
                positions[position]["bb_won"] += pot_won / big_blind

            # Track VPIP by position
            action = log.get("final_action", {}).get("action", "fold")
            if log.get("street", "preflop") == "preflop" and action in ["call", "raise"]:
                positions[position]["voluntary_actions"] += 1

        # Calculate statistics
        for pos_name, pos_data in positions.items():
            if pos_data["hands"] > 0:
                pos_data["win_rate"] = pos_data["bb_won"] / pos_data["hands"] * 100
                pos_data["vpip"] = pos_data["voluntary_actions"] / pos_data["hands"]

        return positions

    def _categorize_position(self, position: str) -> str:
        """Categorize position into Early, Late, or Blinds."""
        position_lower = str(position).lower()

        if any(pos in position_lower for pos in ["utg", "mp", "early"]):
            return "early"
        elif any(pos in position_lower for pos in ["co", "btn", "button", "late"]):
            return "late"
        elif any(pos in position_lower for pos in ["sb", "bb", "blind"]):
            return "blinds"
        else:
            return "unknown"

    def _analyze_performance_vs_opponent_types(
        self, session_logs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze performance against TAG, LAG, and Nit opponents."""
        opponent_types = {
            "vs_tag": {"hands": 0, "bb_won": 0, "win_rate": 0.0},
            "vs_lag": {"hands": 0, "bb_won": 0, "win_rate": 0.0},
            "vs_nit": {"hands": 0, "bb_won": 0, "win_rate": 0.0},
        }

        for log in session_logs:
            # Categorize primary opponent (simplified)
            opponents = log.get("opponent_model", {}).get("opponents", {})
            primary_opponent_type = self._categorize_primary_opponent(opponents)

            if primary_opponent_type != "unknown":
                type_key = f"vs_{primary_opponent_type}"
                if type_key in opponent_types:
                    opponent_types[type_key]["hands"] += 1

                    pot_won = log.get("pot_won", 0)
                    big_blind = log.get("big_blind", 1)
                    if big_blind > 0:
                        opponent_types[type_key]["bb_won"] += pot_won / big_blind

        # Calculate win rates
        for opp_type, data in opponent_types.items():
            if data["hands"] > 0:
                data["win_rate"] = data["bb_won"] / data["hands"] * 100

        return opponent_types

    def _categorize_primary_opponent(self, opponents: Dict[str, Any]) -> str:
        """Categorize the primary opponent as TAG, LAG, or Nit."""
        if not opponents:
            return "unknown"

        # Find the most active opponent (simplified)
        most_active = max(
            opponents.items(), key=lambda x: x[1].get("hands_observed", 0), default=("", {})
        )
        opp_stats = most_active[1].get("stats", {})

        vpip = opp_stats.get("vpip", 0.25)
        pfr = opp_stats.get("pfr", 0.15)

        if vpip < 0.18 and pfr < 0.12:
            return "nit"
        elif vpip > 0.35 and pfr > 0.25:
            return "lag"
        elif 0.18 <= vpip <= 0.28 and 0.12 <= pfr <= 0.22:
            return "tag"
        else:
            return "unknown"

    def _analyze_cognitive_path_usage(self, session_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze percentage of decisions made via Fast Path vs Slow Path."""
        path_usage = {
            "fast_path_percentage": 0.0,
            "slow_path_percentage": 0.0,
            "total_decisions": len(session_logs),
        }

        fast_path_count = 0
        slow_path_count = 0

        for log in session_logs:
            # Determine cognitive path used (simplified heuristic)
            processing_time = log.get("total_processing_time", 0.1)
            confidence_score = log.get("confidence_score", 0.5)

            # Fast path: quick decisions with high confidence
            if processing_time < 0.2 and confidence_score > 0.7:
                fast_path_count += 1
            # Slow path: longer processing or lower confidence
            else:
                slow_path_count += 1

        total_decisions = fast_path_count + slow_path_count
        if total_decisions > 0:
            path_usage["fast_path_percentage"] = fast_path_count / total_decisions * 100
            path_usage["slow_path_percentage"] = slow_path_count / total_decisions * 100

        return path_usage

    def _analyze_gto_vs_exploitation(self, session_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze percentage of time agent's action deviated from GTO recommendation."""
        gto_analysis = {
            "gto_adherence_percentage": 0.0,
            "exploitation_percentage": 0.0,
            "decisions_with_gto_data": 0,
        }

        gto_matches = 0
        total_with_gto = 0

        for log in session_logs:
            # Check if we have GTO recommendation data
            system1_inputs = log.get("system1_inputs", {})
            gto_rec = system1_inputs.get("gto", {})
            final_action = log.get("final_action", {})

            if gto_rec and final_action:
                total_with_gto += 1

                # Compare final action to GTO recommendation
                if gto_rec.get("action") == final_action.get("action"):
                    gto_matches += 1

        if total_with_gto > 0:
            gto_analysis["gto_adherence_percentage"] = gto_matches / total_with_gto * 100
            gto_analysis["exploitation_percentage"] = (
                (total_with_gto - gto_matches) / total_with_gto * 100
            )
            gto_analysis["decisions_with_gto_data"] = total_with_gto

        return gto_analysis

    def _analyze_confidence_patterns(self, session_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze agent's confidence patterns on winning vs losing hands."""
        confidence_analysis = {
            "avg_confidence_winning_hands": 0.0,
            "avg_confidence_losing_hands": 0.0,
            "confidence_accuracy": 0.0,
            "high_confidence_win_rate": 0.0,
        }

        winning_confidences = []
        losing_confidences = []
        high_confidence_hands = []

        for log in session_logs:
            confidence = log.get("confidence_score", 0.5)
            pot_won = log.get("pot_won", 0)

            if pot_won > 0:
                winning_confidences.append(confidence)
            else:
                losing_confidences.append(confidence)

            # Track high confidence hands (>0.8)
            if confidence > 0.8:
                high_confidence_hands.append(pot_won > 0)

        # Calculate averages
        if winning_confidences:
            confidence_analysis["avg_confidence_winning_hands"] = statistics.mean(
                winning_confidences
            )
        if losing_confidences:
            confidence_analysis["avg_confidence_losing_hands"] = statistics.mean(losing_confidences)

        # Calculate high confidence win rate
        if high_confidence_hands:
            confidence_analysis["high_confidence_win_rate"] = (
                sum(high_confidence_hands) / len(high_confidence_hands) * 100
            )

        # Calculate confidence accuracy (correlation between confidence and success)
        all_confidences = []
        all_outcomes = []
        for log in session_logs:
            all_confidences.append(log.get("confidence_score", 0.5))
            all_outcomes.append(1 if log.get("pot_won", 0) > 0 else 0)

        if len(all_confidences) > 1:
            # Simple correlation approximation
            confidence_analysis["confidence_accuracy"] = self._calculate_correlation(
                all_confidences, all_outcomes
            )

        return confidence_analysis

    def _calculate_correlation(self, x: List[float], y: List[int]) -> float:
        """Calculate simple correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        sum_y2 = sum(yi * yi for yi in y)

        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5

        return numerator / denominator if denominator != 0 else 0.0

    def _analyze_risk_management(self, session_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze risk management effectiveness."""
        risk_metrics = {
            "average_risk_per_decision": 0.0,
            "risk_adjusted_returns": 0.0,
            "conservative_decision_rate": 0.0,
        }

        total_risk = 0
        conservative_decisions = 0
        total_decisions = len(session_logs)

        for log in session_logs:
            # Estimate risk per decision (amount at risk / stack)
            final_action = log.get("final_action", {})
            amount = final_action.get("amount", 0)
            our_stack = log.get("our_stack", 1000)

            if our_stack > 0:
                risk_ratio = amount / our_stack
                total_risk += risk_ratio

                # Conservative decision if risk < 5% of stack or fold
                if final_action.get("action") == "fold" or risk_ratio < 0.05:
                    conservative_decisions += 1

        if total_decisions > 0:
            risk_metrics["average_risk_per_decision"] = total_risk / total_decisions
            risk_metrics["conservative_decision_rate"] = (
                conservative_decisions / total_decisions * 100
            )

        return risk_metrics

    def _analyze_adaptation_efficiency(self, session_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how efficiently the agent adapts to opponents."""
        adaptation_metrics = {
            "opponent_model_usage_rate": 0.0,
            "adaptation_improvement_trend": 0.0,
        }

        decisions_with_opponent_data = 0

        for log in session_logs:
            opponent_model = log.get("opponent_model", {})
            if opponent_model and opponent_model.get("opponents"):
                decisions_with_opponent_data += 1

        if len(session_logs) > 0:
            adaptation_metrics["opponent_model_usage_rate"] = (
                decisions_with_opponent_data / len(session_logs) * 100
            )

        return adaptation_metrics

    def _analyze_tournament_survival(
        self, session_logs: List[Dict[str, Any]], tournament_results: Optional[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Analyze tournament survival metrics."""
        survival_metrics = {
            "average_survival_time": 0.0,
            "bubble_survival_rate": 0.0,
            "late_stage_performance": 0.0,
        }

        if tournament_results:
            # Calculate average finish position (lower is better)
            finish_positions = [result.get("finish_position", 999) for result in tournament_results]
            survival_metrics["average_survival_time"] = (
                statistics.mean(finish_positions) if finish_positions else 0.0
            )

            # Calculate bubble survival (finished in money)
            bubble_survivors = sum(
                1
                for result in tournament_results
                if result.get("finish_position", 999) <= result.get("itm_positions", 0)
            )
            if tournament_results:
                survival_metrics["bubble_survival_rate"] = (
                    bubble_survivors / len(tournament_results) * 100
                )

        return survival_metrics

    def _generate_strategic_insights(
        self, session_logs: List[Dict[str, Any]], tournament_results: Optional[List[Dict[str, Any]]]
    ) -> List[str]:
        """Generate strategic insights and recommendations based on analytics."""
        insights = []

        # Analyze decision patterns
        total_hands = len(session_logs)
        if total_hands < 100:
            insights.append(
                "SAMPLE SIZE: Increase sample size to at least 100 hands for more reliable analytics."
            )

        # Analyze confidence patterns
        high_confidence_decisions = sum(
            1 for log in session_logs if log.get("confidence_score", 0.5) > 0.8
        )
        if high_confidence_decisions / total_hands < 0.2:
            insights.append(
                "CONFIDENCE: Consider increasing decision confidence through better opponent modeling and situational analysis."
            )

        # Analyze positional play
        position_data = self._analyze_performance_by_position(session_logs)
        early_vpip = position_data["early"].get("vpip", 0)
        late_vpip = position_data["late"].get("vpip", 0)

        if abs(early_vpip - late_vpip) < 0.1:
            insights.append(
                "POSITIONAL PLAY: Increase positional awareness - VPIP should vary significantly by position."
            )

        # Analyze GTO vs Exploitation balance
        gto_analysis = self._analyze_gto_vs_exploitation(session_logs)
        gto_adherence = gto_analysis.get("gto_adherence_percentage", 50)

        if gto_adherence > 90:
            insights.append(
                "STRATEGY: Consider more exploitative play - 100% GTO adherence may leave money on the table."
            )
        elif gto_adherence < 60:
            insights.append(
                "STRATEGY: Consider more GTO-based play for better balance and defense against strong opponents."
            )

        return insights[:5]  # Return top 5 insights
