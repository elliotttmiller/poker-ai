#!/usr/bin/env python3
"""
Configurable Gauntlet Runner for Project PokerMind - Final Integrity Protocol.

This script provides a configurable, automated tool for running and analyzing
tournament series with comprehensive agent-centric reporting.

Usage:
    python run_gauntlet.py --num-tournaments 10    # Shakedown run
    python run_gauntlet.py --num-tournaments 100   # Tuning run
"""

import argparse
import logging
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agent.cognitive_core import CognitiveCore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("gauntlet_run.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


class GauntletRunner:
    """
    Configurable tournament gauntlet runner with comprehensive analytics.

    Executes series of poker tournaments and generates world-class
    agent-centric performance reports.
    """

    def __init__(self):
        """Initialize the Gauntlet Runner."""
        self.logger = logging.getLogger(__name__)
        try:
            # Import PostGameAnalyzer here to avoid import issues during initialization
            from agent.toolkit.post_game_analyzer import PostGameAnalyzer

            self.analyzer = PostGameAnalyzer()
        except Exception as e:
            self.logger.warning(f"Could not import PostGameAnalyzer: {e}")
            self.analyzer = None

        # Results storage
        self.tournament_results = []
        self.session_logs = []

        # Create results directory
        self.results_dir = Path("tournament_results")
        self.results_dir.mkdir(exist_ok=True)

        self.logger.info("Gauntlet Runner initialized")

    def run_gauntlet(
        self,
        num_tournaments: int,
        tournament_type: str = "standard",
        save_results: bool = True,
        generate_report: bool = True,
        autonomous_tuning: bool = False,
        tuning_frequency: int = 25,
    ) -> Dict[str, Any]:
        """
        Run a configurable gauntlet of tournaments with optional autonomous tuning.
        
        Enhanced for Ultimate Intelligence Protocol with Play -> Analyze -> Tune -> Repeat cycle.

        Args:
            num_tournaments: Number of tournaments to run
            tournament_type: Type of tournament (standard, turbo, etc.)
            save_results: Whether to save results to file
            generate_report: Whether to generate comprehensive analytics report
            autonomous_tuning: Enable autonomous tuning loop
            tuning_frequency: Apply tuning every N tournaments

        Returns:
            Dictionary containing complete gauntlet results and analytics
        """
        start_time = time.time()
        self.logger.info(f"Starting gauntlet run: {num_tournaments} tournaments")
        
        # Import tuning components if autonomous tuning is enabled
        if autonomous_tuning:
            try:
                from apply_tuning import AutomatedTuner
                from agent.toolkit.post_game_analyzer import PostGameAnalyzer
                tuner = AutomatedTuner()
                analyzer = PostGameAnalyzer() 
                self.logger.info(f"Autonomous tuning enabled - will tune every {tuning_frequency} tournaments")
            except ImportError as e:
                self.logger.warning(f"Autonomous tuning disabled due to import error: {e}")
                autonomous_tuning = False

        try:
            # Initialize cognitive core for decision making
            self.cognitive_core = CognitiveCore()
            
            # Autonomous tuning loop variables
            tournaments_since_tuning = 0
            tuning_cycles = 0
            last_tuning_performance = 0.0

            # Run tournaments
            for i in range(num_tournaments):
                self.logger.info(f"Running tournament {i+1}/{num_tournaments}")

                tournament_result = self._run_single_tournament(
                    tournament_id=i + 1, tournament_type=tournament_type
                )

                if tournament_result:
                    self.tournament_results.append(tournament_result)

                tournaments_since_tuning += 1

                # Autonomous tuning cycle
                if autonomous_tuning and tournaments_since_tuning >= tuning_frequency:
                    self.logger.info("🔧 Starting autonomous tuning cycle...")
                    
                    try:
                        # ANALYZE: Generate tuning suggestions
                        tuning_suggestions = analyzer.generate_tuning_suggestions(
                            self.session_logs, 
                            self.tournament_results
                        )
                        
                        if tuning_suggestions.get("suggested_parameter_changes"):
                            # Save tuning suggestions
                            suggestions_path = f"tuning_suggestions_cycle_{tuning_cycles + 1}.json"
                            with open(suggestions_path, 'w') as f:
                                json.dump(tuning_suggestions, f, indent=2)
                            
                            self.logger.info(f"📊 Generated tuning suggestions: {suggestions_path}")
                            
                            # TUNE: Apply suggested changes
                            result = tuner.apply_tuning_suggestions(suggestions_path, dry_run=False)
                            
                            if result["success"]:
                                changes_applied = len(result["changes_applied"])
                                self.logger.info(f"✅ Applied {changes_applied} tuning changes successfully")
                                
                                # Log tuning cycle info
                                tuning_cycles += 1
                                self.logger.info(f"🎯 Completed autonomous tuning cycle {tuning_cycles}")
                                
                                # Calculate performance improvement (simplified)
                                current_performance = self._calculate_recent_performance()
                                if last_tuning_performance > 0:
                                    improvement = current_performance - last_tuning_performance
                                    self.logger.info(f"📈 Performance change since last tuning: {improvement:+.2%}")
                                
                                last_tuning_performance = current_performance
                                
                            else:
                                self.logger.warning(f"⚠️ Autonomous tuning failed: {result.get('error')}")
                        else:
                            self.logger.info("🔄 No tuning changes suggested - continuing with current configuration")
                    
                    except Exception as e:
                        self.logger.error(f"❌ Autonomous tuning cycle failed: {e}")
                    
                    # Reset counter
                    tournaments_since_tuning = 0
                    
                    # Brief pause to let changes take effect
                    time.sleep(1)

                # Progress logging
                if (i + 1) % max(1, num_tournaments // 10) == 0:
                    progress = (i + 1) / num_tournaments * 100
                    self.logger.info(f"Progress: {progress:.1f}% ({i+1}/{num_tournaments})")

            # Generate comprehensive analytics
            gauntlet_results = {
                "gauntlet_config": {
                    "num_tournaments": num_tournaments,
                    "tournament_type": tournament_type,
                    "start_time": datetime.fromtimestamp(start_time).isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "total_runtime": time.time() - start_time,
                    "autonomous_tuning": autonomous_tuning,
                    "tuning_frequency": tuning_frequency if autonomous_tuning else None,
                    "tuning_cycles_completed": tuning_cycles if autonomous_tuning else 0,
                },
                "tournament_results": self.tournament_results,
                "session_logs": self.session_logs,
                "summary_statistics": self._calculate_summary_statistics(),
            }

            # Generate full analytics report if requested
            if generate_report and self.analyzer:
                self.logger.info("Generating comprehensive analytics report...")
                full_report = self.analyzer.generate_full_report(
                    session_logs=self.session_logs, tournament_results=self.tournament_results
                )
                gauntlet_results["full_analytics_report"] = full_report
            elif generate_report:
                self.logger.warning("Analytics report requested but PostGameAnalyzer not available")

            # Generate final tuning suggestions for future use
            if autonomous_tuning and self.session_logs:
                try:
                    final_suggestions = analyzer.generate_tuning_suggestions(
                        self.session_logs, 
                        self.tournament_results
                    )
                    
                    with open("final_tuning_suggestions.json", 'w') as f:
                        json.dump(final_suggestions, f, indent=2)
                    
                    gauntlet_results["final_tuning_suggestions"] = final_suggestions
                    self.logger.info("📝 Generated final tuning suggestions for future use")
                    
                except Exception as e:
                    self.logger.error(f"Failed to generate final tuning suggestions: {e}")

            # Save results if requested
            if save_results:
                self._save_results(gauntlet_results, num_tournaments)

            # Print summary
            self._print_summary(gauntlet_results, autonomous_tuning, tuning_cycles)

            return gauntlet_results

        except Exception as e:
            self.logger.error(f"Gauntlet run failed: {e}")
            raise
    
    def _calculate_recent_performance(self) -> float:
        """
        Calculate recent performance for autonomous tuning feedback.
        
        Returns:
            Performance metric (ROI or win rate)
        """
        if not self.tournament_results:
            return 0.0
        
        # Look at last 10 tournaments or all if fewer
        recent_results = self.tournament_results[-10:]
        
        total_winnings = sum(result.get("winnings", 0) for result in recent_results)
        total_buyins = sum(result.get("buyin", 100) for result in recent_results)
        
        if total_buyins > 0:
            roi = (total_winnings - total_buyins) / total_buyins
            return roi
        
        return 0.0

    def _run_single_tournament(self, tournament_id: int, tournament_type: str) -> Dict[str, Any]:
        """
        Run a single tournament and collect results.

        Args:
            tournament_id: Unique identifier for the tournament
            tournament_type: Type of tournament to run

        Returns:
            Tournament result dictionary
        """
        try:
            # Simulate tournament configuration
            tournament_config = self._get_tournament_config(tournament_type)

            # For this implementation, we'll simulate a tournament
            # In a real implementation, this would interface with a poker engine
            tournament_result = self._simulate_tournament(tournament_id, tournament_config)

            return tournament_result

        except Exception as e:
            self.logger.error(f"Tournament {tournament_id} failed: {e}")
            return None

    def _get_tournament_config(self, tournament_type: str) -> Dict[str, Any]:
        """Get configuration for tournament type."""
        configs = {
            "standard": {
                "buyin": 100,
                "starting_stack": 10000,
                "blind_levels": 20,  # minutes
                "players": 180,
                "structure": "standard",
            },
            "turbo": {
                "buyin": 50,
                "starting_stack": 8000,
                "blind_levels": 12,  # minutes
                "players": 180,
                "structure": "turbo",
            },
        }
        return configs.get(tournament_type, configs["standard"])

    def _simulate_tournament(self, tournament_id: int, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate a tournament for testing purposes.

        In a production environment, this would be replaced with actual
        tournament play using a poker engine.
        """
        import random

        # Simulate tournament outcome
        total_players = config["players"]
        buyin = config["buyin"]

        # Simulate our finish position (weighted toward reasonable performance)
        finish_positions = list(range(1, total_players + 1))
        weights = [
            1 / pos**0.3 for pos in finish_positions
        ]  # Better players finish higher more often
        finish_position = random.choices(finish_positions, weights=weights)[0]

        # Calculate ITM positions (typically top 15-20%)
        itm_positions = max(1, total_players // 6)

        # Calculate winnings based on finish position
        if finish_position <= itm_positions:
            # Simplified payout structure
            if finish_position == 1:
                winnings = buyin * total_players * 0.3  # 30% to winner
            elif finish_position <= 3:
                winnings = buyin * total_players * 0.15  # 15% to 2nd-3rd
            elif finish_position <= 9:
                winnings = buyin * total_players * 0.05  # 5% to others in final table
            else:
                winnings = buyin * 1.5  # Min cash
        else:
            winnings = 0

        # Generate some decision logs
        num_hands = random.randint(50, 200)  # Simulate varying tournament length
        for hand_id in range(num_hands):
            decision_log = self._generate_mock_decision_log(tournament_id, hand_id)
            self.session_logs.append(decision_log)

        return {
            "tournament_id": tournament_id,
            "buyin": buyin,
            "starting_stack": config["starting_stack"],
            "finish_position": finish_position,
            "total_players": total_players,
            "itm_positions": itm_positions,
            "winnings": winnings,
            "hands_played": num_hands,
            "tournament_type": config["structure"],
            "timestamp": datetime.now().isoformat(),
        }

    def _generate_mock_decision_log(self, tournament_id: int, hand_id: int) -> Dict[str, Any]:
        """Generate a mock decision log for testing purposes."""
        import random

        streets = ["preflop", "flop", "turn", "river"]
        actions = ["fold", "call", "raise", "bet"]
        positions = ["UTG", "MP", "CO", "BTN", "SB", "BB"]

        # Generate realistic decision packet
        street = random.choice(streets)
        action = random.choice(actions)
        position = random.choice(positions)

        # Simulate some basic game state
        big_blind = random.choice([25, 50, 100, 200, 400])
        pot_size = random.randint(big_blind, big_blind * 10)
        our_stack = random.randint(big_blind * 5, big_blind * 50)

        # Simulate decision confidence (better decisions have higher confidence)
        base_confidence = random.uniform(0.4, 0.9)

        # Simulate processing time (faster for easier decisions)
        processing_time = random.uniform(0.05, 0.5)

        # Simulate pot won (action-dependent probability)
        win_probability = {"fold": 0.0, "call": 0.3, "raise": 0.5, "bet": 0.6}.get(action, 0.3)
        pot_won = pot_size if random.random() < win_probability else 0

        return {
            "tournament_id": tournament_id,
            "hand_id": hand_id,
            "timestamp": datetime.now().isoformat(),
            "street": street,
            "position": position,
            "pot_size": pot_size,
            "big_blind": big_blind,
            "small_blind": big_blind // 2,
            "our_stack": our_stack,
            "final_action": {
                "action": action,
                "amount": random.randint(big_blind, pot_size) if action in ["raise", "bet"] else 0,
            },
            "confidence_score": base_confidence,
            "total_processing_time": processing_time,
            "pot_won": pot_won,
            "hole_cards": ["As", "Ks"],  # Simplified
            "community_cards": [],  # Simplified
            "system1_inputs": {
                "gto": {"action": action, "confidence": base_confidence},
                "heuristics": {"recommendation": action, "confidence": base_confidence * 0.8},
            },
            "opponent_model": {
                "opponents": {
                    f"opponent_{random.randint(1,5)}": {
                        "stats": {
                            "vpip": random.uniform(0.15, 0.45),
                            "pfr": random.uniform(0.10, 0.30),
                        },
                        "hands_observed": random.randint(10, 100),
                    }
                }
            },
        }

    def _calculate_summary_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics for the gauntlet run."""
        if not self.tournament_results:
            return {}

        total_buyins = sum(result["buyin"] for result in self.tournament_results)
        total_winnings = sum(result["winnings"] for result in self.tournament_results)
        itm_count = sum(
            1
            for result in self.tournament_results
            if result["finish_position"] <= result["itm_positions"]
        )

        return {
            "total_tournaments": len(self.tournament_results),
            "total_hands": len(self.session_logs),
            "total_buyins": total_buyins,
            "total_winnings": total_winnings,
            "net_profit": total_winnings - total_buyins,
            "roi_percentage": (
                ((total_winnings - total_buyins) / total_buyins * 100) if total_buyins > 0 else 0.0
            ),
            "itm_count": itm_count,
            "itm_percentage": (
                (itm_count / len(self.tournament_results) * 100) if self.tournament_results else 0.0
            ),
            "average_finish": sum(result["finish_position"] for result in self.tournament_results)
            / len(self.tournament_results),
        }

    def _save_results(self, results: Dict[str, Any], num_tournaments: int) -> None:
        """Save gauntlet results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gauntlet_run_{num_tournaments}t_{timestamp}.json"
        filepath = self.results_dir / filename

        try:
            with open(filepath, "w") as f:
                json.dump(results, f, indent=2, default=str)
            self.logger.info(f"Results saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")

    def _print_summary(self, results: Dict[str, Any], autonomous_tuning: bool = False, tuning_cycles: int = 0) -> None:
        """Print summary of gauntlet results to console."""
        config = results["gauntlet_config"]
        summary = results["summary_statistics"]

        print("\n" + "=" * 60)
        print("POKERMIND GAUNTLET RUN SUMMARY")
        print("=" * 60)
        print(f"Tournaments: {config['num_tournaments']}")
        print(f"Runtime: {config['total_runtime']:.1f} seconds")
        print(f"Total Hands: {summary.get('total_hands', 0)}")
        
        # Autonomous tuning summary
        if autonomous_tuning:
            print("-" * 40)
            print("AUTONOMOUS TUNING:")
            print(f"🔧 Tuning Cycles: {tuning_cycles}")
            print(f"🎯 Tuning Frequency: Every {config.get('tuning_frequency', 'N/A')} tournaments")
            if tuning_cycles > 0:
                print("✅ Agent evolved autonomously during this run")
            else:
                print("🔄 No tuning cycles completed")
        
        print("-" * 40)
        print(f"Total Buyins: ${summary.get('total_buyins', 0):,.2f}")
        print(f"Total Winnings: ${summary.get('total_winnings', 0):,.2f}")
        print(f"Net Profit: ${summary.get('net_profit', 0):,.2f}")
        print(f"ROI: {summary.get('roi_percentage', 0):.2f}%")
        print(f"ITM Rate: {summary.get('itm_percentage', 0):.1f}%")
        print(f"Average Finish: {summary.get('average_finish', 0):.1f}")
        print("=" * 60)

        # Print insights if available
        full_report = results.get("full_analytics_report", {})
        insights = full_report.get("insights_and_recommendations", [])
        if insights:
            print("TOP INSIGHTS:")
            for i, insight in enumerate(insights[:3], 1):
                print(f"{i}. {insight}")
        
        # Print final tuning suggestions if available
        final_suggestions = results.get("final_tuning_suggestions", {})
        if final_suggestions.get("suggested_parameter_changes"):
            print("\nFINAL TUNING SUGGESTIONS AVAILABLE:")
            print("📊 Check final_tuning_suggestions.json for future improvements")
        
        print()


def main():
    """Main entry point for the gauntlet runner."""
    parser = argparse.ArgumentParser(
        description="Run configurable poker tournament gauntlet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_gauntlet.py --num-tournaments 10              # Quick shakedown
  python run_gauntlet.py --num-tournaments 100             # Full tuning run
  python run_gauntlet.py --num-tournaments 50 --turbo      # Turbo tournaments
  python run_gauntlet.py --num-tournaments 25 --no-report  # Skip full analytics
        """,
    )

    parser.add_argument(
        "--num-tournaments", type=int, required=True, help="Number of tournaments to run"
    )

    parser.add_argument(
        "--tournament-type",
        choices=["standard", "turbo"],
        default="standard",
        help="Type of tournament to run (default: standard)",
    )

    parser.add_argument(
        "--turbo",
        action="store_const",
        const="turbo",
        dest="tournament_type",
        help="Shortcut for --tournament-type turbo",
    )

    parser.add_argument(
        "--no-save", action="store_false", dest="save_results", help="Don't save results to file"
    )

    parser.add_argument(
        "--no-report",
        action="store_false",
        dest="generate_report",
        help="Skip comprehensive analytics report generation",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # Autonomous tuning arguments
    parser.add_argument(
        "--autonomous-tuning",
        action="store_true",
        help="Enable autonomous tuning loop (Play -> Analyze -> Tune -> Repeat)",
    )
    
    parser.add_argument(
        "--tuning-frequency",
        type=int,
        default=25,
        help="Apply tuning every N tournaments (default: 25)",
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    if args.num_tournaments < 1:
        print("Error: Number of tournaments must be at least 1")
        sys.exit(1)

    if args.num_tournaments > 1000:
        print("Warning: Running more than 1000 tournaments may take a very long time")
        response = input("Continue? (y/N): ")
        if response.lower() != "y":
            sys.exit(0)
    
    # Validate autonomous tuning settings
    if args.autonomous_tuning and args.tuning_frequency > args.num_tournaments:
        print("Warning: Tuning frequency is greater than total tournaments - no tuning will occur")
        response = input("Continue? (y/N): ")
        if response.lower() != "y":
            sys.exit(0)

    try:
        # Run the gauntlet
        runner = GauntletRunner()
        results = runner.run_gauntlet(
            num_tournaments=args.num_tournaments,
            tournament_type=args.tournament_type,
            save_results=args.save_results,
            generate_report=args.generate_report,
            autonomous_tuning=args.autonomous_tuning,
            tuning_frequency=args.tuning_frequency,
        )

        logger.info("Gauntlet run completed successfully")

    except KeyboardInterrupt:
        logger.info("Gauntlet run interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Gauntlet run failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
