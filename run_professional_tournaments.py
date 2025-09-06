#!/usr/bin/env python3
"""
Professional Tournament Simulation with Elite Opponents

This script creates tournaments using the professional opponent archetypes:
- The TAG (Tight-Aggressive)
- The LAG (Loose-Aggressive) 
- The Nit (The Rock)
- PokerMind Agent (when available)
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from run_tournament import TournamentDirector, TournamentResult
from agent.opponents.The_TAG import TightAggressivePlayer
from agent.opponents.The_LAG import LooseAggressivePlayer
from agent.opponents.The_Nit import NitPlayer


def create_professional_tournament(tournament_id: str = None, 
                                 initial_stack: int = 1500,
                                 include_pokermind: bool = True) -> TournamentDirector:
    """Create a tournament with professional opponent archetypes."""
    
    if not tournament_id:
        tournament_id = f"pro_tournament_{int(__import__('time').time())}"
        
    tournament = TournamentDirector(tournament_id, initial_stack)
    
    # Register professional opponents
    tag_agent = TightAggressivePlayer("TAG_Pro")
    tournament.register_player("TAG_Pro", lambda: tag_agent)
    
    lag_agent = LooseAggressivePlayer("LAG_Pro") 
    tournament.register_player("LAG_Pro", lambda: lag_agent)
    
    nit_agent = NitPlayer("Nit_Pro")
    tournament.register_player("Nit_Pro", lambda: nit_agent)
    
    # Add additional opponents for 6-player tournament
    tag_agent2 = TightAggressivePlayer("TAG_2")
    tournament.register_player("TAG_2", lambda: tag_agent2)
    
    lag_agent2 = LooseAggressivePlayer("LAG_2")
    tournament.register_player("LAG_2", lambda: lag_agent2)
    
    if include_pokermind:
        # Try to include PokerMind agent if available
        try:
            from agent.agent import PokerMindAgent
            pokermind = PokerMindAgent()
            tournament.register_player("PokerMind", lambda: pokermind)
        except Exception as e:
            print(f"⚠️ Could not load PokerMind agent: {e}")
            # Add another TAG instead
            tag_agent3 = TightAggressivePlayer("TAG_3")
            tournament.register_player("TAG_3", lambda: tag_agent3)
    else:
        # Add another nit
        nit_agent2 = NitPlayer("Nit_2")
        tournament.register_player("Nit_2", lambda: nit_agent2)
    
    return tournament


def run_professional_gauntlet(num_tournaments: int = 10, 
                            save_results: bool = True,
                            initial_stack: int = 1500) -> List[TournamentResult]:
    """Run multiple tournaments for data collection."""
    
    print(f"🎯 Starting Professional Tournament Gauntlet")
    print(f"Running {num_tournaments} tournaments with {initial_stack} starting stacks")
    print("="*60)
    
    results = []
    
    for i in range(num_tournaments):
        tournament_id = f"gauntlet_{i+1:03d}"
        print(f"\n🏆 Tournament {i+1}/{num_tournaments}: {tournament_id}")
        
        tournament = create_professional_tournament(tournament_id, initial_stack)
        
        try:
            result = tournament.run_tournament()
            results.append(result)
            
            print(f"✅ Winner: {result.winner} ({result.total_hands} hands)")
            
            if save_results:
                tournament.save_results()
                
        except Exception as e:
            print(f"❌ Tournament {tournament_id} failed: {e}")
            continue
    
    # Summary statistics
    if results:
        print(f"\n{'='*60}")
        print(f"🏁 GAUNTLET RESULTS SUMMARY")
        print(f"{'='*60}")
        
        winner_counts = {}
        total_hands = 0
        total_duration = 0
        
        for result in results:
            winner = result.winner
            winner_counts[winner] = winner_counts.get(winner, 0) + 1
            total_hands += result.total_hands
            total_duration += result.duration_seconds
        
        print(f"Tournaments Completed: {len(results)}")
        print(f"Average Hands per Tournament: {total_hands / len(results):.1f}")
        print(f"Average Duration: {total_duration / len(results):.2f} seconds")
        
        print(f"\n🏆 WINNER STATISTICS:")
        for winner, count in sorted(winner_counts.items(), key=lambda x: x[1], reverse=True):
            win_rate = (count / len(results)) * 100
            print(f"  {winner}: {count} wins ({win_rate:.1f}%)")
    
    return results


def analyze_opponent_performance(results: List[TournamentResult]) -> Dict[str, Any]:
    """Analyze performance statistics for each opponent type."""
    
    print(f"\n📊 OPPONENT PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Collect finish data
    finish_data = {}
    
    for result in results:
        for standing in result.final_standings:
            name = standing["name"]
            place = standing["place"]
            
            if name not in finish_data:
                finish_data[name] = {"finishes": [], "wins": 0, "itm": 0}
            
            finish_data[name]["finishes"].append(place)
            
            if place == 1:
                finish_data[name]["wins"] += 1
            if place <= 3:  # Top 3 = In The Money
                finish_data[name]["itm"] += 1
    
    # Calculate statistics
    analysis = {}
    
    for name, data in finish_data.items():
        if data["finishes"]:
            avg_finish = sum(data["finishes"]) / len(data["finishes"])
            win_rate = (data["wins"] / len(data["finishes"])) * 100
            itm_rate = (data["itm"] / len(data["finishes"])) * 100
            
            analysis[name] = {
                "tournaments": len(data["finishes"]),
                "avg_finish": avg_finish,
                "win_rate": win_rate,
                "itm_rate": itm_rate,
                "wins": data["wins"]
            }
    
    # Display analysis
    for name, stats in sorted(analysis.items(), key=lambda x: x[1]["avg_finish"]):
        print(f"{name}:")
        print(f"  Average Finish: {stats['avg_finish']:.2f}")
        print(f"  Win Rate: {stats['win_rate']:.1f}%")
        print(f"  ITM Rate: {stats['itm_rate']:.1f}%")
        print(f"  Total Wins: {stats['wins']}")
        print()
    
    return analysis


def main():
    """Main function for professional tournament simulation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run professional poker tournaments with elite opponents")
    parser.add_argument("--tournaments", "-n", type=int, default=5, 
                       help="Number of tournaments to run (default: 5)")
    parser.add_argument("--stack", type=int, default=1500,
                       help="Initial stack size (default: 1500)")
    parser.add_argument("--no-save", action="store_true", 
                       help="Don't save individual tournament results")
    parser.add_argument("--single", action="store_true",
                       help="Run a single tournament")
    parser.add_argument("--analysis", action="store_true",
                       help="Run analysis on existing results")
    
    args = parser.parse_args()
    
    if args.single:
        # Run single tournament
        print("🎯 Running Single Professional Tournament")
        tournament = create_professional_tournament("single_pro", args.stack)
        result = tournament.run_tournament()
        
        print(f"\n🏆 Tournament Results:")
        print(f"Winner: {result.winner}")
        print(f"Total Hands: {result.total_hands}")
        print(f"Duration: {result.duration_seconds:.2f} seconds")
        
        if not args.no_save:
            tournament.save_results()
            print(f"Results saved!")
            
    else:
        # Run tournament gauntlet
        results = run_professional_gauntlet(
            num_tournaments=args.tournaments,
            save_results=not args.no_save,
            initial_stack=args.stack
        )
        
        if results:
            analyze_opponent_performance(results)
            
            # Save summary
            if not args.no_save:
                import json
                summary_file = Path("tournament_results") / "gauntlet_summary.json"
                summary_data = {
                    "tournaments_run": len(results),
                    "results": [
                        {
                            "tournament_id": r.tournament_id,
                            "winner": r.winner,
                            "total_hands": r.total_hands,
                            "final_standings": r.final_standings
                        }
                        for r in results
                    ]
                }
                
                Path("tournament_results").mkdir(exist_ok=True)
                with open(summary_file, 'w') as f:
                    json.dump(summary_data, f, indent=2)
                
                print(f"📁 Gauntlet summary saved to: {summary_file}")


if __name__ == "__main__":
    main()