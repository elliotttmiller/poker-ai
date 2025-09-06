#!/usr/bin/env python3
"""
Tournament Director for Project PokerMind - Progressive 6-Player Tournament Simulation

This script implements a realistic, progressive, 6-player tournament environment
with dynamic blind levels, elimination tracking, and comprehensive logging.
"""

import sys
import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


@dataclass
class TournamentPlayer:
    """Represents a player in the tournament."""
    name: str
    agent_class: Any  # The actual agent class
    agent_instance: Any  # Instantiated agent
    stack: int
    seat_id: int
    eliminated_hand: Optional[int] = None
    finishing_place: Optional[int] = None


@dataclass
class BlindLevel:
    """Represents a blind level in the tournament structure."""
    hand_start: int
    small_blind: int
    big_blind: int
    ante: int = 0


@dataclass
class TournamentResult:
    """Contains the complete results of a tournament."""
    tournament_id: str
    start_time: str
    end_time: str
    total_hands: int
    final_standings: List[Dict[str, Any]]
    blind_levels_reached: int
    winner: str
    duration_seconds: float


class TournamentDirector:
    """
    The main tournament director that manages the progressive tournament simulation.
    
    Handles:
    - Player management and elimination
    - Blind level progression  
    - Hand-by-hand simulation
    - Result tracking and logging
    """

    def __init__(self, tournament_id: Optional[str] = None, initial_stack: int = 1500):
        """Initialize the tournament director."""
        self.tournament_id = tournament_id or f"tournament_{int(time.time())}"
        self.initial_stack = initial_stack
        self.logger = self._setup_logging()
        
        # Tournament state
        self.players: List[TournamentPlayer] = []
        self.eliminated_players: List[TournamentPlayer] = []
        self.hand_count = 0
        self.current_blind_level_index = 0
        
        # Blind schedule - standard tournament structure
        self.blind_schedule = [
            BlindLevel(1, 10, 20, 0),      # Level 1: 10/20
            BlindLevel(7, 15, 30, 0),      # Level 2: 15/30  
            BlindLevel(13, 25, 50, 0),     # Level 3: 25/50
            BlindLevel(19, 50, 100, 0),    # Level 4: 50/100
            BlindLevel(25, 75, 150, 0),    # Level 5: 75/150
            BlindLevel(31, 100, 200, 25),  # Level 6: 100/200 with 25 ante
            BlindLevel(37, 150, 300, 25),  # Level 7: 150/300 with 25 ante
            BlindLevel(43, 200, 400, 50),  # Level 8: 200/400 with 50 ante
            BlindLevel(49, 300, 600, 75),  # Level 9: 300/600 with 75 ante
            BlindLevel(55, 500, 1000, 100), # Level 10: 500/1000 with 100 ante
        ]
        
        self.tournament_result: Optional[TournamentResult] = None
        
    def _setup_logging(self) -> logging.Logger:
        """Set up tournament-specific logging."""
        logger = logging.getLogger(f"tournament_{self.tournament_id}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Create file handler
            log_file = f"logs/tournament_{self.tournament_id}.log"
            Path("logs").mkdir(exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            
        return logger

    def register_player(self, name: str, agent_class: Any) -> None:
        """Register a player for the tournament."""
        seat_id = len(self.players) + 1
        try:
            agent_instance = agent_class() if callable(agent_class) else agent_class
            player = TournamentPlayer(
                name=name,
                agent_class=agent_class,
                agent_instance=agent_instance,
                stack=self.initial_stack,
                seat_id=seat_id
            )
            self.players.append(player)
            self.logger.info(f"Registered player {name} in seat {seat_id}")
        except Exception as e:
            self.logger.error(f"Failed to register player {name}: {e}")
            raise

    def get_current_blinds(self) -> BlindLevel:
        """Get the current blind level based on hand count."""
        # Find the appropriate blind level
        current_level = self.blind_schedule[0]  # Default to first level
        
        for i, blind_level in enumerate(self.blind_schedule):
            if self.hand_count >= blind_level.hand_start:
                current_level = blind_level
                self.current_blind_level_index = i
            else:
                break
                
        return current_level

    def eliminate_player(self, player: TournamentPlayer) -> None:
        """Eliminate a player from the tournament."""
        if player in self.players:
            player.eliminated_hand = self.hand_count
            player.finishing_place = len(self.players)  # Current number of remaining players
            
            self.players.remove(player)
            self.eliminated_players.append(player)
            
            self.logger.info(
                f"Player {player.name} eliminated in hand {self.hand_count}, "
                f"finishing in place {player.finishing_place}"
            )

    def create_mock_game_state(self) -> Dict[str, Any]:
        """Create a mock game state for simulation when PyPokerEngine is not available."""
        current_blinds = self.get_current_blinds()
        
        return {
            "tournament_id": self.tournament_id,
            "hand_count": self.hand_count,
            "blind_level": {
                "small_blind": current_blinds.small_blind,
                "big_blind": current_blinds.big_blind,
                "ante": current_blinds.ante
            },
            "players": [
                {
                    "name": p.name,
                    "seat_id": p.seat_id,
                    "stack": p.stack,
                    "status": "active"
                }
                for p in self.players
            ],
            "pot_size": 0,
            "community_cards": [],
            "street": "preflop"
        }

    def simulate_hand_mock(self) -> List[Dict[str, Any]]:
        """Simulate a single hand with mock logic when PyPokerEngine is not available."""
        if len(self.players) < 2:
            return []
            
        # Mock hand simulation - randomly determine stack changes
        import random
        
        current_blinds = self.get_current_blinds()
        total_pot = current_blinds.small_blind + current_blinds.big_blind
        
        # Add antes if applicable
        if current_blinds.ante > 0:
            total_pot += current_blinds.ante * len(self.players)
        
        # Simple mock: redistribute some chips randomly
        results = []
        stack_changes = {}
        
        # Randomly select a winner (weighted by current stack)
        total_chips = sum(p.stack for p in self.players)
        if total_chips == 0:
            return results
            
        weights = [p.stack / total_chips for p in self.players]
        winner_idx = random.choices(range(len(self.players)), weights=weights)[0]
        winner = self.players[winner_idx]
        
        # Determine pot size (between blinds and 20% of average stack)
        avg_stack = total_chips / len(self.players)
        max_pot = min(total_pot * 3, int(avg_stack * 0.2))
        pot_size = random.randint(total_pot, max(total_pot, max_pot))
        
        # Distribute losses among other players
        remaining_players = [p for p in self.players if p != winner]
        for player in remaining_players:
            # Each player loses a small random amount
            max_loss = max(current_blinds.small_blind, min(player.stack // 10, pot_size // max(len(remaining_players), 1)))
            min_loss = min(current_blinds.small_blind, max_loss)
            if max_loss > min_loss:
                loss = random.randint(min_loss, max_loss)
            else:
                loss = min_loss
            loss = min(loss, player.stack)  # Can't lose more than they have
            if loss > 0:  # Only record losses > 0
                stack_changes[player.name] = -loss
            
        # Winner gets the total pot
        total_losses = sum(abs(loss) for loss in stack_changes.values())
        stack_changes[winner.name] = total_losses
        
        # Apply stack changes and track eliminations
        players_to_eliminate = []
        
        for player in self.players:
            if player.name in stack_changes:
                player.stack += stack_changes[player.name]
                player.stack = max(0, player.stack)  # Can't go negative
                
                if player.stack == 0:
                    players_to_eliminate.append(player)
                    
                results.append({
                    "player": player.name,
                    "stack_change": stack_changes[player.name],
                    "final_stack": player.stack,
                    "eliminated": player.stack == 0
                })
        
        # Eliminate players with zero stacks
        for player in players_to_eliminate:
            self.eliminate_player(player)
            
        return results

    def simulate_hand_pypokerengine(self) -> List[Dict[str, Any]]:
        """Simulate a single hand using PyPokerEngine (if available)."""
        try:
            from pypokerengine.api.game import setup_config, start_poker
            
            config = setup_config(
                max_round=1,  # Play just one hand
                initial_stack=0,  # We'll set individual stacks
                small_blind_amount=self.get_current_blinds().small_blind
            )
            
            # Register all active players
            for player in self.players:
                config.register_player(
                    name=player.name,
                    algorithm=player.agent_instance
                )
                # Set their current stack
                config.set_stack(player.name, player.stack)
            
            # Run the hand
            game_result = start_poker(config, verbose=0)
            
            # Process results and update player stacks
            results = []
            players_to_eliminate = []
            
            for player in self.players:
                # Find this player's result in the game result
                old_stack = player.stack
                # Update based on game result (this would need proper parsing of PyPokerEngine results)
                new_stack = game_result.get(player.name, {}).get('stack', old_stack)
                
                player.stack = new_stack
                
                if player.stack == 0:
                    players_to_eliminate.append(player)
                    
                results.append({
                    "player": player.name,
                    "stack_change": new_stack - old_stack,
                    "final_stack": new_stack,
                    "eliminated": new_stack == 0
                })
            
            # Eliminate players with zero stacks
            for player in players_to_eliminate:
                self.eliminate_player(player)
                
            return results
            
        except ImportError:
            self.logger.warning("PyPokerEngine not available, using mock simulation")
            return self.simulate_hand_mock()
        except Exception as e:
            self.logger.error(f"Error in PyPokerEngine simulation: {e}")
            return self.simulate_hand_mock()

    def run_tournament(self) -> TournamentResult:
        """Run the complete tournament simulation."""
        if len(self.players) < 2:
            raise ValueError("Need at least 2 players to run tournament")
            
        start_time = datetime.now()
        self.logger.info(f"Starting tournament {self.tournament_id} with {len(self.players)} players")
        
        # Log initial state
        for player in self.players:
            self.logger.info(f"Player {player.name} starting with {player.stack} chips")
        
        # Main tournament loop - continue while more than 1 player remains
        while len(self.players) > 1:
            self.hand_count += 1
            current_blinds = self.get_current_blinds()
            
            self.logger.info(
                f"Hand {self.hand_count}: {len(self.players)} players remaining, "
                f"Blinds: {current_blinds.small_blind}/{current_blinds.big_blind}"
            )
            
            # Simulate the hand
            hand_results = self.simulate_hand_pypokerengine()
            
            # Log hand results
            for result in hand_results:
                self.logger.info(
                    f"  {result['player']}: {result['stack_change']:+d} chips -> "
                    f"{result['final_stack']} total"
                    + (" (ELIMINATED)" if result['eliminated'] else "")
                )
            
            # Check if we need to pause (safety mechanism)
            if self.hand_count > 200:  # Reasonable upper limit for a 6-player tournament
                self.logger.warning("Tournament exceeded 200 hands - ending early")
                break
        
        end_time = datetime.now()
        
        # Determine winner
        winner = self.players[0] if self.players else None
        if winner:
            winner.finishing_place = 1
            self.logger.info(f"Tournament won by {winner.name} with {winner.stack} chips")
        
        # Create final standings (winner + eliminated players in reverse elimination order)
        final_standings = []
        if winner:
            final_standings.append({
                "place": 1,
                "name": winner.name,
                "stack": winner.stack,
                "eliminated_hand": None
            })
        
        # Add eliminated players in order of elimination (latest eliminated = higher place)
        for player in reversed(self.eliminated_players):
            final_standings.append({
                "place": player.finishing_place,
                "name": player.name,
                "stack": 0,
                "eliminated_hand": player.eliminated_hand
            })
        
        # Create tournament result
        self.tournament_result = TournamentResult(
            tournament_id=self.tournament_id,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            total_hands=self.hand_count,
            final_standings=final_standings,
            blind_levels_reached=self.current_blind_level_index + 1,
            winner=winner.name if winner else "No winner",
            duration_seconds=(end_time - start_time).total_seconds()
        )
        
        self.logger.info(f"Tournament completed in {self.hand_count} hands")
        self.logger.info(f"Final standings: {json.dumps(final_standings, indent=2)}")
        
        return self.tournament_result

    def save_results(self, filepath: Optional[Path] = None) -> Path:
        """Save tournament results to a JSON file."""
        if not self.tournament_result:
            raise ValueError("No tournament results to save")
            
        if not filepath:
            results_dir = Path("tournament_results")
            results_dir.mkdir(exist_ok=True)
            filepath = results_dir / f"{self.tournament_id}.json"
        
        result_dict = {
            "tournament_id": self.tournament_result.tournament_id,
            "start_time": self.tournament_result.start_time,
            "end_time": self.tournament_result.end_time,
            "total_hands": self.tournament_result.total_hands,
            "final_standings": self.tournament_result.final_standings,
            "blind_levels_reached": self.tournament_result.blind_levels_reached,
            "winner": self.tournament_result.winner,
            "duration_seconds": self.tournament_result.duration_seconds,
            "initial_stack": self.initial_stack,
            "blind_schedule": [
                {
                    "hand_start": bl.hand_start,
                    "small_blind": bl.small_blind,
                    "big_blind": bl.big_blind,
                    "ante": bl.ante
                }
                for bl in self.blind_schedule
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        self.logger.info(f"Tournament results saved to {filepath}")
        return filepath


def create_example_tournament() -> TournamentDirector:
    """Create an example tournament with mock agents for testing."""
    tournament = TournamentDirector("example_tournament", initial_stack=1500)
    
    # Create simple mock agent classes for testing
    class MockAgent:
        def __init__(self, name: str, style: str = "balanced"):
            self.name = name
            self.style = style
            
        def declare_action(self, valid_actions, hole_card, round_state):
            # Very simple mock decision making
            import random
            return random.choice(["fold", "call", "raise"]), random.randint(0, 100)
    
    # Register 6 players
    tournament.register_player("Player1", lambda: MockAgent("Player1", "tight"))
    tournament.register_player("Player2", lambda: MockAgent("Player2", "aggressive"))  
    tournament.register_player("Player3", lambda: MockAgent("Player3", "loose"))
    tournament.register_player("Player4", lambda: MockAgent("Player4", "balanced"))
    tournament.register_player("Player5", lambda: MockAgent("Player5", "tight"))
    tournament.register_player("Player6", lambda: MockAgent("Player6", "aggressive"))
    
    return tournament


def main():
    """Main function to run a sample tournament."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run a poker tournament simulation")
    parser.add_argument("--tournament_id", help="Tournament ID")
    parser.add_argument("--initial_stack", type=int, default=1500, help="Initial stack size")
    parser.add_argument("--save_results", action="store_true", help="Save results to file")
    
    args = parser.parse_args()
    
    # Create and run tournament
    tournament = create_example_tournament()
    if args.tournament_id:
        tournament.tournament_id = args.tournament_id
    if args.initial_stack:
        tournament.initial_stack = args.initial_stack
        
    print(f"🎯 Starting tournament {tournament.tournament_id}")
    
    try:
        result = tournament.run_tournament()
        
        print(f"\n🏆 Tournament Results:")
        print(f"Winner: {result.winner}")
        print(f"Total Hands: {result.total_hands}")
        print(f"Duration: {result.duration_seconds:.1f} seconds")
        print(f"Blind Levels Reached: {result.blind_levels_reached}")
        
        print(f"\n📊 Final Standings:")
        for standing in result.final_standings:
            place_emoji = "🥇" if standing["place"] == 1 else "🥈" if standing["place"] == 2 else "🥉" if standing["place"] == 3 else "📍"
            print(f"{place_emoji} {standing['place']}. {standing['name']} - {standing['stack']} chips")
        
        if args.save_results:
            filepath = tournament.save_results()
            print(f"Results saved to: {filepath}")
            
    except Exception as e:
        print(f"❌ Tournament failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())