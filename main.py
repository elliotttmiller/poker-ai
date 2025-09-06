#!/usr/bin/env python3
"""
Main entry point for Project PokerMind.

This script sets up and runs a poker simulation using PyPokerEngine
with the PokerMind AI agent.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pypokerengine.api.game import setup_config, start_poker
from agent.agent import PokerMindAgent


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pokermind.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def create_game_config():
    """Create the poker game configuration."""
    config = setup_config(
        max_round=10,
        initial_stack=1000,
        small_blind_amount=10
    )
    
    # Register PokerMind agent
    pokermind_agent = PokerMindAgent()
    config.register_player(name="PokerMind", algorithm=pokermind_agent)
    
    # Register a simple opponent for testing
    from pypokerengine.players import FishPlayer
    config.register_player(name="Fish", algorithm=FishPlayer())
    
    return config


def main():
    """Main function to run the poker simulation."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Project PokerMind simulation...")
    
    try:
        # Create and run the game
        config = create_game_config()
        game_result = start_poker(config, verbose=1)
        
        logger.info("Game completed successfully!")
        logger.info(f"Game result: {game_result}")
        
    except Exception as e:
        logger.error(f"Error running simulation: {e}")
        raise


if __name__ == "__main__":
    main()