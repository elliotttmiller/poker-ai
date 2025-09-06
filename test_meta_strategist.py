#!/usr/bin/env python3
"""
Test script for the enhanced GTOCore meta-strategist.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.modules.gto_core import GTOCore

def test_meta_strategist():
    """Test the meta-strategist functionality."""
    print("Testing Enhanced GTOCore Meta-Strategist")
    print("=" * 50)
    
    # Initialize the meta-strategist
    gto_core = GTOCore()
    
    # Check specialist loading status
    status = gto_core.get_specialist_status()
    print(f"Loaded specialists: {status['loaded_specialists']}")
    print(f"Total specialists: {status['total_specialists']}")
    print(f"Is loaded: {status['is_loaded']}")
    print()
    
    # Test preflop scenario
    preflop_state = {
        "street": "preflop",
        "hole_cards": ["As", "Kd"],
        "community_cards": [],
        "pot_size": 10,
        "our_stack": 1000,
        "valid_actions": [
            {"action": "fold", "amount": 0},
            {"action": "call", "amount": 5},
            {"action": "raise", "amount": {"min": 15, "max": 1000}}
        ]
    }
    
    print("Preflop scenario test:")
    preflop_rec = gto_core.get_recommendation(preflop_state)
    print(f"  Action: {preflop_rec['action']}")
    print(f"  Amount: {preflop_rec['amount']}")
    print(f"  Confidence: {preflop_rec['confidence']:.3f}")
    print(f"  Specialist: {preflop_rec.get('specialist_used', 'none')}")
    print()
    
    # Test river scenario
    river_state = {
        "street": "river",
        "hole_cards": ["As", "Kd"],
        "community_cards": ["Ah", "Kh", "7c", "2s", "9d"],
        "pot_size": 100,
        "our_stack": 800,
        "valid_actions": [
            {"action": "fold", "amount": 0},
            {"action": "call", "amount": 50},
            {"action": "raise", "amount": {"min": 150, "max": 800}}
        ]
    }
    
    print("River scenario test:")
    river_rec = gto_core.get_recommendation(river_state)
    print(f"  Action: {river_rec['action']}")
    print(f"  Amount: {river_rec['amount']}")
    print(f"  Confidence: {river_rec['confidence']:.3f}")
    print(f"  Specialist: {river_rec.get('specialist_used', 'none')}")
    print()
    
    print("Meta-strategist test completed successfully!")

if __name__ == "__main__":
    test_meta_strategist()