"""
Integration test for the cognitive core and decision-making pipeline.
"""

import sys
import os
import logging

# Add the parent directory to the path so we can import the agent module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent.cognitive_core import CognitiveCore


def test_decision_pipeline():
    """Test that the entire decision pipeline works without crashing."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    print("Testing the cognitive core decision pipeline...")
    
    # Initialize the cognitive core
    cognitive_core = CognitiveCore()
    print("✓ Cognitive Core initialized successfully")
    
    # Create a sample game state
    sample_game_state = {
        'hole_cards': ['Ah', 'Kd'],
        'community_cards': ['Qc', 'Js', 'Th'],
        'pot_size': 100,
        'our_stack': 1000,
        'street': 'flop',
        'round_count': 1,
        'valid_actions': [
            {'action': 'fold', 'amount': 0},
            {'action': 'call', 'amount': 50},
            {'action': 'raise', 'amount': {'min': 100, 'max': 1000}}
        ]
    }
    
    # Make a decision
    try:
        final_action, decision_packet = cognitive_core.make_decision(sample_game_state)
        
        print("✓ Decision made successfully")
        print(f"  Action: {final_action['action']}")
        print(f"  Amount: {final_action.get('amount', 0)}")
        print(f"  Confidence: {decision_packet.confidence_score:.2f}")
        print(f"  Reasoning: {decision_packet.reasoning_summary}")
        print(f"  Processing time: {decision_packet.total_processing_time:.3f}s")
        
        # Verify the action is valid
        valid_action_types = [a['action'] for a in sample_game_state['valid_actions']]
        assert final_action['action'] in valid_action_types, f"Invalid action: {final_action['action']}"
        print("✓ Action is valid")
        
        # Verify decision packet is populated
        assert decision_packet.hole_cards == sample_game_state['hole_cards']
        assert decision_packet.community_cards == sample_game_state['community_cards']
        assert decision_packet.pot_size == sample_game_state['pot_size']
        print("✓ Decision packet properly populated")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during decision making: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_scenarios():
    """Test multiple decision scenarios."""
    
    cognitive_core = CognitiveCore()
    
    scenarios = [
        {
            'name': 'Strong preflop hand',
            'game_state': {
                'hole_cards': ['Ah', 'Ad'],  # Pocket Aces
                'community_cards': [],
                'pot_size': 30,
                'our_stack': 1000,
                'street': 'preflop',
                'round_count': 1,
                'valid_actions': [
                    {'action': 'fold', 'amount': 0},
                    {'action': 'call', 'amount': 20},
                    {'action': 'raise', 'amount': {'min': 40, 'max': 1000}}
                ]
            }
        },
        {
            'name': 'Weak hand facing bet',
            'game_state': {
                'hole_cards': ['2h', '7c'],  # Very weak hand
                'community_cards': ['Ac', 'Kd', 'Qh'],
                'pot_size': 200,
                'our_stack': 800,
                'street': 'flop',
                'round_count': 1,
                'valid_actions': [
                    {'action': 'fold', 'amount': 0},
                    {'action': 'call', 'amount': 150},  # Expensive call
                ]
            }
        },
        {
            'name': 'Medium strength hand',
            'game_state': {
                'hole_cards': ['Jh', 'Ts'],
                'community_cards': ['9c', '8d', '2h'],
                'pot_size': 100,
                'our_stack': 900,
                'street': 'flop',
                'round_count': 1,
                'valid_actions': [
                    {'action': 'fold', 'amount': 0},
                    {'action': 'call', 'amount': 50},
                    {'action': 'raise', 'amount': {'min': 100, 'max': 900}}
                ]
            }
        }
    ]
    
    results = []
    for scenario in scenarios:
        print(f"\nTesting scenario: {scenario['name']}")
        print(f"  Hole cards: {scenario['game_state']['hole_cards']}")
        print(f"  Community: {scenario['game_state']['community_cards']}")
        print(f"  Pot size: {scenario['game_state']['pot_size']}")
        
        try:
            final_action, decision_packet = cognitive_core.make_decision(scenario['game_state'])
            
            print(f"  → Decision: {final_action['action']} (amount: {final_action.get('amount', 0)})")
            print(f"  → Confidence: {decision_packet.confidence_score:.2f}")
            print(f"  → Reasoning: {decision_packet.reasoning_summary}")
            
            results.append({
                'scenario': scenario['name'],
                'action': final_action['action'],
                'success': True
            })
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                'scenario': scenario['name'],
                'action': 'error',
                'success': False
            })
    
    return results


if __name__ == '__main__':
    print("="*60)
    print("POKERMIND COGNITIVE CORE INTEGRATION TEST")
    print("="*60)
    
    # Basic pipeline test
    success = test_decision_pipeline()
    
    if success:
        print("\n" + "="*60)
        print("TESTING MULTIPLE SCENARIOS")
        print("="*60)
        
        results = test_multiple_scenarios()
        
        print(f"\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        successful = sum(1 for r in results if r['success'])
        total = len(results)
        
        print(f"Scenarios tested: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {total - successful}")
        
        if successful == total:
            print("✓ ALL TESTS PASSED - Cognitive Core is operational!")
        else:
            print("✗ Some tests failed - Review the errors above")
            
    else:
        print("✗ Basic pipeline test failed - Core system not working")
        
    print("="*60)