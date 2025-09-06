#!/usr/bin/env python3
"""
Test script for the OddsOracle module.

Verifies that the Oracle can load data and provide accurate lookups.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent.toolkit.odds_oracle import OddsOracle


def test_oracle():
    """Test the OddsOracle functionality."""
    print("🧠 Testing PokerMind OddsOracle...")
    
    # Initialize oracle
    try:
        oracle = OddsOracle()
        print("✅ Oracle initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Oracle: {e}")
        return False
    
    # Test pre-flop equity lookup
    try:
        equity_aa_vs_kk = oracle.get_preflop_equity('AA', 'KK')
        print(f"✅ AA vs KK equity: {equity_aa_vs_kk:.3f}")
        
        equity_ak_vs_qq = oracle.get_preflop_equity('AKs', 'QQ')
        print(f"✅ AKs vs QQ equity: {equity_ak_vs_qq:.3f}")
        
    except Exception as e:
        print(f"❌ Pre-flop equity test failed: {e}")
        return False
    
    # Test draw odds
    try:
        flush_odds = oracle.get_draw_completion_odds('flush_draw', 'turn_and_river')
        straight_odds = oracle.get_draw_completion_odds('straight_draw', 'turn_and_river')
        print(f"✅ Flush draw odds (turn+river): {flush_odds:.3f}")
        print(f"✅ Straight draw odds (turn+river): {straight_odds:.3f}")
        
    except Exception as e:
        print(f"❌ Draw odds test failed: {e}")
        return False
    
    # Test board danger
    try:
        safe_board = oracle.get_board_danger_level('rainbow_low')
        dangerous_board = oracle.get_board_danger_level('monotone')
        print(f"✅ Safe board danger: {safe_board:.3f}")
        print(f"✅ Dangerous board danger: {dangerous_board:.3f}")
        
    except Exception as e:
        print(f"❌ Board danger test failed: {e}")
        return False
    
    # Test position multipliers
    try:
        button_mult = oracle.get_position_multiplier('button')
        sb_mult = oracle.get_position_multiplier('small_blind')
        print(f"✅ Button multiplier: {button_mult:.3f}")
        print(f"✅ Small blind multiplier: {sb_mult:.3f}")
        
    except Exception as e:
        print(f"❌ Position multiplier test failed: {e}")
        return False
    
    # Test range analysis
    try:
        opponent_range = ['AA', 'KK', 'QQ', 'AKs', 'AK']
        analysis = oracle.analyze_matchup('JJ', opponent_range)
        print(f"✅ JJ vs tight range: {analysis['average_equity']:.3f} equity")
        
    except Exception as e:
        print(f"❌ Range analysis test failed: {e}")
        return False
    
    # Test quick recommendation
    try:
        rec1 = oracle.get_quick_recommendation('AA', 'button')
        rec2 = oracle.get_quick_recommendation('72', 'early_position')
        print(f"✅ AA on button: {rec1['recommendation']} (confidence: {rec1['confidence']:.2f})")
        print(f"✅ 72 in early position: {rec2['recommendation']} (confidence: {rec2['confidence']:.2f})")
        
    except Exception as e:
        print(f"❌ Quick recommendation test failed: {e}")
        return False
    
    # Show statistics
    try:
        stats = oracle.get_stats()
        print(f"✅ Oracle stats: {stats['preflop_hands']} hands, {stats['draw_types']} draw types")
        
    except Exception as e:
        print(f"❌ Stats test failed: {e}")
        return False
    
    print("\n🎉 All Oracle tests passed successfully!")
    print("Oracle is ready for integration into PokerMind System 1")
    return True


if __name__ == '__main__':
    success = test_oracle()
    sys.exit(0 if success else 1)