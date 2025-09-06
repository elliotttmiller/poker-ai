"""
Unified Cognitive Core for Project PokerMind.

This module implements the dual-process cognitive architecture:
- System 1 (Intuition): Fast, parallel processing
- System 2 (Deliberation): Analytical decision-making
"""

import logging
import time
import threading
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from .modules.gto_core import GTOCore
from .modules.opponent_modeler import OpponentModeler
from .modules.heuristics import HeuristicsEngine
from .modules.synthesizer import Synthesizer
from .modules.hand_strength_estimator import HandStrengthEstimator


@dataclass
class DecisionPacket:
    """
    Structured data packet containing the full context of a decision.
    
    This packet is used for logging, analysis, and LLM narration.
    """
    timestamp: str
    round_count: int
    street: str
    hole_cards: list
    community_cards: list
    pot_size: int
    our_stack: int
    
    # System 1 outputs
    gto_recommendation: Dict[str, Any]
    opponent_model: Dict[str, Any]
    heuristics_output: Dict[str, Any]
    
    # System 2 processing
    synthesizer_analysis: Dict[str, Any]
    final_action: Dict[str, Any]
    
    # Confidence and reasoning
    confidence_score: float
    reasoning_summary: str
    
    # Performance metrics
    total_processing_time: float
    system1_time: float
    system2_time: float
    
    # Optional fields (with defaults)
    hand_strength_estimate: Optional[Dict[str, Any]] = None


class CognitiveCore:
    """
    The Unified Cognitive Core implementing the dual-process architecture.
    
    Coordinates System 1 (fast intuition) and System 2 (deliberate analysis)
    to make optimal poker decisions.
    """

    def __init__(self):
        """Initialize the Cognitive Core and all its modules."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize System 1 modules (parallel processors)
        self.gto_core = GTOCore()
        self.opponent_modeler = OpponentModeler()
        self.heuristics_engine = HeuristicsEngine()
        self.hand_strength_estimator = HandStrengthEstimator()
        
        # Initialize System 2 module (synthesizer)
        self.synthesizer = Synthesizer()
        
        # Game state tracking
        self.current_round = 0
        self.current_street = "preflop"
        self.hole_cards = []
        self.seats = []
        
        # Asynchronous processing
        self.async_thread_pool = []
        
        self.logger.info("Unified Cognitive Core initialized")

    def make_decision(self, game_state: Dict[str, Any]) -> Tuple[Dict[str, Any], DecisionPacket]:
        """
        Main decision-making method implementing the dual-process architecture.
        
        Args:
            game_state: Parsed game state from PyPokerEngine
            
        Returns:
            Tuple of (final_action, decision_packet)
        """
        start_time = time.time()
        
        try:
            # Phase 1: System 1 - Parallel Processing (Intuition)
            system1_start = time.time()
            system1_outputs = self._run_system1_parallel(game_state)
            system1_time = time.time() - system1_start
            
            # Phase 2: System 2 - Synthesizer (Deliberation)
            system2_start = time.time()
            final_action, synthesizer_analysis = self.synthesizer.synthesize_decision(
                game_state, system1_outputs
            )
            system2_time = time.time() - system2_start
            
            # Create decision packet
            total_time = time.time() - start_time
            decision_packet = self._create_decision_packet(
                game_state, system1_outputs, synthesizer_analysis, 
                final_action, total_time, system1_time, system2_time
            )
            
            # Phase 3: Asynchronous Processing (Learning & Narration)
            self._trigger_async_processing(decision_packet)
            
            self.logger.info(
                f"Decision: {final_action['action']} "
                f"(confidence: {decision_packet.confidence_score:.2f}) "
                f"in {total_time:.3f}s"
            )
            
            return final_action, decision_packet
            
        except Exception as e:
            self.logger.error(f"Error in make_decision: {e}")
            # Return safe fallback
            fallback_action = {'action': 'fold', 'amount': 0}
            empty_packet = self._create_empty_decision_packet()
            return fallback_action, empty_packet

    def _run_system1_parallel(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run System 1 parallel processors for fast intuitive processing.
        
        Args:
            game_state: Current game state
            
        Returns:
            Dict containing outputs from all System 1 modules
        """
        system1_outputs = {}
        
        # Run all System 1 modules in parallel using threading
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all System 1 tasks
            future_to_module = {
                executor.submit(self.gto_core.get_recommendation, game_state): 'gto',
                executor.submit(self.opponent_modeler.get_opponent_analysis, game_state): 'opponents',
                executor.submit(self.heuristics_engine.check_trivial_decisions, game_state): 'heuristics',
                executor.submit(self.hand_strength_estimator.estimate, game_state): 'hand_strength',
            }
            
            # Collect results with timeout
            for future in concurrent.futures.as_completed(future_to_module, timeout=0.5):
                module_name = future_to_module[future]
                try:
                    result = future.result()
                    system1_outputs[module_name] = result
                    self.logger.debug(f"System 1 module {module_name} completed successfully")
                except Exception as e:
                    self.logger.warning(f"System 1 module {module_name} failed: {e}")
                    system1_outputs[module_name] = self._get_fallback_output(module_name)
        
        return system1_outputs

    def _create_decision_packet(
        self, 
        game_state: Dict[str, Any],
        system1_outputs: Dict[str, Any],
        synthesizer_analysis: Dict[str, Any],
        final_action: Dict[str, Any],
        total_time: float,
        system1_time: float,
        system2_time: float
    ) -> DecisionPacket:
        """Create a complete decision packet for logging and analysis."""
        
        return DecisionPacket(
            timestamp=datetime.now().isoformat(),
            round_count=game_state.get('round_count', 0),
            street=game_state.get('street', 'preflop'),
            hole_cards=game_state.get('hole_cards', []),
            community_cards=game_state.get('community_cards', []),
            pot_size=game_state.get('pot_size', 0),
            our_stack=game_state.get('our_stack', 0),
            
            gto_recommendation=system1_outputs.get('gto', {}),
            opponent_model=system1_outputs.get('opponents', {}),
            heuristics_output=system1_outputs.get('heuristics', {}),
            hand_strength_estimate=system1_outputs.get('hand_strength'),
            
            synthesizer_analysis=synthesizer_analysis,
            final_action=final_action,
            
            confidence_score=synthesizer_analysis.get('confidence', 0.5),
            reasoning_summary=synthesizer_analysis.get('reasoning', 'No reasoning provided'),
            
            total_processing_time=total_time,
            system1_time=system1_time,
            system2_time=system2_time
        )

    def _trigger_async_processing(self, decision_packet: DecisionPacket):
        """Trigger asynchronous processing for learning and narration."""
        # TODO: Implement LLM narration in separate thread
        # TODO: Implement learning module logging
        pass

    def _get_fallback_output(self, module_name: str) -> Dict[str, Any]:
        """Get fallback output when a System 1 module fails."""
        fallbacks = {
            'gto': {'action': 'fold', 'confidence': 0.1},
            'opponents': {'analysis': 'no data'},
            'heuristics': {'recommendation': None},
            'hand_strength': {'strength': 0.5}
        }
        return fallbacks.get(module_name, {})

    def _create_empty_decision_packet(self) -> DecisionPacket:
        """Create an empty decision packet for error cases."""
        return DecisionPacket(
            timestamp=datetime.now().isoformat(),
            round_count=0, street='unknown', hole_cards=[], community_cards=[],
            pot_size=0, our_stack=0, gto_recommendation={}, opponent_model={},
            heuristics_output={}, synthesizer_analysis={}, final_action={'action': 'fold'},
            confidence_score=0.0, reasoning_summary='Error occurred',
            total_processing_time=0.0, system1_time=0.0, system2_time=0.0
        )

    # Game state tracking methods
    def reset_for_new_game(self):
        """Reset the cognitive core for a new game."""
        self.opponent_modeler.reset()
        self.logger.info("Cognitive core reset for new game")

    def reset_for_new_round(self, round_count: int, hole_cards: list, seats: list):
        """Reset for a new round (hand)."""
        self.current_round = round_count
        self.hole_cards = hole_cards
        self.seats = seats
        self.current_street = "preflop"

    def update_street(self, street: str, round_state: Dict[str, Any]):
        """Update when moving to a new street."""
        self.current_street = street

    def process_opponent_action(self, action: Dict[str, Any], round_state: Dict[str, Any]):
        """Process an opponent's action for learning."""
        self.opponent_modeler.update_from_action(action, round_state)

    def process_round_result(
        self, 
        winners: list, 
        hand_info: list, 
        round_state: Dict[str, Any]
    ):
        """Process the result of a completed round."""
        self.opponent_modeler.update_from_result(winners, hand_info, round_state)