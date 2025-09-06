"""
Learning Module for Project PokerMind.

This module handles asynchronous logging of hand data for offline training
and analysis, creating structured JSON Lines files.
"""

import json
import logging
import threading
import os
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import asdict
import uuid


class LearningModule:
    """
    Asynchronous learning and data logging system for poker hands.

    Logs structured data to JSON Lines (.jsonl) files for future
    offline training and analysis.
    """

    def __init__(self, output_directory: str = "data/hand_history/"):
        """
        Initialize the Learning Module.

        Args:
            output_directory: Directory to store hand history files
        """
        self.logger = logging.getLogger(__name__)

        # Set up output directory
        self.output_directory = output_directory
        os.makedirs(self.output_directory, exist_ok=True)

        # Create session-specific log file
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.hand_history_file = os.path.join(
            self.output_directory, f"hand_history_{session_id}.jsonl"
        )

        # Threading for async operations
        self.async_queue = []
        self.lock = threading.Lock()

        # Session tracking
        self.session_stats = {
            "hands_logged": 0,
            "session_start": datetime.now().isoformat(),
            "session_id": session_id,
        }

        self.logger.info(f"Learning Module initialized - logging to {self.hand_history_file}")

    def log_hand(self, decision_packet, hand_outcome: Dict[str, Any]) -> None:
        """
        Log a completed hand for learning and analysis.

        This method runs asynchronously in a separate thread as requested in Sub-Task 3.4.

        Args:
            decision_packet: DecisionPacket containing the decision context
            hand_outcome: Final hand result (pot won, winning hand, etc.)
        """
        # Run in separate thread for non-blocking operation
        thread = threading.Thread(
            target=self._async_log_hand,
            args=(decision_packet, hand_outcome),
            daemon=True,
        )
        thread.start()
        self.logger.debug("Async hand logging thread started")

    def _async_log_hand(self, decision_packet, hand_outcome: Dict[str, Any]) -> None:
        """
        Internal asynchronous hand logging method.

        Args:
            decision_packet: DecisionPacket object
            hand_outcome: Hand outcome data
        """
        try:
            # Convert decision packet to serializable format
            if hasattr(decision_packet, "__dict__"):
                packet_data = (
                    asdict(decision_packet)
                    if hasattr(decision_packet, "__dataclass_fields__")
                    else vars(decision_packet)
                )
            else:
                packet_data = decision_packet

            # Create structured learning record
            learning_record = self._create_learning_record(packet_data, hand_outcome)

            # Write to JSON Lines file
            with self.lock:  # Thread-safe file writing
                self._write_jsonl_record(learning_record)
                self.session_stats["hands_logged"] += 1

            self.logger.debug(
                f"Hand logged successfully - total hands: {self.session_stats['hands_logged']}"
            )

        except Exception as e:
            self.logger.error(f"Error logging hand: {e}")

    def _create_learning_record(
        self, packet_data: Dict[str, Any], hand_outcome: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a structured learning record for ML training.

        Combines decision context with hand outcome for supervised learning.
        """
        # Generate unique hand ID
        hand_id = str(uuid.uuid4())[:8]

        # Extract core features for learning
        learning_record = {
            # Metadata
            "hand_id": hand_id,
            "timestamp": packet_data.get("timestamp", datetime.now().isoformat()),
            "session_id": self.session_stats["session_id"],
            # Game context features
            "game_state": {
                "street": packet_data.get("street", "unknown"),
                "pot_size": packet_data.get("pot_size", 0),
                "our_stack": packet_data.get("our_stack", 0),
                "round_count": packet_data.get("round_count", 0),
                "hole_cards": packet_data.get("hole_cards", []),
                "community_cards": packet_data.get("community_cards", []),
            },
            # Decision analysis features
            "decision_features": {
                "our_equity": packet_data.get("synthesizer_analysis", {}).get("our_equity", 0),
                "required_equity": packet_data.get("synthesizer_analysis", {}).get(
                    "required_equity", 0
                ),
                "confidence_score": packet_data.get("confidence_score", 0),
                "hand_strength_estimate": packet_data.get("hand_strength_estimate", {}),
                "reasoning_summary": packet_data.get("reasoning_summary", ""),
                "processing_time": packet_data.get("total_processing_time", 0),
            },
            # System 1 module outputs
            "system1_analysis": {
                "gto_recommendation": packet_data.get("gto_recommendation", {}),
                "opponent_model": packet_data.get("opponent_model", {}),
                "heuristics_output": packet_data.get("heuristics_output", {}),
            },
            # Final decision
            "decision_made": packet_data.get("final_action", {}),
            # Hand outcome (target for supervised learning)
            "hand_outcome": {
                "pot_won": hand_outcome.get("pot_won", 0),
                "winning_hand": hand_outcome.get("winning_hand", "unknown"),
                "showdown": hand_outcome.get("showdown", False),
                "bluff_success": hand_outcome.get("bluff_success", None),
                "final_pot_size": hand_outcome.get("final_pot_size", 0),
                "profit_loss": hand_outcome.get("profit_loss", 0),
            },
            # Performance metrics
            "performance_metrics": {
                "decision_quality_score": self._calculate_decision_quality_score(
                    packet_data, hand_outcome
                ),
                "equity_realization": self._calculate_equity_realization(packet_data, hand_outcome),
                "timing_efficiency": packet_data.get("total_processing_time", 0),
            },
        }

        return learning_record

    def _calculate_decision_quality_score(
        self, packet_data: Dict[str, Any], hand_outcome: Dict[str, Any]
    ) -> float:
        """
        Calculate a decision quality score for learning.

        This will be used for training data quality assessment.
        """
        try:
            # Get key decision metrics
            confidence = packet_data.get("confidence_score", 0)
            profit_loss = hand_outcome.get("profit_loss", 0)
            our_equity = packet_data.get("synthesizer_analysis", {}).get("our_equity", 0)

            # Simple quality score based on confidence and outcome
            base_score = confidence * 0.6  # Confidence component

            # Outcome component (positive for profitable decisions)
            if profit_loss > 0:
                outcome_component = 0.3
            elif profit_loss == 0:
                outcome_component = 0.1
            else:
                outcome_component = -0.2

            # Equity alignment component
            equity_component = our_equity * 0.1

            quality_score = base_score + outcome_component + equity_component
            return max(0.0, min(1.0, quality_score))

        except Exception as e:
            self.logger.debug(f"Error calculating decision quality: {e}")
            return 0.5

    def _calculate_equity_realization(
        self, packet_data: Dict[str, Any], hand_outcome: Dict[str, Any]
    ) -> float:
        """
        Calculate how well we realized our equity.

        Useful for training on equity realization skills.
        """
        try:
            our_equity = packet_data.get("synthesizer_analysis", {}).get("our_equity", 0.5)
            pot_won = hand_outcome.get("pot_won", 0)
            final_pot = hand_outcome.get("final_pot_size", 1)

            if final_pot <= 0:
                return 0.0

            # Calculate actual return ratio
            actual_return = pot_won / final_pot

            # Compare to expected equity
            if our_equity > 0:
                equity_realization = actual_return / our_equity
                return min(2.0, max(0.0, equity_realization))  # Cap at 200%

            return 0.5

        except Exception as e:
            self.logger.debug(f"Error calculating equity realization: {e}")
            return 0.5

    def _write_jsonl_record(self, record: Dict[str, Any]) -> None:
        """
        Write a single JSON Lines record to the hand history file.

        Args:
            record: Learning record dictionary
        """
        try:
            json_line = json.dumps(record, ensure_ascii=False) + "\n"

            with open(self.hand_history_file, "a", encoding="utf-8") as f:
                f.write(json_line)

        except Exception as e:
            self.logger.error(f"Failed to write JSONL record: {e}")

    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics."""
        with self.lock:
            stats = self.session_stats.copy()

        # Add file information
        stats["hand_history_file"] = self.hand_history_file
        stats["file_exists"] = os.path.exists(self.hand_history_file)

        if stats["file_exists"]:
            try:
                file_size = os.path.getsize(self.hand_history_file)
                stats["file_size_bytes"] = file_size
                stats["file_size_mb"] = round(file_size / (1024 * 1024), 2)
            except Exception as e:
                stats["file_size_error"] = str(e)

        return stats

    def export_training_data(
        self, output_file: str = None, filter_criteria: Dict[str, Any] = None
    ) -> str:
        """
        Export filtered training data for ML model training.

        Args:
            output_file: Output file path (optional)
            filter_criteria: Filtering criteria (optional)

        Returns:
            Path to exported file
        """
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_directory, f"training_export_{timestamp}.jsonl")

        try:
            exported_count = 0

            with open(output_file, "w", encoding="utf-8") as outfile:
                # Read and filter existing hand history
                if os.path.exists(self.hand_history_file):
                    with open(self.hand_history_file, "r", encoding="utf-8") as infile:
                        for line in infile:
                            try:
                                record = json.loads(line.strip())

                                # Apply filters if specified
                                if self._passes_filter(record, filter_criteria):
                                    outfile.write(line)
                                    exported_count += 1

                            except json.JSONDecodeError:
                                continue

            self.logger.info(f"Exported {exported_count} training records to {output_file}")
            return output_file

        except Exception as e:
            self.logger.error(f"Error exporting training data: {e}")
            return ""

    def _passes_filter(self, record: Dict[str, Any], filter_criteria: Dict[str, Any]) -> bool:
        """Check if a record passes the filter criteria."""
        if not filter_criteria:
            return True

        try:
            # Example filters
            if "min_confidence" in filter_criteria:
                if (
                    record.get("decision_features", {}).get("confidence_score", 0)
                    < filter_criteria["min_confidence"]
                ):
                    return False

            if "street" in filter_criteria:
                if record.get("game_state", {}).get("street") != filter_criteria["street"]:
                    return False

            if "min_quality_score" in filter_criteria:
                if (
                    record.get("performance_metrics", {}).get("decision_quality_score", 0)
                    < filter_criteria["min_quality_score"]
                ):
                    return False

            return True

        except Exception:
            return True  # Include record if filter evaluation fails

    def cleanup_old_files(self, days_to_keep: int = 30) -> None:
        """Clean up old hand history files."""
        try:
            cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
            cleaned_count = 0

            for filename in os.listdir(self.output_directory):
                if filename.startswith("hand_history_") and filename.endswith(".jsonl"):
                    filepath = os.path.join(self.output_directory, filename)

                    if os.path.getmtime(filepath) < cutoff_time:
                        os.remove(filepath)
                        cleaned_count += 1

            if cleaned_count > 0:
                self.logger.info(f"Cleaned up {cleaned_count} old hand history files")

        except Exception as e:
            self.logger.error(f"Error cleaning up old files: {e}")
