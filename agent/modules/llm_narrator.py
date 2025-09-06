"""
LLM Narrator Module for Project PokerMind.

This module provides asynchronous natural language narration of poker decisions
using the configured Llama 3.1 model via LM Studio.
"""

import json
import logging
import threading
import time
from typing import Dict, Any, Optional
from datetime import datetime
import requests
from dataclasses import asdict
import os
# Try to load environment variables from .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not available, use environment variables as-is
    pass


class LLMNarrator:
    """
    Asynchronous LLM-powered narrator for poker decisions.

    Uses the Llama 3.1 model via LM Studio to provide human-readable
    explanations of the agent's decision-making process.
    """

    def __init__(self, config_path: str = "config/llm_config.json"):
        """
        Initialize the LLM Narrator.

        Args:
            config_path: Path to LLM configuration file
        """
        self.logger = logging.getLogger(__name__)

        # Load configuration
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                self.prompts = config["prompts"]
                base_narration_settings = config["narration_settings"]
        except Exception as e:
            self.logger.warning(
                f"Failed to load LLM config from file: {e}, using environment variables and defaults"
            )
            self.prompts = {
                "decision_analysis": {
                    "system": "You are PokerMind, an advanced poker AI. Analyze the decision made and provide natural language explanation of the reasoning process. Be concise but insightful.",
                    "user_template": "Game State: {game_state}\nDecision Made: {action}\nReasoning Data: {decision_packet}\n\nProvide a brief analysis of this poker decision:",
                }
            }
            base_narration_settings = {}

        # Load LLM configuration from environment variables (overrides file config)
        self.llm_config = {
            "base_url": os.getenv("LLM_BASE_URL", "http://localhost:1234/v1"),
            "api_key": os.getenv("LLM_API_KEY", "not-needed-for-local"),
            "model_name": os.getenv("LLM_MODEL_NAME", "Meta-Llama-3.1-8B-Instruct-Q5_K_M"),
            "timeout": int(os.getenv("LLM_TIMEOUT", "30")),
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "500")),
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
        }

        # Load narration settings from environment variables (overrides file config)
        self.narration_settings = {
            "enabled": os.getenv("SAVE_DECISIONS", "true").lower() == "true",
            "async_mode": os.getenv("PARALLEL_PROCESSING", "true").lower() == "true",
            "save_to_file": os.getenv("SAVE_HAND_HISTORY", "true").lower() == "true",
            "output_directory": os.getenv(
                "NARRATION_OUTPUT_DIR", base_narration_settings.get("output_directory", "data/hand_history/narrations/")
            ),
        }

        # Ensure output directory exists
        self.output_dir = self.narration_settings.get("output_directory", "data/")
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize logging file
        self.log_file = os.getenv(
            "NARRATION_LOG_PATH", os.path.join(self.output_dir, "narration_log.txt")
        )

        self.logger.info("LLM Narrator initialized")

    def narrate_decision(self, decision_packet) -> None:
        """
        Generate asynchronous narration of a poker decision.

        This method runs in a separate thread as requested in Sub-Task 3.3.

        Args:
            decision_packet: DecisionPacket object containing full decision context
        """
        if not self.narration_settings.get("enabled", True):
            return

        if self.narration_settings.get("async_mode", True):
            # Run in separate thread as requested
            thread = threading.Thread(
                target=self._generate_narration, args=(decision_packet,), daemon=True
            )
            thread.start()
            self.logger.debug("Async narration thread started")
        else:
            # Synchronous mode for testing
            self._generate_narration(decision_packet)

    def _generate_narration(self, decision_packet) -> None:
        """
        Internal method to generate the actual narration.

        Args:
            decision_packet: DecisionPacket object
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

            # Prepare the prompt
            game_context = self._extract_game_context(packet_data)
            action_summary = self._extract_action_summary(packet_data)

            # Generate narration via LLM API
            narration = self._call_llm_api(game_context, action_summary, packet_data)

            if narration:
                # Log the narration
                self._log_narration(packet_data, narration)
                self.logger.debug("Narration generated and logged successfully")
            else:
                self.logger.warning("Failed to generate narration")

        except Exception as e:
            self.logger.error(f"Error generating narration: {e}")

    def _extract_game_context(self, packet_data: Dict[str, Any]) -> str:
        """Extract key game context for narration."""
        street = packet_data.get("street", "unknown")
        pot_size = packet_data.get("pot_size", 0)
        our_stack = packet_data.get("our_stack", 0)
        hole_cards = packet_data.get("hole_cards", [])
        community_cards = packet_data.get("community_cards", [])

        context = f"Street: {street}, Pot: {pot_size}, Our Stack: {our_stack}"
        if hole_cards:
            context += f", Hole Cards: {hole_cards}"
        if community_cards:
            context += f", Board: {community_cards}"

        return context

    def _extract_action_summary(self, packet_data: Dict[str, Any]) -> str:
        """Extract action summary for narration."""
        final_action = packet_data.get("final_action", {})
        action = final_action.get("action", "unknown")
        amount = final_action.get("amount", 0)
        confidence = packet_data.get("confidence_score", 0)

        if action == "fold":
            return f"FOLDED (confidence: {confidence:.2f})"
        elif action == "call":
            return f"CALLED {amount} (confidence: {confidence:.2f})"
        elif action == "raise":
            return f"RAISED to {amount} (confidence: {confidence:.2f})"
        else:
            return f"{action.upper()} {amount} (confidence: {confidence:.2f})"

    def _call_llm_api(
        self, game_context: str, action_summary: str, packet_data: Dict[str, Any]
    ) -> Optional[str]:
        """
        Make API call to LM Studio for narration generation.

        Implements robust error handling as requested in Sub-Task 3.3.
        """
        try:
            # Get prompt template
            decision_prompt = self.prompts.get("decision_analysis", {})
            system_prompt = decision_prompt.get("system", "Analyze this poker decision.")

            # Format user prompt
            user_content = f"""
Game Context: {game_context}
Action Taken: {action_summary}
Reasoning: {packet_data.get('reasoning_summary', 'No reasoning available')}
Confidence: {packet_data.get('confidence_score', 0.0):.2f}
Processing Time: {packet_data.get('total_processing_time', 0.0):.3f}s

Provide a concise analysis of this poker decision in 2-3 sentences:
"""

            # Prepare API request
            payload = {
                "model": self.llm_config.get("model_name", "local-model"),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "max_tokens": self.llm_config.get("max_tokens", 500),
                "temperature": self.llm_config.get("temperature", 0.7),
                "stream": False,
            }

            # Make the API call with timeout
            response = requests.post(
                f"{self.llm_config['base_url']}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.llm_config.get('api_key', 'not-needed')}",
                },
                json=payload,
                timeout=self.llm_config.get("timeout", 30),
            )

            if response.status_code == 200:
                response_data = response.json()
                narration = (
                    response_data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
                return narration
            else:
                self.logger.warning(f"LLM API error: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.Timeout:
            self.logger.warning("LLM API timeout - narration skipped")
            return None
        except requests.exceptions.ConnectionError:
            self.logger.warning("LLM API connection error - narration skipped")
            return None
        except Exception as e:
            self.logger.error(f"LLM API call failed: {e}")
            return None

    def _log_narration(self, packet_data: Dict[str, Any], narration: str) -> None:
        """
        Log the narration to file as requested in Sub-Task 3.3.

        Appends to data/narration_log.txt with structured format.
        """
        try:
            timestamp = packet_data.get("timestamp", datetime.now().isoformat())
            street = packet_data.get("street", "unknown")
            action = packet_data.get("final_action", {}).get("action", "unknown")

            log_entry = f"""
[{timestamp}] Street: {street} | Action: {action}
Context: {self._extract_game_context(packet_data)}
Decision: {self._extract_action_summary(packet_data)}
AI Analysis: {narration}
---
"""

            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_entry)

        except Exception as e:
            self.logger.error(f"Failed to log narration: {e}")

    def get_narration_status(self) -> Dict[str, Any]:
        """Get status of the narration system."""
        return {
            "enabled": self.narration_settings.get("enabled", True),
            "async_mode": self.narration_settings.get("async_mode", True),
            "log_file": self.log_file,
            "log_file_exists": os.path.exists(self.log_file),
            "api_endpoint": self.llm_config.get("base_url", "unknown"),
        }
