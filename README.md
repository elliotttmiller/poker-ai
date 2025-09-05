Project PokerMind: The Autonomous Cognitive Poker Agent
1. Vision & Core Purpose
Project PokerMind is a local-first, private, and fully autonomous AI agent designed to play Texas Hold'em at a world-class level. The agent operates on a sophisticated, multi-tiered cognitive architecture that mimics human-like reasoning, combining rapid intuition with deep, analytical thought.
The primary goal is not just to create a strong poker bot, but to develop a "thinking" machine that can adapt its strategy, model its opponents, and reflect on its own decision-making process, all while running efficiently on personal desktop hardware.
2. Core Philosophy: The System 1 & System 2 Cognitive Model
The agent's intelligence is architected around Daniel Kahneman's dual-process model of human cognition, ensuring a workflow that is both incredibly fast and deeply analytical.
System 1 ("Intuition"): A lightning-fast, parallel-processed "gut reaction." This is handled by a highly optimized GTO Core, an opponent data lookup, and a heuristics engine. It provides an immediate, high-quality baseline decision in under a second.
System 2 ("Conscious Thought"): A slower, more deliberate analytical process. This is the Synthesizer (Meta-Cognitive Layer) that weighs the intuitive inputs, assesses uncertainty, and makes a final, reasoned decision. The LLM Narrator then provides a post-hoc "self-reflection" on why the decision was made.
This design ensures the agent can act in real-time while still leveraging complex, multi-layered reasoning.
3. Full System Architecture: The Unified Cognitive Core
The agent's logic is not a simple sequential script but a unified, data-flow architecture designed for maximum performance and reliability.
code
Code
+---------------------------+
INPUT: Raw Game State --> |    UNIFIED COGNITIVE CORE   | --> OUTPUT: Final Action
                                     |                           |
                                     | [A] PARALLEL PROCESSORS   |
                                     |   /         |         \   |
                                     | GTO      Opponent    Heuristics
                                     | Core      Modeler      Engine
                                     |   \         |         /   |
                                     |           |           |
                                     | [B] THE SYNTHESIZER       |
                                     | (Meta-Cognitive Layer)    |
                                     |           |           |
                                     +---------------------------+
                                                 |
                                                 | [C] THE DECISION PACKET
                                                 |
                                     +---------------------------+
                                     | [D] ASYNCHRONOUS MODULES  |
                                     |      /               \    |
                                     |   LLM Narrator    Learning
                                     |  (Self-Reflection)   Module
                                     +---------------------------+
Component Breakdown:
[A] Parallel Processors (System 1):
GTO Core: A deep reinforcement learning model, trained using the PokerRL framework and optimized for inference with TensorRT/ONNX. It provides the GTO-sound baseline action.
Opponent Modeler: A fast, in-memory database/hash map that provides a real-time statistical snapshot of the opponent's tendencies (VPIP, PFR, etc.).
Heuristics Engine: A simple, hard-coded rules engine to instantly handle obvious, computationally trivial decisions (e.g., folding 7-2 offsuit pre-flop).
[B] The Synthesizer (System 2):
The "executive function" of the agent. A lightweight, CPU-bound module that receives the outputs from the parallel processors.
It weighs the GTO recommendation against the opponent model and heuristic flags to make the final, optimal decision. It is responsible for blending GTO play with exploitative adjustments.
[C] The Decision Packet:
A structured data object (JSON/dict) that contains the entire context of a single decision: the raw game state, all processor outputs, and the final action. This is the "memory" of the decision.
[D] Asynchronous Modules (The "Subconscious"):
LLM Narrator: The Decision Packet is sent to a local LLM (e.g., Mistral 7B via LM Studio). The LLM is strictly "grounded" by the packet's data and generates a natural language explanation of the agent's thought process. This prevents hallucinations and ensures the LLM acts as a reliable narrator, not a decision-maker.
Learning Module: The Decision Packet and the hand's outcome are logged for offline analysis, updating the opponent model, and flagging potential leaks in the agent's own strategy.
4. Technology Stack
Core Language: Python 3.10+
Poker Environment (Dev/Test): PyPokerEngine
Poker Environment (Visualization): WorldSeriesOfPython (or similar Pygame-based GUI, requires manual integration)
GTO Core Framework: PokerRL
AI/ML Libraries: PyTorch or TensorFlow
Inference Optimization: NVIDIA TensorRT or ONNX Runtime
LLM Integration: LM Studio (local server API)
Data Handling: NumPy, Pandas
5. End-to-End Cognitive Workflow (A Single Decision)
This entire workflow is designed to complete within a human-like timeframe (2-10 seconds).
t = 0ms (Perception): The PyPokerEngine environment calls the agent's declare_action method. The raw game state is vectorized.
t = 50ms (System 1 Activation): The vectorized state is fed to the GTO Core, Opponent Modeler, and Heuristics Engine simultaneously.
t = 500ms (Intuition Arrives): The Synthesizer receives the outputs from the parallel processors. It now has a GTO action, opponent stats, and any heuristic flags.
t = 750ms (System 2 Deliberation): The Synthesizer applies its logic, weighing the inputs. It might decide the GTO action is best, or it might override it with a calculated exploit.
t = 800ms (Action Taken): The final action is returned to PyPokerEngine. The agent has acted.
t > 800ms (Asynchronous Reflection): A Decision Packet is created and sent to the LLM Narrator and Learning Module in a separate, non-blocking thread. The agent is already waiting for the next game action, completely unaffected by the speed of these background tasks.
6. Project Directory Structure
A clean, modular structure is essential for this project.
code
Code
/project_pokermind
|
├── agent/
│   ├── __init__.py
│   ├── agent.py                # Main agent class inheriting from BasePokerPlayer
│   ├── cognitive_core.py       # The Unified Cognitive Core implementation
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── gto_core.py         # Wrapper for the PokerRL/ONNX model
│   │   ├── opponent_modeler.py
│   │   ├── heuristics.py
│   │   └── synthesizer.py
│   └── utils/                  # Helper functions, state vectorization, etc.
|
├── config/
│   ├── agent_config.yaml       # Hyperparameters, model paths, risk tolerance
│   └── llm_config.json         # LM Studio endpoint, prompts
|
├── data/
│   ├── hand_history/           # Logged Decision Packets and outcomes
│   └── opponent_profiles/      # Saved opponent models
|
├── models/
│   └── gto_core_v1.onnx        # The optimized GTO model
|
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   └── 02_model_evaluation.ipynb
|
├── training/
│   └── train_pokerRL.py        # Scripts for training/fine-tuning the GTO Core
|
├── main.py                     # Main script to run simulations using PyPokerEngine
└── README.md                   # This file
7. Development & Testing Workflow
Core Development (95% of Time): All logic is developed and tested within the PyPokerEngine framework. Its speed and direct Python integration are ideal for rapid iteration, debugging, and automated testing.
Visualization (5% of Time): For visual debugging or showcasing, manually integrate the agent into a graphical UI like WorldSeriesOfPython. This is a secondary task focused on presentation, not core development.
8. Getting Started & Roadmap
Phase 0: Environment Setup
Install all dependencies from the Technology Stack.
Set up LM Studio with a quantized 7B parameter model (e.g., Mistral 7B Q4_K_M).
Clone the project directory structure.
Phase 1: The Skeleton
Implement the basic agent.py class.
Integrate it with main.py to play a game of random actions against baseline bots in PyPokerEngine.
Phase 2: The GTO Core
Use PokerRL to train a baseline model for Heads-Up No-Limit Hold'em.
Optimize this model to ONNX/TensorRT format and place it in /models.
Implement the gto_core.py module to load and run inference with this model.
Phase 3: The Cognitive Layers
Build the OpponentModeler, Heuristics, and Synthesizer modules.
Wire them together in the CognitiveCore class according to the architecture.
Phase 4: The Subconscious
Implement the asynchronous call to the LM Studio API for narration.
Set up the logging mechanism for the Learning Module.
9. Hardware Considerations
System: Personal Desktop (16GB RAM, NVIDIA GTX 1070 8GB VRAM)
Training: The primary bottleneck. Training a world-class GTO Core from scratch is not feasible. The strategy is to use pre-trained models or conduct very long training runs for simpler game formats (e.g., Heads-Up).
Inference (Gameplay): Fully Feasible. The 8GB of VRAM is sufficient to run an optimized GTO Core (~1.5GB) alongside a quantized 7B LLM (~5GB). The 16GB of system RAM is adequate but requires efficient memory management. The Unified Cognitive Core architecture is specifically designed to maximize performance on this hardware.