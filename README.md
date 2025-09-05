# PokerMind: The Autonomous Cognitive Poker Agent

---

> **A local-first, private, and fully autonomous AI agent designed to play Texas Hold'em at a world-class level.**

---

## Table of Contents
- [Vision & Core Purpose](#vision--core-purpose)
- [Core Philosophy: System 1 & System 2 Cognitive Model](#core-philosophy-system-1--system-2-cognitive-model)
- [System Architecture](#system-architecture)
- [Component Breakdown](#component-breakdown)
- [Technology Stack](#technology-stack)
- [Cognitive Workflow](#cognitive-workflow)
- [Project Directory Structure](#project-directory-structure)
- [Development & Testing Workflow](#development--testing-workflow)
- [Getting Started & Roadmap](#getting-started--roadmap)
- [Hardware Considerations](#hardware-considerations)

---

## Vision & Core Purpose
PokerMind is built to be more than just a strong poker bot. Its goal is to develop a "thinking" machine that adapts its strategy, models opponents, and reflects on its own decision-making process—all while running efficiently on personal desktop hardware.

## Core Philosophy: System 1 & System 2 Cognitive Model
Inspired by Daniel Kahneman's dual-process model:
- **System 1 (Intuition):** Lightning-fast, parallel-processed "gut reaction" using a GTO Core, opponent data lookup, and heuristics engine. Provides a high-quality baseline decision in under a second.
- **System 2 (Conscious Thought):** Slower, deliberate analytical process. The Synthesizer (Meta-Cognitive Layer) weighs intuitive inputs, assesses uncertainty, and makes a final, reasoned decision. The LLM Narrator provides post-hoc self-reflection.

This ensures real-time action with complex, multi-layered reasoning.

## System Architecture
```
INPUT: Raw Game State --> [UNIFIED COGNITIVE CORE] --> OUTPUT: Final Action
                             |
                             | [A] PARALLEL PROCESSORS
                             |   /      |      \
                             | GTO   Opponent  Heuristics
                             | Core   Modeler   Engine
                             |   \      |      /
                             |        |
                             | [B] THE SYNTHESIZER
                             | (Meta-Cognitive Layer)
                             |        |
                             +----------------------+
                                      |
                                      | [C] THE DECISION PACKET
                                      |
                             +----------------------+
                             | [D] ASYNCHRONOUS MODULES
                             |   /             \
                             | LLM Narrator   Learning
                             | (Reflection)   Module
                             +----------------------+
```

## Component Breakdown
- **[A] Parallel Processors (System 1):**
  - **GTO Core:** Deep RL model (PokerRL, TensorRT/ONNX) for GTO-sound baseline actions.
  - **Opponent Modeler:** Real-time statistical snapshot of opponent tendencies.
  - **Heuristics Engine:** Hard-coded rules for trivial decisions.
- **[B] The Synthesizer (System 2):**
  - Weighs GTO, opponent model, and heuristics to make the final decision, blending GTO and exploitative play.
- **[C] The Decision Packet:**
  - Structured object (JSON/dict) containing the full context of a decision.
- **[D] Asynchronous Modules:**
  - **LLM Narrator:** Local LLM generates natural language explanations, grounded by the Decision Packet.
  - **Learning Module:** Logs decisions and outcomes for offline analysis and model updates.

## Technology Stack
- **Core Language:** Python 3.10+
- **Poker Environment (Dev/Test):** PyPokerEngine
- **Visualization:** WorldSeriesOfPython (Pygame-based GUI)
- **GTO Core Framework:** PokerRL
- **AI/ML Libraries:** PyTorch or TensorFlow
- **Inference Optimization:** NVIDIA TensorRT or ONNX Runtime
- **LLM Integration:** LM Studio (local server API)
- **Data Handling:** NumPy, Pandas

## Cognitive Workflow
A single decision is made in a human-like timeframe (2–10 seconds):

| Time      | Step                                                                 |
|-----------|----------------------------------------------------------------------|
| 0ms       | Perception: PyPokerEngine calls `declare_action`. State is vectorized |
| 50ms      | System 1: State fed to GTO Core, Opponent Modeler, Heuristics Engine  |
| 500ms     | Intuition: Synthesizer receives outputs from parallel processors      |
| 750ms     | System 2: Synthesizer applies logic, may override GTO with exploit   |
| 800ms     | Action: Final action returned to PyPokerEngine                       |
| >800ms    | Reflection: Decision Packet sent to LLM Narrator & Learning Module   |

## Project Directory Structure
```
/project_pokermind
|
├── agent/
│   ├── __init__.py
│   ├── agent.py                # Main agent class
│   ├── cognitive_core.py       # Unified Cognitive Core
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── gto_core.py         # PokerRL/ONNX model wrapper
│   │   ├── opponent_modeler.py
│   │   ├── heuristics.py
│   │   └── synthesizer.py
│   └── utils/                  # Helper functions
|
├── config/
│   ├── agent_config.yaml       # Hyperparameters, model paths
│   └── llm_config.json         # LM Studio endpoint, prompts
|
├── data/
│   ├── hand_history/           # Decision Packets & outcomes
│   └── opponent_profiles/      # Saved opponent models
|
├── models/
│   └── gto_core_v1.onnx        # Optimized GTO model
|
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   └── 02_model_evaluation.ipynb
|
├── training/
│   └── train_pokerRL.py        # Training scripts
|
├── main.py                     # Main simulation script
└── README.md                   # This file
```

## Development & Testing Workflow
- **Core Development:**
  - Develop and test logic within PyPokerEngine for rapid iteration and automated testing.
- **Visualization:**
  - For visual debugging or showcasing, integrate with a graphical UI (e.g., WorldSeriesOfPython).

## Getting Started & Roadmap
**Phase 0: Environment Setup**
- Install all dependencies from the Technology Stack
- Set up LM Studio with a quantized 7B model (e.g., Mistral 7B Q4_K_M)
- Clone the project directory structure

**Phase 1: The Skeleton**
- Implement `agent.py`
- Integrate with `main.py` to play random actions against baseline bots

**Phase 2: The GTO Core**
- Use PokerRL to train a baseline model for Heads-Up No-Limit Hold'em
- Optimize model to ONNX/TensorRT and place in `/models`
- Implement `gto_core.py` for inference

**Phase 3: The Cognitive Layers**
- Build OpponentModeler, Heuristics, and Synthesizer modules
- Integrate in CognitiveCore class

**Phase 4: The Subconscious**
- Implement async LLM narration
- Set up logging for the Learning Module

## Hardware Considerations
- **System:** Personal Desktop (16GB RAM, NVIDIA GTX 1070 8GB VRAM)
- **Training:** Use pre-trained models or long runs for simple formats
- **Inference:** 8GB VRAM is sufficient for GTO Core (~1.5GB) and quantized 7B LLM (~5GB). 16GB RAM is adequate with efficient management.

---

*For questions, contributions, or issues, please open an issue or pull request on GitHub.*