# From Representation to Reasoning: A Unified Framework via JEPA, Active Inference, and Developmental Principles

## Project Overview

This repository hosts the official open-source implementation for the doctoral dissertation titled "From Representation to Reasoning: A Unified Framework via JEPA, Active Inference, and Developmental Principles." It provides a modular, PyTorch-based framework that integrates **Joint Embedding Predictive Architectures (JEPA)**, **Active Inference**, and principles inspired by **Developmental Intelligence**.

The goal of this project is to demonstrate how autonomous machine intelligence can emerge from an intrinsic drive for consistency and energy minimization, moving beyond conventional AI paradigms that rely heavily on explicit supervision or pre-defined rewards. The framework enables agents to learn predictive world models, plan proactively by simulating future trajectories, and continuously self-improve.

### Key Features:

* **Unified Agent Architecture:** A cohesive system combining advanced self-supervised learning (JEPA) with variational control (Active Inference) and biologically inspired developmental principles.
* **Energy-Based Reasoning:** Implementation of an intrinsic energy function to guide planning and decision-making by minimizing internal inconsistencies.
* **Self-Improvement Loop:** Mechanisms for continuous, autonomous model refinement driven by residual energy, enabling lifelong learning.
* **Modular & Distributed Design:** Components inspired by developmental biology, fostering scalability, fault tolerance, and emergent collective behaviors.
* **Illustrative Simulations:** Practical demonstrations in GridWorld environments, including studies on deceptive agents and comprehensive ablation experiments.

## Core Concepts (Thesis Pillars)

This project is built upon three foundational theoretical pillars:

1.  **Joint Embedding Predictive Architectures (JEPA):** Learns compact, causally informed latent representations by predicting future observations in an abstract space (Chapter 3).
2.  **Active Inference:** Guides autonomous control and planning by minimizing expected free energy, unifying perception, action, and learning as inference (Chapter 4).
3.  **Developmental Intelligence:** Incorporates principles from morphogenesis and biological development for emergent self-organization, adaptability, and robustness in the agent's structure and function (Chapter 6).

These pillars converge to enable a sophisticated form of **Autonomous Machine Intelligence** capable of operating in complex, dynamic, and open-ended environments.

## Installation and Setup

To set up the project and run the simulations, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/HyungseokSeo/unified-ai.git
    cd unified-ai
    ```
    *(Remember to replace `YourUsername/your-thesis-unified-ai.git` with your actual GitHub repository URL)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

    The `requirements.txt` file (located in the root directory of this repository) will contain:
    ```
    torch>=1.10.0
    numpy>=1.20.0
    matplotlib>=3.4.0
    gymnasium>=0.28.0 # Or gym if using older versions
    pyyaml>=6.0 # For config files
    # Add any other libraries used in your code
    ```

## Project Structure

The codebase is organized into modular components mirroring the theoretical framework discussed in the dissertation:

```
your-thesis-unified-ai/
├── README.md                      # Project overview and instructions
├── requirements.txt               # Python dependencies
├── src/                           # Source code for the core framework
│   ├── agents/
│   │   ├── unified_agent.py       # Integrates all framework components into a single agent
│   │   └── base_agent.py          # Abstract base class or interfaces for agents
│   ├── components/
│   │   ├── jepa_module.py         # Implementation of JEPA Encoder, Predictor, Target Encoder
│   │   ├── latent_dynamics.py     # Neural network for learning state transitions (h_psi)
│   │   ├── goal_model.py          # Probabilistic model for inferring goals (p_beta(g|z_t))
│   │   ├── energy_function.py     # Defines the intrinsic energy calculation (E(z,g))
│   │   └── utils.py               # Utility functions (e.g., EMA updates, tensor operations)
│   ├── environments/
│   │   ├── gridworld.py           # GridWorld simulation environment
│   │   └── __init__.py            # Makes 'environments' a Python package
│   ├── planning/
│   │   ├── active_planner.py      # Orchestrates latent rollouts, energy evaluation, action selection
│   │   └── fake_goal_masking.py   # Logic for pruning implausible trajectories
│   └── training/
│       ├── jepa_trainer.py        # Script for pre-training the JEPA module
│       └── agent_trainer.py       # Main script for training the full unified agent
├── experiments/                   # Scripts and configurations for specific experiments
│   ├── ablation_study/
│   │   ├── run_ablation.py        # Script to run all ablation conditions from Appendix D
│   │   └── config.yaml            # Configuration file for ablation experiments (e.g., enable/disable modules)
│   │   └── results/               # Directory for storing logs, plots, and trained model checkpoints
│   ├── fake_goal_masking_demo/
│   │   └── run_demo.py            # Script for the illustrative simulation from Appendix E
│   └── custom_experiments/        # Placeholder for new experimental setups
├── data/                          # (Optional) Small datasets or scripts to download larger ones
├── notebooks/                     # Jupyter notebooks for interactive analysis and visualization
│   └── tutorial_analysis.ipynb    # Example notebook for analyzing simulation results
└── docs/                          # (Optional) Additional documentation or thesis chapters if included
```

## How to Run Experiments

All experiment scripts are designed to be run from the root directory of the repository.

### 1. Run the Fake Goal Masking Demonstration (Appendix E)

This script demonstrates the "fake goal masking" concept in a simplified GridWorld environment, illustrating how an agent can infer true intent despite deceptive movements.

```bash
python experiments/fake_goal_masking_demo/run_demo.py
```
This script will print simulation details and automatically display the generated plots (agent paths and belief updates).

### 2. Run the Ablation Studies (Appendix D)

This script allows you to run the various ablation conditions discussed in Appendix D, quantifying the contribution of each architectural component.

```bash
python experiments/ablation_study/run_ablation.py --config experiments/ablation_study/config.yaml
```
You can modify `config.yaml` to enable/disable specific modules for different ablation runs. The script will output performance metrics and potentially save plots to `experiments/ablation_study/results/`.

### 3. Training the Full Unified Agent

To train the complete unified agent framework, you would typically first pre-train the JEPA module, then fine-tune the entire agent.

*(Note: These are conceptual commands; the exact implementation will depend on the `jepa_trainer.py` and `agent_trainer.py` scripts.)*

```bash
# Step 1: Pre-train JEPA (if applicable)
python src/training/jepa_trainer.py --config_path configs/jepa_config.yaml

# Step 2: Train the full unified agent
python src/training/agent_trainer.py --config_path configs/agent_config.yaml --jepa_weights_path /path/to/pretrained_jepa_weights.pt
```

## Contributions and Future Work

This project represents a foundational step towards building genuinely autonomous and adaptive intelligent systems. Future work, as outlined in Chapter 9 of the dissertation, includes:

* **Hierarchical Memory and Temporal Abstraction:** Enhancing the agent's ability to reason over long timescales.
* **Language-Guided Reasoning and Neuro-Symbolic Integration:** Combining sub-symbolic representations with explicit symbolic reasoning.
* **Massive, Self-Supervised Generative World Models:** Scaling world model learning to enable rich, self-generated training data.
* **Highly Embodied, Continual, and Multi-Modal Interaction:** Extending the framework to robotics with diverse sensory streams (touch, smell, proprioception) for real-world adaptation.
* **Multi-Agent Coordination and Emergent Social Intelligence:** Exploring collective behaviors and distributed decision-making.
* **Refined Safety and Interpretability Mechanisms:** Continuously developing tools for auditing and aligning autonomous systems.

We welcome contributions and collaborations from the research community to further develop this framework. Please refer to the `CONTRIBUTING.md` (if available) for guidelines.

## Citation

If you find this work useful in your research, please consider citing the accompanying dissertation:

```
@phdthesis{seo2025unified,
  author       = {Seo, Hyungseok},
  title        = {From Representation to Reasoning: A Unified Framework via JEPA, Active Inference, and Developmental Principles},
  school       = {Chungbuk National University},
  year         = {2025},
  month        = {May} # Or your submission month
}
```
