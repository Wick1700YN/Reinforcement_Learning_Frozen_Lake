# FrozenLake Q-Learning Agent (Gymnasium)

A Reinforcement Learning project that trains an agent using **Q-Learning** to solve the FrozenLake environment from Gymnasium.

The agent learns to navigate an 4×4 grid world, avoid holes, and reach the goal using trial-and-error learning.

---

## Project Overview

This project implements a **tabular Q-Learning algorithm** for the FrozenLake-v1 environment.

- Environment: FrozenLake-v1 (Gymnasium)
- Grid size: 4x4
- Algorithm: Q-Learning
- Policy: Epsilon-Greedy
- Model storage: Pickle (`.pkl` file)

### Key Features:
- Train once and reuse model automatically
- Saves trained Q-table
- Loads model if it already exists
- Optional visualization support
- Reward tracking and learning curve plot

---

## How It Works

### 1. Training Phase

- Q-table is initialized with zeros for all state-action pairs
- Agent explores the environment using epsilon-greedy policy
- Q-values are updated using the Bellman equation:

\[
Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma \max Q(s',a') - Q(s,a)\right]
\]

- The trained Q-table is saved as:
        frozen_lake_q_table.pkl

---

### 2. Testing Phase

- If a saved model exists, it is automatically loaded
- The agent uses the learned policy (no random exploration)
- Runs deterministic actions based on best Q-values
- No retraining is required

---


---

## Installation

Install required dependencies:

```bash
pip install gymnasium numpy matplotlib
Run Frozen_lake.py

