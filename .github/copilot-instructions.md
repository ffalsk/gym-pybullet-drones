# Gym-Pybullet-Drones Copilot Instructions

This repository is a Gymnasium-compatible PyBullet drone simulation environment for Reinforcement Learning (RL) and control. It supports single and multi-agent scenarios.

## 🏗 Project Architecture

- **`gym_pybullet_drones/envs/`**: Contains all environment classes.
  - **`BaseAviary.py`**: The core class handling PyBullet physics, drone dynamics, and rendering.
  - **`BaseRLAviary.py`**: Extends `BaseAviary` for RL, defining `action_space` and `observation_space` conformant to Gymnasium.
  - **`[Task]Aviary.py`**: Specific tasks (e.g., `HoverAviary`, `MultiHoverAviary`, `DroneShowAviary`). Inherit from `BaseRLAviary` for RL tasks.
- **`gym_pybullet_drones/control/`**: PID and other controllers (e.g., `DSLPIDControl`) for baseline comparisons or low-level control.
- **`gym_pybullet_drones/examples/`**: Scripts to run PID demos (`pid.py`) or RL training (`learn.py`).
- **`gym_pybullet_drones/utils/`**: Utilities including `Logger`, and critical `enums` (`DroneModel`, `Physics`, `ActionType`, `ObservationType`).

## 🧠 Core Concepts & Conventions

- **Environment Inheritance**: 
  - ALWAYS inherit from `BaseRLAviary` for new RL environments.
  - Implement `_computeReward()`, `_computeTerminated()`, `_computeTruncated()`, and `_computeInfo()`.
  
- **Configuration (Enums)**:
  - Use `ActionType` (`RPM`, `PID`, `VEL`, `ONE_D_RPM`, etc.) to define the control mode.
  - Use `ObservationType` (`KIN`, `RGB`) to define input features.
  - **Physics frequency (`pyb_freq`)** is typically 240Hz.
  - **Control frequency (`ctrl_freq`)** is typically lower (e.g., 30Hz or 48Hz) for RL.

- **Multi-Agent Support**:
  - The `num_drones` argument controls the number of agents.
  - State vectors (positions, velocities) are generally handled as `(NUM_DRONES, 12)` or similar arrays.
  - Step methods return dictionaries or arrays keyed/indexed by drone ID.

- **Coordinate System**:
  - Uses **PyBullet's default Z-up** coordinate system.
  - Quaternions are `[x, y, z, w]`.

## 🛠 Development Workflow

- **Installation**: `pip install -e .` (editable mode).
- **Running Examples**:
  - RL Training: `python gym_pybullet_drones/examples/learn.py`
  - PID Control: `python gym_pybullet_drones/examples/pid.py`
- **Testing**: Run `pytest tests/` from the root.

## ⚠️ Critical Implementation Details

1.  **Stable Baselines 3 Integration**:
    - The environments are designed to work with `stable_baselines3`.
    - Use `make_vec_env` for creating vectorized environments for training.
    
2.  **Physics vs. Control Steps**:
    - `BaseAviary.step()` advances the simulation by `int(PYB_FREQ/CTRL_FREQ)` physics steps for every 1 control step.
    - Do NOT manually loop `p.stepSimulation()` inside a `step()` method unless implementing custom sub-stepping logic.

3.  **Action/Observation Normalization**:
    - RL actions are typically clipped to `[-1, 1]` in `BaseRLAviary` and converted to PWM/RPM internally.
    - Ensure custom environments handle `UserDebugGUI` updates if `gui=True`.
