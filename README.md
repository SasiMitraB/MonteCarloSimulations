# Monte Carlo Simulations - Science Day 2025

This repository contains interactive physics simulations demonstrated during **Science Day 2025** at **IISER Tirupati**. These scripts illustrate fundamental concepts in statistical mechanics, stochastic processes, and collective behavior through real-time visualizations using Pygame.

## 📋 Contents

### 1. Random Walk (`random_walk.py`)
A 2D random walk simulation visualizing stochastic motion of a single particle.

**Features:**
- Single particle performing random walk in 2D space
- Real-time visualization using Pygame
- Elastic collisions with walls
- Random velocity updates simulating stochastic motion
- Path tracing visualization
- Performance-optimized with Numba JIT compilation

**Physics Concepts:** Random walk, stochastic processes, diffusion

### 2. Brownian Motion (`brownian_motion.py`)
Multi-particle simulation demonstrating Brownian motion with collision dynamics.

**Features:**
- 300 interacting particles with elastic collisions
- Central "big ball" influenced by surrounding particles
- Conservation of momentum and energy in collisions
- Velocity updates via acceleration for realistic dynamics
- Real-time rendering at 60 FPS
- Numba-accelerated collision detection

**Physics Concepts:** Brownian motion, kinetic theory, particle dynamics, momentum conservation

### 3. Ising Model (`ising_model.py`)
2D Ising model simulation using the Metropolis Monte Carlo algorithm.

**Features:**
- Interactive lattice visualization (red: +1 spin, blue: -1 spin)
- Real-time temperature and coupling constant control
- Live plots of energy per site and magnetization
- Phase transition observation
- Pause/resume and reset functionality
- Screenshot capture (press 'S')
- Command-line arguments for customization

**Controls:**
- `SPACE`: Pause/Resume
- `↑/↓`: Increase/Decrease temperature
- `←/→`: Adjust coupling constant J
- `R`: Reset lattice
- `S`: Save screenshot

**Physics Concepts:** Statistical mechanics, phase transitions, ferromagnetism, critical phenomena, Metropolis algorithm

**Usage:**
```bash
python ising_model.py [size] [temperature] [cell_size]
# Defaults: size=50, temperature=2.0, cell_size=10
```

### 4. Vicsek Model (`viscek_model.py`)
Flocking simulation demonstrating collective behavior and self-organization.

**Features:**
- Interactive flocking/swarming simulation
- Real-time adjustable parameters via sliders (noise, interaction radius)
- Mouse cursor repulsion field
- Agent alignment based on local neighborhood
- Bird sprite visualization with rotation
- Numba-accelerated computation

**Controls:**
- `SPACE`: Scatter agents randomly
- Mouse movement: Creates repulsion field
- Sliders: Adjust noise and interaction radius

**Physics Concepts:** Collective behavior, self-organization, active matter, emergence

## 🚀 Installation

### Prerequisites
All simulations require Python 3.7+ and the following dependencies:

```bash
pip install pygame numpy numba pandas pygame-widgets
```

### Individual Requirements
- **Random Walk & Brownian Motion:** `pygame`, `numpy`, `numba`, `pandas`
- **Ising Model:** `pygame`, `numpy`
- **Vicsek Model:** `pygame`, `numpy`, `numba`, `pygame-widgets`

**Note:** For the Vicsek model, ensure a `bird.png` sprite (30×30 pixels or larger) is placed in the same directory as the script.

## 🎮 Running the Simulations

Simply execute any script directly:

```bash
python random_walk.py
python brownian_motion.py
python ising_model.py
python viscek_model.py
```

Close the window to exit each simulation.

## 🎓 Educational Context

These simulations were designed to showcase:
- **Randomness in Physical Systems:** From microscopic particle motion to macroscopic phase transitions
- **Monte Carlo Methods:** Statistical sampling and importance sampling (Metropolis algorithm)
- **Emergence:** How simple local rules lead to complex collective behavior
- **Phase Transitions:** Critical phenomena in the Ising model
- **Computational Physics:** Real-time numerical simulations with performance optimization

## 📊 Performance Notes

- **Numba JIT compilation** accelerates compute-intensive loops (first run may be slower due to compilation)
- **Frame rate:** Most simulations target 30-60 FPS
- **Scalability:** Larger particle counts or lattice sizes increase computation time
- For optimal performance, adjust parameters like `number_of_balls`, `delta_t`, or lattice `size`

## 📄 License

This project is licensed under the terms specified in the `LICENSE` file. These educational simulations are provided for learning and demonstration purposes, and this repository is made to archive the code developed.

## 👥 Acknowledgments

Developed and presented during **Science Day 2025** at **IISER Tirupati**. This whole repository has been documented with AI, with a proofread and small additions wherever deemed nessecary.
