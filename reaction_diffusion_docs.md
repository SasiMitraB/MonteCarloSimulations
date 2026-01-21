# Reaction-Diffusion Systems: Theory and Implementation

## Table of Contents
1. [Introduction](#introduction)
2. [Historical Background](#historical-background)
3. [Mathematical Foundation](#mathematical-foundation)
4. [The Gray-Scott Model](#the-gray-scott-model)
5. [Numerical Implementation](#numerical-implementation)
6. [Pattern Formation and Parameter Space](#pattern-formation-and-parameter-space)
7. [Code Architecture](#code-architecture)
8. [References](#references)

---

## Introduction

Reaction-diffusion systems are mathematical models that describe how the concentration of one or more substances distributed in space changes under the influence of two processes:

1. **Chemical reactions** - substances are transformed into each other
2. **Diffusion** - substances spread out over a surface

These systems are remarkable because they can spontaneously generate complex spatial patterns from nearly uniform initial conditions—a phenomenon known as **Turing instability** or **diffusion-driven instability**.

### Real-World Applications

Reaction-diffusion systems appear throughout nature and science:

- **Biology**: Animal coat patterns (zebra stripes, leopard spots), morphogenesis, neural signal propagation
- **Chemistry**: Belousov-Zhabotinsky reactions, catalytic surface reactions
- **Ecology**: Population dynamics, vegetation patterns in semi-arid regions
- **Physics**: Semiconductor physics, plasma dynamics

---

## Historical Background

### Alan Turing's Morphogenesis Paper (1952)

The theoretical foundation was laid by **Alan Turing** in his seminal 1952 paper *"The Chemical Basis of Morphogenesis"*. Turing proposed that a system of chemical substances (which he called **morphogens**) reacting together and diffusing through tissue could account for the main phenomena of morphogenesis—the development of patterns and shapes in organisms.

Turing's key insight was counterintuitive: **diffusion, which normally acts to smooth out concentration differences, can actually destabilize a uniform steady state and lead to pattern formation** when combined with appropriate reaction kinetics.

### The Gray-Scott Model (1983-1984)

The Gray-Scott model, developed by Peter Gray and Stephen Scott in the 1980s, is a specific reaction-diffusion system that exhibits an extraordinarily rich variety of patterns depending on its parameters. It has become one of the most studied models for pattern formation due to its simplicity and the diversity of patterns it can produce.

---

## Mathematical Foundation

### General Reaction-Diffusion Equations

A general reaction-diffusion system for two chemicals $U$ and $V$ is described by coupled partial differential equations (PDEs):

$$\frac{\partial U}{\partial t} = D_U \nabla^2 U + f(U, V)$$

$$\frac{\partial V}{\partial t} = D_V \nabla^2 V + g(U, V)$$

Where:
- $U(x, y, t)$ and $V(x, y, t)$ are the concentrations of the two chemicals
- $D_U$ and $D_V$ are the **diffusion coefficients**
- $\nabla^2$ is the **Laplacian operator** (measures local curvature of concentration)
- $f(U, V)$ and $g(U, V)$ are the **reaction terms**

### The Laplacian Operator

In two dimensions, the Laplacian is defined as:

$$\nabla^2 U = \frac{\partial^2 U}{\partial x^2} + \frac{\partial^2 U}{\partial y^2}$$

The Laplacian at a point measures how much the value at that point differs from the average of its neighbors. It is the key operator in diffusion:

- If $\nabla^2 U > 0$: The concentration at this point is **lower** than the local average → substance flows **in**
- If $\nabla^2 U < 0$: The concentration at this point is **higher** than the local average → substance flows **out**

### Turing Instability Conditions

For a reaction-diffusion system to exhibit pattern formation through Turing instability, several conditions must be met:

1. **Stable homogeneous equilibrium**: Without diffusion, the system settles to a uniform steady state
2. **Activator-inhibitor dynamics**: One chemical (activator) promotes both itself and the other; the other (inhibitor) suppresses both
3. **Differential diffusion**: The inhibitor must diffuse faster than the activator ($D_V > D_U$ or vice versa, depending on formulation)

The mathematical condition for Turing instability involves analyzing the eigenvalues of the Jacobian matrix of the linearized system.

---

## The Gray-Scott Model

### Chemical Reactions

The Gray-Scott model describes an autocatalytic chemical reaction in a continuously stirred tank reactor. The chemical reactions are:

$$U + 2V \rightarrow 3V \quad \text{(autocatalytic conversion)}$$

$$V \rightarrow P \quad \text{(decay of V to inert product P)}$$

Additionally:
- Chemical $U$ is continuously **fed** into the system
- Chemical $V$ is continuously **removed** (along with the inert product)

### Governing Equations

The Gray-Scott model PDEs are:

$$\frac{\partial U}{\partial t} = D_U \nabla^2 U - UV^2 + f(1 - U)$$

$$\frac{\partial V}{\partial t} = D_V \nabla^2 V + UV^2 - (f + k)V$$

Let's break down each term:

#### For Chemical U:
| Term | Meaning |
|------|---------|
| $D_U \nabla^2 U$ | Diffusion of U |
| $-UV^2$ | Consumption of U in the autocatalytic reaction (one U reacts with two V) |
| $f(1-U)$ | Feed term: U is replenished toward concentration 1 at rate $f$ |

#### For Chemical V:
| Term | Meaning |
|------|---------|
| $D_V \nabla^2 V$ | Diffusion of V |
| $+UV^2$ | Production of V from the autocatalytic reaction |
| $-(f+k)V$ | Removal: V is killed at rate $k$ and also washed out at rate $f$ |

### Parameters

| Parameter | Symbol | Typical Range | Physical Meaning |
|-----------|--------|---------------|------------------|
| Diffusion of U | $D_U$ | 0.16-0.21 | How fast U spreads |
| Diffusion of V | $D_V$ | 0.08-0.11 | How fast V spreads (usually $D_V < D_U$) |
| Feed rate | $f$ | 0.01-0.10 | Rate of U replenishment |
| Kill rate | $k$ | 0.04-0.07 | Rate of V removal |

### Steady States

The system has several equilibrium points:

1. **Trivial steady state**: $(U, V) = (1, 0)$ — no V present, U at maximum
2. **Non-trivial steady states**: Found by solving $f(1-U) = UV^2$ and $(f+k)V = UV^2$

The stability and nature of patterns depend critically on the $(f, k)$ parameter values.

---

## Numerical Implementation

### Discretization

To simulate the continuous PDEs on a computer, we discretize space and time:

- **Space**: A 2D grid of size $N_x \times N_y$ with spacing $\Delta x = \Delta y = 1$
- **Time**: Discrete steps of size $\Delta t$

### Discrete Laplacian

The Laplacian is approximated using a **9-point stencil** (weighted average of neighbors):

$$\nabla^2 U_{i,j} \approx \sum_{m,n} w_{m,n} \cdot U_{i+m, j+n} - U_{i,j}$$

The weights used in our implementation:

```
┌───────┬───────┬───────┐
│ 0.05  │  0.2  │ 0.05  │
├───────┼───────┼───────┤
│  0.2  │ -1.0  │  0.2  │
├───────┼───────┼───────┤
│ 0.05  │  0.2  │ 0.05  │
└───────┴───────┴───────┘
```

This 9-point stencil provides better isotropy (rotational symmetry) than the simpler 5-point stencil:

**5-point stencil** (simpler but less isotropic):
$$\nabla^2 U_{i,j} \approx U_{i+1,j} + U_{i-1,j} + U_{i,j+1} + U_{i,j-1} - 4U_{i,j}$$

**9-point stencil** (better isotropy):
$$\nabla^2 U_{i,j} \approx 0.2(U_{i+1,j} + U_{i-1,j} + U_{i,j+1} + U_{i,j-1}) + 0.05(U_{i+1,j+1} + U_{i+1,j-1} + U_{i-1,j+1} + U_{i-1,j-1}) - U_{i,j}$$

### Time Integration (Euler Method)

We use the explicit Euler method for time stepping:

$$U^{n+1}_{i,j} = U^n_{i,j} + \Delta t \left[ D_U \nabla^2 U^n_{i,j} - U^n_{i,j}(V^n_{i,j})^2 + f(1 - U^n_{i,j}) \right]$$

$$V^{n+1}_{i,j} = V^n_{i,j} + \Delta t \left[ D_V \nabla^2 V^n_{i,j} + U^n_{i,j}(V^n_{i,j})^2 - (f + k)V^n_{i,j} \right]$$

### Boundary Conditions

We use **periodic boundary conditions** (wrapping), meaning:
- The left edge connects to the right edge
- The top edge connects to the bottom edge

This creates a toroidal topology and avoids edge artifacts.

### Algorithm Pseudocode

```
Initialize:
    U[all] = 1.0  (uniform concentration)
    V[all] = 0.0  (no V initially)
    Add small patches of V as "seeds"

For each time step:
    1. Compute Laplacian of U: lap_U = convolve(U, laplacian_kernel)
    2. Compute Laplacian of V: lap_V = convolve(V, laplacian_kernel)
    3. Compute reaction term: uvv = U * V * V
    4. Update U: U += dt * (Du * lap_U - uvv + f * (1 - U))
    5. Update V: V += dt * (Dv * lap_V + uvv - (f + k) * V)
    6. Clamp values to [0, 1]
    7. Render V concentration as colors
```

### Stability Considerations

The explicit Euler method has a stability constraint. For the diffusion equation, the CFL (Courant-Friedrichs-Lewy) condition requires:

$$\Delta t \leq \frac{(\Delta x)^2}{4 \cdot \max(D_U, D_V)}$$

With our parameters ($D_U = 0.16$, $\Delta x = 1$), this gives $\Delta t \leq 1.56$. We use $\Delta t = 1.0$ to be safe.

---

## Pattern Formation and Parameter Space

### The $(f, k)$ Parameter Space

The Gray-Scott model exhibits a rich variety of patterns depending on the feed rate $f$ and kill rate $k$. The parameter space can be divided into regions:

```
         k (kill rate) →
    0.04   0.05   0.06   0.07
    ┌──────┬──────┬──────┬──────┐
0.08│      │Waves │      │      │  ↑
    ├──────┼──────┼──────┼──────┤  │
0.06│Stripe│Spots │      │Death │  f
    ├──────┼──────┼──────┼──────┤  │
0.04│Maze  │Coral │      │      │(feed)
    ├──────┼──────┼──────┼──────┤  │
0.02│Spiral│Worms │      │      │  │
    └──────┴──────┴──────┴──────┘  ↓
```

### Pattern Types

| Pattern | f | k | Description |
|---------|---|---|-------------|
| **Mitosis/Spots** | 0.055 | 0.062 | Self-replicating spots that divide like cells |
| **Coral/Maze** | 0.039 | 0.058 | Labyrinthine structures, coral-like growth |
| **Stripes** | 0.026 | 0.052 | Parallel stripe patterns |
| **Waves** | 0.078 | 0.061 | Propagating wave fronts |
| **Spirals** | 0.014 | 0.047 | Rotating spiral waves |
| **Worms** | 0.025 | 0.060 | Meandering worm-like structures |

### Physical Intuition

- **High $f$**: U is replenished quickly → more fuel for reactions → more V production possible
- **High $k$**: V is removed quickly → patterns decay faster → harder to sustain
- **Low $f$, Low $k$**: Slow dynamics, complex persistent patterns
- **High $f$, High $k$**: Fast dynamics, patterns may die out or show rapid changes

### Wavelength Selection

The characteristic wavelength $\lambda$ of patterns is approximately:

$$\lambda \sim 2\pi \sqrt{\frac{D_U + D_V}{f}}$$

This explains why patterns become finer (smaller wavelength) at higher feed rates.

---

## Code Architecture

### Class Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    SimulationConfig                         │
│  - Grid dimensions, diffusion coefficients                  │
│  - Reaction parameters, preset patterns                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   ReactionDiffusion                         │
│  - U, V concentration arrays (numpy)                        │
│  - step(): Compute one time step                            │
│  - _compute_laplacian(): 9-point stencil convolution        │
│  - add_chemical_at(): Interactive seeding                   │
│  - reset(), clear_with_seeds(): State management            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Visualizer                             │
│  - Pygame display and rendering                             │
│  - Colormap application                                     │
│  - Event handling (keyboard, mouse)                         │
│  - UI overlay (parameters, FPS, controls)                   │
└─────────────────────────────────────────────────────────────┘
```

### Key Implementation Details

#### Efficient Laplacian Computation

```python
def _compute_laplacian(self, arr):
    # Pad with wrapped values for periodic boundaries
    padded = np.pad(arr, 1, mode='wrap')
    
    # Vectorized 9-point stencil
    laplacian = (
        0.05 * padded[:-2, :-2] +   # top-left
        0.2  * padded[:-2, 1:-1] +  # top
        0.05 * padded[:-2, 2:] +    # top-right
        0.2  * padded[1:-1, :-2] +  # left
        -1.0 * padded[1:-1, 1:-1] + # center
        0.2  * padded[1:-1, 2:] +   # right
        0.05 * padded[2:, :-2] +    # bottom-left
        0.2  * padded[2:, 1:-1] +   # bottom
        0.05 * padded[2:, 2:]       # bottom-right
    )
    return laplacian
```

#### Colormap Design

The colormap is designed for high contrast and perceptual uniformity:

```
Concentration:  0% ────────────────────────────── 100%
Color:          Deep Blue → Cyan → Yellow → Orange → Red → White
```

This allows easy visual identification of:
- Low V regions (dark blue/black)
- Transition zones (cyan/green)
- High V regions (orange/red/white)

#### Performance Optimizations

1. **NumPy vectorization**: All array operations are vectorized (no Python loops)
2. **In-place operations**: Using `np.clip(..., out=arr)` to avoid memory allocation
3. **Multiple steps per frame**: Running 8 simulation steps between renders
4. **Surface reuse**: Reusing Pygame surfaces instead of creating new ones

---

## References

### Original Papers

1. **Turing, A.M.** (1952). "The Chemical Basis of Morphogenesis." *Philosophical Transactions of the Royal Society of London B*, 237(641), 37-72.

2. **Gray, P., & Scott, S.K.** (1983). "Autocatalytic reactions in the isothermal, continuous stirred tank reactor: Isolas and other forms of multistability." *Chemical Engineering Science*, 38(1), 29-43.

3. **Gray, P., & Scott, S.K.** (1984). "Autocatalytic reactions in the isothermal, continuous stirred tank reactor: Oscillations and instabilities in the system A + 2B → 3B; B → C." *Chemical Engineering Science*, 39(6), 1087-1097.

4. **Pearson, J.E.** (1993). "Complex patterns in a simple system." *Science*, 261(5118), 189-192.

### Books and Reviews

5. **Murray, J.D.** (2003). *Mathematical Biology II: Spatial Models and Biomedical Applications*. Springer.

6. **Cross, M.C., & Hohenberg, P.C.** (1993). "Pattern formation outside of equilibrium." *Reviews of Modern Physics*, 65(3), 851.

### Online Resources

7. **Karl Sims' Reaction-Diffusion Tutorial**: [karlsims.com/rd.html](http://karlsims.com/rd.html)

8. **Robert Munafo's Gray-Scott Explorer**: [mrob.com/pub/comp/xmorphia/](http://mrob.com/pub/comp/xmorphia/)

---

## Appendix: Mathematical Derivations

### A. Linear Stability Analysis

To understand when patterns form, we linearize the system around the homogeneous steady state $(U_0, V_0)$:

Let $u = U - U_0$ and $v = V - V_0$ be small perturbations.

The linearized equations are:

$$\frac{\partial}{\partial t}\begin{pmatrix} u \\ v \end{pmatrix} = \begin{pmatrix} D_U \nabla^2 + f_{U} & f_{V} \\ g_{U} & D_V \nabla^2 + g_{V} \end{pmatrix} \begin{pmatrix} u \\ v \end{pmatrix}$$

Where $f_U, f_V, g_U, g_V$ are partial derivatives of the reaction terms evaluated at the steady state.

For the Gray-Scott model at $(U_0, V_0) = (1, 0)$:
- $f_U = -f$
- $f_V = 0$
- $g_U = 0$
- $g_V = -(f+k)$

Looking for solutions of the form $e^{\sigma t + i\mathbf{k}\cdot\mathbf{r}}$ with wavenumber $|\mathbf{k}| = q$:

$$\sigma = \frac{1}{2}\left[(f_U + g_V) - (D_U + D_V)q^2 \pm \sqrt{[(f_U - g_V) - (D_U - D_V)q^2]^2 + 4f_V g_U}\right]$$

Pattern formation occurs when $\text{Re}(\sigma) > 0$ for some $q > 0$.

### B. Derivation of Characteristic Wavelength

At the onset of instability, the most unstable wavenumber $q_c$ satisfies:

$$q_c^2 = \sqrt{\frac{f_U g_V - f_V g_U}{D_U D_V}}$$

The characteristic wavelength is then:

$$\lambda_c = \frac{2\pi}{q_c}$$

For typical Gray-Scott parameters, this gives patterns with wavelength on the order of 10-50 grid cells.

---

*Document created for the Monte Carlo Simulations project.*
*Implementation: `reaction_diffusion.py`*
