"""
Reaction-Diffusion System Visualization
========================================
An interactive toy model demonstrating the Gray-Scott reaction-diffusion system.
This produces fascinating patterns like spots, stripes, and labyrinthine structures.

Controls:
---------
- LEFT CLICK: Add chemical V (creates seed points for patterns)
- RIGHT CLICK: Add chemical U (erases patterns)
- SPACE: Pause/Resume simulation
- R: Reset simulation
- C: Clear and restart with new random seeds
- UP/DOWN: Adjust feed rate (f parameter)
- LEFT/RIGHT: Adjust kill rate (k parameter)
- 1-5: Load preset patterns (spots, stripes, waves, etc.)
- S: Save current frame as image
- Q/ESC: Quit

Mathematical Model:
-------------------
The Gray-Scott model describes two chemicals U and V:
∂U/∂t = Du∇²U - UV² + f(1-U)
∂V/∂t = Dv∇²V + UV² - (f+k)V

Where:
- Du, Dv: Diffusion rates
- f: Feed rate (replenishes U)
- k: Kill rate (removes V)
"""

import numpy as np
import pygame
from pygame import surfarray
import sys

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

class SimulationConfig:
    """Configuration for the reaction-diffusion simulation."""
    
    # Grid dimensions
    WIDTH = 400
    HEIGHT = 400
    
    # Display scale (each simulation cell becomes SCALE x SCALE pixels)
    SCALE = 2
    
    # Diffusion coefficients
    DU = 0.16  # Diffusion rate of U
    DV = 0.08  # Diffusion rate of V
    
    # Default reaction parameters (Gray-Scott)
    FEED_RATE = 0.055  # f: feed rate
    KILL_RATE = 0.062  # k: kill rate
    
    # Time step
    DT = 1.0
    
    # Steps per frame (more = faster simulation)
    STEPS_PER_FRAME = 8
    
    # Preset patterns (f, k) - different parameter combinations create different patterns
    PRESETS = {
        1: (0.055, 0.062, "Mitosis (spots)"),
        2: (0.039, 0.058, "Coral/Maze"),
        3: (0.026, 0.052, "Stripes"),
        4: (0.078, 0.061, "Waves"),
        5: (0.014, 0.047, "Spirals"),
    }


# ============================================================================
# COLORMAP
# ============================================================================

def create_colormap():
    """
    Create a high-contrast colormap for visualization.
    Maps chemical V concentration to colors.
    """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    
    for i in range(256):
        t = i / 255.0
        
        if t < 0.1:
            # Deep blue/black for low concentrations
            r = int(10 + 20 * (t / 0.1))
            g = int(10 + 30 * (t / 0.1))
            b = int(40 + 60 * (t / 0.1))
        elif t < 0.3:
            # Blue to cyan transition
            tt = (t - 0.1) / 0.2
            r = int(30 + 20 * tt)
            g = int(40 + 160 * tt)
            b = int(100 + 100 * tt)
        elif t < 0.5:
            # Cyan to green/yellow
            tt = (t - 0.3) / 0.2
            r = int(50 + 150 * tt)
            g = int(200 + 55 * tt)
            b = int(200 - 100 * tt)
        elif t < 0.7:
            # Yellow to orange
            tt = (t - 0.5) / 0.2
            r = int(200 + 55 * tt)
            g = int(255 - 80 * tt)
            b = int(100 - 50 * tt)
        elif t < 0.85:
            # Orange to red
            tt = (t - 0.7) / 0.15
            r = 255
            g = int(175 - 100 * tt)
            b = int(50 + 30 * tt)
        else:
            # Red to white/pink (high concentration)
            tt = (t - 0.85) / 0.15
            r = 255
            g = int(75 + 180 * tt)
            b = int(80 + 175 * tt)
        
        colormap[i] = [r, g, b]
    
    return colormap


def create_alternative_colormap():
    """
    Alternative colormap: purple to gold with high contrast.
    """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    
    for i in range(256):
        t = i / 255.0
        
        if t < 0.2:
            # Deep purple/black
            tt = t / 0.2
            r = int(15 + 25 * tt)
            g = int(5 + 10 * tt)
            b = int(30 + 50 * tt)
        elif t < 0.4:
            # Purple to magenta
            tt = (t - 0.2) / 0.2
            r = int(40 + 120 * tt)
            g = int(15 + 30 * tt)
            b = int(80 + 80 * tt)
        elif t < 0.6:
            # Magenta to orange
            tt = (t - 0.4) / 0.2
            r = int(160 + 95 * tt)
            g = int(45 + 120 * tt)
            b = int(160 - 110 * tt)
        elif t < 0.8:
            # Orange to yellow
            tt = (t - 0.6) / 0.2
            r = 255
            g = int(165 + 70 * tt)
            b = int(50 - 20 * tt)
        else:
            # Yellow to white
            tt = (t - 0.8) / 0.2
            r = 255
            g = int(235 + 20 * tt)
            b = int(30 + 225 * tt)
        
        colormap[i] = [r, g, b]
    
    return colormap


# ============================================================================
# REACTION-DIFFUSION SIMULATION
# ============================================================================

class ReactionDiffusion:
    """
    Gray-Scott Reaction-Diffusion System.
    
    Simulates two chemicals U and V that diffuse and react:
    - U is consumed and V is produced in the reaction U + 2V -> 3V
    - U is fed into the system at rate f
    - V decays at rate k
    """
    
    def __init__(self, width, height, config=None):
        self.width = width
        self.height = height
        self.config = config or SimulationConfig()
        
        # Chemical concentrations
        self.U = np.ones((height, width), dtype=np.float64)
        self.V = np.zeros((height, width), dtype=np.float64)
        
        # Parameters
        self.Du = self.config.DU
        self.Dv = self.config.DV
        self.f = self.config.FEED_RATE
        self.k = self.config.KILL_RATE
        self.dt = self.config.DT
        
        # Laplacian kernel for convolution (discrete Laplacian)
        self.laplacian_kernel = np.array([
            [0.05, 0.2, 0.05],
            [0.2, -1.0, 0.2],
            [0.05, 0.2, 0.05]
        ])
        
        # Initialize with some random seeds
        self._add_random_seeds(5)
    
    def _add_random_seeds(self, n_seeds):
        """Add random seed points of chemical V."""
        for _ in range(n_seeds):
            cx = np.random.randint(50, self.width - 50)
            cy = np.random.randint(50, self.height - 50)
            radius = np.random.randint(5, 15)
            self._add_circle(cx, cy, radius, chemical='V')
    
    def _add_circle(self, cx, cy, radius, chemical='V', value=1.0):
        """Add a circular region of chemical."""
        y, x = np.ogrid[:self.height, :self.width]
        mask = (x - cx)**2 + (y - cy)**2 <= radius**2
        
        if chemical == 'V':
            self.V[mask] = value
        else:
            self.U[mask] = value
    
    def _compute_laplacian(self, arr):
        """
        Compute the discrete Laplacian using convolution.
        Uses periodic boundary conditions (wrapping).
        """
        # Pad array with wrapped values for periodic boundaries
        padded = np.pad(arr, 1, mode='wrap')
        
        # Apply 3x3 Laplacian stencil
        laplacian = (
            0.05 * padded[:-2, :-2] +  # top-left
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
    
    def step(self):
        """
        Perform one time step of the simulation.
        
        Gray-Scott equations:
        ∂U/∂t = Du∇²U - UV² + f(1-U)
        ∂V/∂t = Dv∇²V + UV² - (f+k)V
        """
        # Compute Laplacians (diffusion terms)
        lap_U = self._compute_laplacian(self.U)
        lap_V = self._compute_laplacian(self.V)
        
        # Reaction term: UV²
        uvv = self.U * self.V * self.V
        
        # Update concentrations
        self.U += self.dt * (self.Du * lap_U - uvv + self.f * (1.0 - self.U))
        self.V += self.dt * (self.Dv * lap_V + uvv - (self.f + self.k) * self.V)
        
        # Clamp values to [0, 1]
        np.clip(self.U, 0, 1, out=self.U)
        np.clip(self.V, 0, 1, out=self.V)
    
    def add_chemical_at(self, x, y, radius=10, chemical='V'):
        """Add chemical at a specific position (for mouse interaction)."""
        self._add_circle(x, y, radius, chemical)
    
    def reset(self):
        """Reset to initial state."""
        self.U.fill(1.0)
        self.V.fill(0.0)
        self._add_random_seeds(5)
    
    def clear_with_seeds(self):
        """Clear and add new random seeds."""
        self.U.fill(1.0)
        self.V.fill(0.0)
        self._add_random_seeds(np.random.randint(3, 8))
    
    def set_parameters(self, f=None, k=None):
        """Update reaction parameters."""
        if f is not None:
            self.f = np.clip(f, 0.001, 0.1)
        if k is not None:
            self.k = np.clip(k, 0.001, 0.1)
    
    def get_visualization_array(self):
        """Get the V concentration array for visualization."""
        return self.V


# ============================================================================
# PYGAME VISUALIZATION
# ============================================================================

class Visualizer:
    """Real-time visualization using Pygame."""
    
    def __init__(self, simulation, config=None):
        self.sim = simulation
        self.config = config or SimulationConfig()
        
        # Initialize Pygame
        pygame.init()
        pygame.display.set_caption("Reaction-Diffusion System - Gray-Scott Model")
        
        # Set up display
        self.display_width = self.config.WIDTH * self.config.SCALE
        self.display_height = self.config.HEIGHT * self.config.SCALE
        self.screen = pygame.display.set_mode((self.display_width, self.display_height + 60))
        
        # Create surface for simulation
        self.sim_surface = pygame.Surface((self.config.WIDTH, self.config.HEIGHT))
        
        # Colormap
        self.colormap = create_colormap()
        self.alt_colormap = create_alternative_colormap()
        self.use_alt_colormap = False
        
        # Font for UI
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 20)
        
        # State
        self.running = True
        self.paused = False
        self.clock = pygame.time.Clock()
        self.frame_count = 0
    
    def _array_to_surface(self):
        """Convert simulation array to Pygame surface."""
        # Get V concentration and scale to 0-255
        data = self.sim.get_visualization_array()
        scaled = (np.clip(data, 0, 1) * 255).astype(np.uint8)
        
        # Apply colormap
        colormap = self.alt_colormap if self.use_alt_colormap else self.colormap
        rgb = colormap[scaled]
        
        # Transpose for Pygame (it expects width x height x 3)
        rgb_transposed = np.transpose(rgb, (1, 0, 2))
        
        # Blit to surface
        surfarray.blit_array(self.sim_surface, rgb_transposed)
    
    def _draw_ui(self):
        """Draw the user interface overlay."""
        # Background for UI
        ui_rect = pygame.Rect(0, self.display_height, self.display_width, 60)
        pygame.draw.rect(self.screen, (30, 30, 40), ui_rect)
        
        # Parameter display
        f_text = f"Feed (f): {self.sim.f:.4f}"
        k_text = f"Kill (k): {self.sim.k:.4f}"
        
        status = "PAUSED" if self.paused else "RUNNING"
        status_color = (255, 100, 100) if self.paused else (100, 255, 100)
        
        fps = self.clock.get_fps()
        fps_text = f"FPS: {fps:.1f}"
        
        # Render text
        f_surface = self.font.render(f_text, True, (200, 200, 255))
        k_surface = self.font.render(k_text, True, (255, 200, 200))
        status_surface = self.font.render(status, True, status_color)
        fps_surface = self.font.render(fps_text, True, (180, 180, 180))
        
        # Draw text
        self.screen.blit(f_surface, (10, self.display_height + 8))
        self.screen.blit(k_surface, (10, self.display_height + 32))
        self.screen.blit(status_surface, (self.display_width - 100, self.display_height + 8))
        self.screen.blit(fps_surface, (self.display_width - 100, self.display_height + 32))
        
        # Controls hint
        controls = "↑↓: f | ←→: k | 1-5: presets | SPACE: pause | R: reset | M: colormap"
        controls_surface = self.small_font.render(controls, True, (120, 120, 130))
        self.screen.blit(controls_surface, (180, self.display_height + 20))
    
    def _handle_events(self):
        """Handle input events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event.key)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_mouse_click(event)
        
        # Handle continuous mouse dragging
        if pygame.mouse.get_pressed()[0] or pygame.mouse.get_pressed()[2]:
            self._handle_mouse_drag()
    
    def _handle_keydown(self, key):
        """Handle keyboard input."""
        if key in (pygame.K_q, pygame.K_ESCAPE):
            self.running = False
        
        elif key == pygame.K_SPACE:
            self.paused = not self.paused
        
        elif key == pygame.K_r:
            self.sim.reset()
        
        elif key == pygame.K_c:
            self.sim.clear_with_seeds()
        
        elif key == pygame.K_m:
            self.use_alt_colormap = not self.use_alt_colormap
        
        elif key == pygame.K_UP:
            self.sim.set_parameters(f=self.sim.f + 0.002)
        
        elif key == pygame.K_DOWN:
            self.sim.set_parameters(f=self.sim.f - 0.002)
        
        elif key == pygame.K_RIGHT:
            self.sim.set_parameters(k=self.sim.k + 0.002)
        
        elif key == pygame.K_LEFT:
            self.sim.set_parameters(k=self.sim.k - 0.002)
        
        elif key == pygame.K_s:
            self._save_screenshot()
        
        # Preset patterns
        elif key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
            preset_num = key - pygame.K_0
            if preset_num in self.config.PRESETS:
                f, k, name = self.config.PRESETS[preset_num]
                self.sim.set_parameters(f=f, k=k)
                print(f"Loaded preset {preset_num}: {name} (f={f}, k={k})")
    
    def _handle_mouse_click(self, event):
        """Handle mouse click events."""
        x, y = event.pos
        
        # Check if click is in simulation area
        if y < self.display_height:
            # Convert to simulation coordinates
            sim_x = x // self.config.SCALE
            sim_y = y // self.config.SCALE
            
            if event.button == 1:  # Left click - add V
                self.sim.add_chemical_at(sim_x, sim_y, radius=8, chemical='V')
            elif event.button == 3:  # Right click - add U (erase)
                self.sim.add_chemical_at(sim_x, sim_y, radius=12, chemical='U')
    
    def _handle_mouse_drag(self):
        """Handle mouse dragging."""
        x, y = pygame.mouse.get_pos()
        
        if y < self.display_height:
            sim_x = x // self.config.SCALE
            sim_y = y // self.config.SCALE
            
            if pygame.mouse.get_pressed()[0]:  # Left drag - add V
                self.sim.add_chemical_at(sim_x, sim_y, radius=5, chemical='V')
            elif pygame.mouse.get_pressed()[2]:  # Right drag - add U
                self.sim.add_chemical_at(sim_x, sim_y, radius=8, chemical='U')
    
    def _save_screenshot(self):
        """Save current frame as image."""
        filename = f"reaction_diffusion_{self.frame_count:06d}.png"
        pygame.image.save(self.screen, filename)
        print(f"Saved screenshot: {filename}")
    
    def run(self):
        """Main visualization loop."""
        print("\n" + "="*60)
        print("Reaction-Diffusion System - Interactive Visualization")
        print("="*60)
        print("\nControls:")
        print("  LEFT CLICK / DRAG  - Add chemical V (create patterns)")
        print("  RIGHT CLICK / DRAG - Add chemical U (erase patterns)")
        print("  SPACE              - Pause/Resume")
        print("  R                  - Reset simulation")
        print("  C                  - Clear with new random seeds")
        print("  M                  - Toggle colormap")
        print("  UP/DOWN            - Adjust feed rate (f)")
        print("  LEFT/RIGHT         - Adjust kill rate (k)")
        print("  1-5                - Load preset patterns")
        print("  S                  - Save screenshot")
        print("  Q/ESC              - Quit")
        print("\nPresets:")
        for num, (f, k, name) in self.config.PRESETS.items():
            print(f"  {num}: {name} (f={f}, k={k})")
        print("="*60 + "\n")
        
        while self.running:
            # Handle events
            self._handle_events()
            
            # Update simulation
            if not self.paused:
                for _ in range(self.config.STEPS_PER_FRAME):
                    self.sim.step()
            
            # Render
            self._array_to_surface()
            
            # Scale and draw simulation
            scaled_surface = pygame.transform.scale(
                self.sim_surface, 
                (self.display_width, self.display_height)
            )
            self.screen.blit(scaled_surface, (0, 0))
            
            # Draw UI
            self._draw_ui()
            
            # Update display
            pygame.display.flip()
            
            # Frame rate control
            self.clock.tick(60)
            self.frame_count += 1
        
        pygame.quit()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    # Create configuration
    config = SimulationConfig()
    
    # Create simulation
    simulation = ReactionDiffusion(config.WIDTH, config.HEIGHT, config)
    
    # Create visualizer and run
    visualizer = Visualizer(simulation, config)
    visualizer.run()


if __name__ == "__main__":
    main()
