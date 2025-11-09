"""
Ising Model - Pygame Simulation

This module implements a 2D Ising model simulation using the Metropolis algorithm
for Monte Carlo updates, visualized in real-time with Pygame. The Ising model
represents spins on a lattice that interact ferromagnetically, exhibiting phase
transitions at critical temperatures. Spins are displayed as colored squares
(red for +1, blue for -1), with live plots of energy and magnetization.

Features
- Real-time lattice visualization with adjustable cell size.
- Interactive controls to modify temperature and coupling constant J.
- Live plotting of energy per site and magnetization over sweeps.
- Pause/resume, reset, and screenshot functionality.
- Command-line arguments for initial size, temperature, and cell size.

Usage
1. Ensure dependencies are installed:
   - pygame, numpy
   Example:
     pip install pygame numpy

2. Run the script with optional arguments:
     python ising_model.py [size] [temperature] [cell_size]
   Defaults: size=50, temperature=2.0, cell_size=10

Controls
- SPACE: Pause/Resume simulation.
- UP/DOWN: Increase/Decrease temperature (T).
- LEFT/RIGHT: Increase/Decrease coupling constant (J).
- S: Save a screenshot (PNG) to the current directory.
- R: Reset the lattice to a new random configuration.
- Close window: Quit the simulation.

Performance notes
- Simulation runs at ~30 FPS; larger lattices increase computation time per sweep.
- Plots update every sweep but are capped at 500 points for efficiency.
- For very large sizes, consider reducing cell_size or increasing update intervals.

Author / License
- Educational example for studying phase transitions; adapt freely.
"""

import numpy as np
import pygame
import sys

class IsingModel:
    def __init__(self, size=50, temperature=2.0):
        self.size = size
        self.temperature = temperature
        # Initialize lattice with random spins (+1 or -1)
        self.lattice = np.random.choice([-1, 1], size=(size, size))
        # Coupling constant
        self.J = 1.0
        
    def energy(self, i, j):
        """Calculate energy at site (i, j)"""
        s = self.lattice[i, j]
        neighbors = 0
        # Periodic boundary conditions
        neighbors += self.lattice[(i+1)%self.size, j]
        neighbors += self.lattice[(i-1)%self.size, j]
        neighbors += self.lattice[i, (j+1)%self.size]
        neighbors += self.lattice[i, (j-1)%self.size]
        return -self.J * s * neighbors
        
    def total_energy(self):
        """Calculate total energy of the lattice"""
        energy = 0
        for i in range(self.size):
            for j in range(self.size):
                energy += self.energy(i, j)
        # Avoid double counting
        return energy / 2
        
    def magnetization(self):
        """Calculate absolute magnetization of the lattice"""
        return np.abs(np.sum(self.lattice)) / (self.size * self.size)
        
    def metropolis_step(self):
        """Perform one step of Metropolis algorithm"""
        # Select a random site
        i, j = np.random.randint(0, self.size, 2)
        
        # Calculate energy change if we flip this spin
        delta_E = -2 * self.energy(i, j)
        
        # Flip if energy decreases or with probability exp(-delta_E/T)
        if delta_E <= 0 or np.random.random() < np.exp(-delta_E / self.temperature):
            self.lattice[i, j] *= -1
            return True
        return False
    
    def metropolis_sweep(self):
        """Perform a full sweep (NÂ²) of Metropolis steps"""
        flips = 0
        for _ in range(self.size * self.size):
            if self.metropolis_step():
                flips += 1
        return flips


class PyGamePlot:
    """A class to handle plotting data in Pygame"""
    def __init__(self, width, height, max_points=500, bg_color=(30, 30, 30)):
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height))
        self.bg_color = bg_color
        self.axis_color = (200, 200, 200)
        self.max_points = max_points
        
        # Margins for the plot area
        self.margin_left = 60
        self.margin_right = 20
        self.margin_top = 40
        self.margin_bottom = 40
        
        # Plot area dimensions
        self.plot_width = width - self.margin_left - self.margin_right
        self.plot_height = height - self.margin_top - self.margin_bottom
        
        # Font for labels
        self.font = pygame.font.SysFont("Arial", 14)
        self.title_font = pygame.font.SysFont("Arial", 18, bold=True)
        
    def draw_plot(self, data, title, x_label, y_label, color=(0, 255, 0), 
                  y_min=None, y_max=None, fixed_y_range=False):
        """Draw a line plot of the data on the surface"""
        # Clear the surface
        self.surface.fill(self.bg_color)
        
        # If no data, just draw the axes and return
        if not data:
            self._draw_axes(title, x_label, y_label, 0, 1, 0, 1)
            return self.surface
        
        # Calculate y-axis range
        if y_min is None or y_max is None or not fixed_y_range:
            calc_y_min = min(data)
            calc_y_max = max(data)
            # Ensure min != max to avoid division by zero
            if calc_y_min == calc_y_max:
                calc_y_min -= 0.1
                calc_y_max += 0.1
            # Add some padding
            padding = (calc_y_max - calc_y_min) * 0.1
            y_min = calc_y_min - padding if y_min is None else y_min
            y_max = calc_y_max + padding if y_max is None else y_max
        
        # Draw the axes
        self._draw_axes(title, x_label, y_label, 0, len(data), y_min, y_max)
        
        # Draw the line
        points = []
        for i, value in enumerate(data):
            # Convert data coordinates to screen coordinates
            x = self.margin_left + (i / len(data)) * self.plot_width
            # Invert y-axis (pygame's y increases downward)
            y = self.margin_top + self.plot_height - ((value - y_min) / (y_max - y_min)) * self.plot_height
            points.append((x, y))
            
        # Draw line segments connecting the points
        if len(points) > 1:
            pygame.draw.lines(self.surface, color, False, points, 2)
        
        return self.surface
    
    def _draw_axes(self, title, x_label, y_label, x_min, x_max, y_min, y_max):
        """Draw the axes, grid lines, and labels"""
        # Draw the title
        title_text = self.title_font.render(title, True, (255, 255, 255))
        title_rect = title_text.get_rect(center=(self.width // 2, self.margin_top // 2))
        self.surface.blit(title_text, title_rect)
        
        # Draw the x-axis
        pygame.draw.line(
            self.surface, self.axis_color,
            (self.margin_left, self.margin_top + self.plot_height),
            (self.margin_left + self.plot_width, self.margin_top + self.plot_height), 
            2
        )
        
        # Draw the y-axis
        pygame.draw.line(
            self.surface, self.axis_color,
            (self.margin_left, self.margin_top),
            (self.margin_left, self.margin_top + self.plot_height), 
            2
        )
        
        # Draw x-axis label
        x_label_text = self.font.render(x_label, True, (200, 200, 200))
        x_label_rect = x_label_text.get_rect(
            center=(self.margin_left + self.plot_width // 2, 
                   self.height - self.margin_bottom // 2)
        )
        self.surface.blit(x_label_text, x_label_rect)
        
        # Draw y-axis label
        y_label_text = self.font.render(y_label, True, (200, 200, 200))
        y_label_text = pygame.transform.rotate(y_label_text, 90)
        y_label_rect = y_label_text.get_rect(
            center=(self.margin_left // 2, 
                   self.margin_top + self.plot_height // 2)
        )
        self.surface.blit(y_label_text, y_label_rect)
        
        # Draw x ticks and grid lines (3 ticks)
        for i in range(4):
            x_pos = self.margin_left + (i / 3) * self.plot_width
            # Tick mark
            pygame.draw.line(
                self.surface, self.axis_color,
                (x_pos, self.margin_top + self.plot_height),
                (x_pos, self.margin_top + self.plot_height + 5), 
                1
            )
            # Grid line (lighter color)
            pygame.draw.line(
                self.surface, (100, 100, 100),
                (x_pos, self.margin_top + self.plot_height),
                (x_pos, self.margin_top), 
                1
            )
            # Tick label
            tick_value = int(x_min + (i / 3) * (x_max - x_min))
            if tick_value < x_max:  # Avoid showing a value beyond our data range
                tick_text = self.font.render(str(tick_value), True, (180, 180, 180))
                tick_rect = tick_text.get_rect(
                    center=(x_pos, self.margin_top + self.plot_height + 15)
                )
                self.surface.blit(tick_text, tick_rect)
        
        # Draw y ticks and grid lines (4 ticks)
        for i in range(5):
            y_pos = self.margin_top + self.plot_height - (i / 4) * self.plot_height
            # Tick mark
            pygame.draw.line(
                self.surface, self.axis_color,
                (self.margin_left, y_pos),
                (self.margin_left - 5, y_pos), 
                1
            )
            # Grid line (lighter color)
            pygame.draw.line(
                self.surface, (100, 100, 100),
                (self.margin_left, y_pos),
                (self.margin_left + self.plot_width, y_pos), 
                1
            )
            # Tick label
            tick_value = round(y_min + (i / 4) * (y_max - y_min), 2)
            tick_text = self.font.render(f"{tick_value:.2f}", True, (180, 180, 180))
            tick_rect = tick_text.get_rect(
                midright=(self.margin_left - 10, y_pos)
            )
            self.surface.blit(tick_text, tick_rect)


class IsingSimulation:
    def __init__(self, size=50, temperature=2.0, cell_size=10):
        # Initialize pygame
        pygame.init()
        self.cell_size = cell_size
        self.size = size
        
        # Calculate the main simulation width and the plot width
        self.sim_width = size * cell_size
        self.plot_width = 400  # Width for each plot
        self.plot_height = 200  # Height for each plot
        
        # Total window dimensions
        self.width = self.sim_width + self.plot_width + 20  # 20px margin between sim and plots
        self.height = max(self.sim_width, 2 * self.plot_height + 20)  # 20px margin between plots
        
        # Create the window
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"Ising Model Simulation (T={temperature})")
        
        # Colors
        self.bg_color = (20, 20, 20)
        self.spin_colors = {
            -1: (0, 0, 255),    # Blue for spin down
            1: (255, 0, 0)      # Red for spin up
        }
        
        # Create Ising model
        self.model = IsingModel(size=size, temperature=temperature)
        
        # For tracking energy and magnetization
        self.energies = [self.model.total_energy() / (self.size * self.size)]
        self.magnetizations = [self.model.magnetization()]
        self.max_data_points = 500  # Max number of points to show in plots
        
        # Create plot objects
        self.energy_plot = PyGamePlot(
            self.plot_width, self.plot_height, max_points=self.max_data_points
        )
        self.mag_plot = PyGamePlot(
            self.plot_width, self.plot_height, max_points=self.max_data_points
        )
        
        # Font for stats
        self.font = pygame.font.SysFont("Arial", 16)
        
        # For tracking when to update the plots
        self.update_interval = 1  # Update plots every N sweeps
        self.sweep_count = 0
        
        # For screenshot functionality
        self.save_dir = "./"
        
    def draw_lattice(self):
        """Draw the current state of the lattice"""
        # Create a surface for the lattice
        lattice_surface = pygame.Surface((self.sim_width, self.sim_width))
        lattice_surface.fill(self.bg_color)
        
        for i in range(self.size):
            for j in range(self.size):
                color = self.spin_colors[self.model.lattice[i, j]]
                pygame.draw.rect(
                    lattice_surface, 
                    color,
                    (j * self.cell_size, i * self.cell_size, self.cell_size, self.cell_size)
                )
        
        # Draw the lattice on the left side of the screen
        self.screen.blit(lattice_surface, (0, 0))
    
    def draw_stats(self):
        """Draw statistics on the screen"""
        energy = self.model.total_energy() / (self.size * self.size)
        mag = self.model.magnetization()
        
        # Store for plotting
        self.energies.append(energy)
        self.magnetizations.append(mag)
        
        # Limit the data points to show in plots
        if len(self.energies) > self.max_data_points:
            self.energies = self.energies[-self.max_data_points:]
            self.magnetizations = self.magnetizations[-self.max_data_points:]
        
        # Create text surfaces
        energy_text = self.font.render(f"Energy/site: {energy:.3f}", True, (255, 255, 255))
        mag_text = self.font.render(f"Magnetization: {mag:.3f}", True, (255, 255, 255))
        temp_text = self.font.render(f"Temperature: {self.model.temperature:.3f}", True, (255, 255, 255))
        j_text = self.font.render(f"Coupling J: {self.model.J:.3f}", True, (255, 255, 255))
        sweep_text = self.font.render(f"Sweeps: {self.sweep_count}", True, (255, 255, 255))
        
        # Position for stats display
        stats_x = 10
        stats_y = self.height - 100
        
        # Draw text at the bottom of the simulation area
        self.screen.blit(energy_text, (stats_x, stats_y))
        self.screen.blit(mag_text, (stats_x, stats_y + 20))
        self.screen.blit(temp_text, (stats_x, stats_y + 40))
        self.screen.blit(j_text, (stats_x, stats_y + 60))
        self.screen.blit(sweep_text, (stats_x, stats_y + 80))
        
        # Draw controls reminder
        controls_text = self.font.render("SPACE: Pause | UP/DOWN: Temp | LEFT/RIGHT: J | R: Reset | S: Screenshot", 
                                       True, (180, 180, 180))
        controls_rect = controls_text.get_rect(center=(self.width // 2, self.height - 15))
        self.screen.blit(controls_text, controls_rect)
    
    def update_plots(self):
        """Update the plots with current data"""
        # Draw the energy plot
        energy_surface = self.energy_plot.draw_plot(
            self.energies,
            "Energy per Site",
            "Sweeps",
            "Energy",
            color=(0, 255, 0),  # Green for energy
            y_min=-2.5,  # Typical range for 2D Ising model energy
            y_max=0.5
        )
        
        # Draw the magnetization plot
        mag_surface = self.mag_plot.draw_plot(
            self.magnetizations,
            "Magnetization",
            "Sweeps",
            "Mag",
            color=(0, 120, 255),  # Blue for magnetization
            y_min=0,
            y_max=1.1,
            fixed_y_range=True
        )
        
        # Draw the plots on the screen
        plot_x = self.sim_width + 20  # 20px margin
        self.screen.blit(energy_surface, (plot_x, 0))
        self.screen.blit(mag_surface, (plot_x, self.plot_height + 20))  # 20px margin between plots
        
    def save_screenshot(self):
        """Save a screenshot of the current state"""
        filename = f"ising_T{self.model.temperature:.1f}_J{self.model.J:.1f}_sweep{self.sweep_count}.png"
        pygame.image.save(self.screen, filename)
        print(f"Screenshot saved as {filename}")
        
    def run(self, max_steps=10000):
        """Run the simulation"""
        step = 0
        running = True
        paused = False
        
        # For tracking FPS
        clock = pygame.time.Clock()
        
        while running and step < max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_UP:
                        self.model.temperature += 0.1
                        pygame.display.set_caption(f"Ising Model Simulation (T={self.model.temperature:.1f})")
                    elif event.key == pygame.K_DOWN:
                        self.model.temperature = max(0.1, self.model.temperature - 0.1)
                        pygame.display.set_caption(f"Ising Model Simulation (T={self.model.temperature:.1f})")
                    elif event.key == pygame.K_RIGHT:
                        self.model.J += 0.1
                        pygame.display.set_caption(f"Ising Model Simulation (T={self.model.temperature:.1f}, J={self.model.J:.1f})")
                    elif event.key == pygame.K_LEFT:
                        self.model.J = max(0.1, self.model.J - 0.1)
                        pygame.display.set_caption(f"Ising Model Simulation (T={self.model.temperature:.1f}, J={self.model.J:.1f})")
                    elif event.key == pygame.K_s:
                        # Save a screenshot
                        self.save_screenshot()
                    elif event.key == pygame.K_r:
                        # Reset the lattice
                        self.model.lattice = np.random.choice([-1, 1], size=(self.size, self.size))
                        self.energies = [self.model.total_energy() / (self.size * self.size)]
                        self.magnetizations = [self.model.magnetization()]
                        self.sweep_count = 0
                        
            if not paused:
                self.model.metropolis_sweep()
                self.sweep_count += 1
                step += 1
                
            # Always update the display
            self.screen.fill(self.bg_color)
            self.draw_lattice()
            
            # Always update plots (they're much more efficient now)
            self.update_plots()
            
            # Draw stats
            self.draw_stats()
            
            # Draw a line separating simulation and plots
            pygame.draw.line(self.screen, (100, 100, 100), 
                            (self.sim_width + 10, 0), 
                            (self.sim_width + 10, self.height), 
                            2)
            
            pygame.display.flip()
            clock.tick(30)  # Cap at 30 FPS
            
        pygame.quit()


if __name__ == "__main__":
    # Get command-line arguments or use defaults
    size = 50
    temperature = 2.0
    cell_size = 10
    
    if len(sys.argv) > 1:
        size = int(sys.argv[1])
    if len(sys.argv) > 2:
        temperature = float(sys.argv[2])
    if len(sys.argv) > 3:
        cell_size = int(sys.argv[3])
        
    # Start simulation
    print("Ising Model Simulation")
    print("Controls:")
    print("  SPACE - Pause/Resume")
    print("  UP/DOWN - Increase/Decrease temperature")
    print("  LEFT/RIGHT - Decrease/Increase coupling constant J")
    print("  S - Save screenshot")
    print("  R - Reset the simulation with a new random lattice")
    
    sim = IsingSimulation(size=size, temperature=temperature, cell_size=cell_size)
    sim.run()