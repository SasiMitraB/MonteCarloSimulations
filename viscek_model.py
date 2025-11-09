"""
Viscek Model - Pygame Simulation

This module implements a simple interactive Viscek-style flocking simulation
using Pygame for rendering, NumPy for numerical operations, and Numba to
accelerate compute-bound loops. Each agent ("arrow"/"bird") aligns its heading
with neighbors within an interaction radius, with optional noise and external
repulsion from the mouse cursor.

Features
- Real-time interactive simulation rendered with Pygame.
- Adjustable parameters via on-screen sliders:
  - Noise: random perturbation applied to headings.
  - Interaction radius: neighborhood radius for alignment.
- Mouse repulsion: agents are pushed away from the mouse when within a
  configurable repulsion radius and strength.
- Press SPACE to randomly scatter agents across the screen.
- Agents are represented by a rotated sprite (bird.png). The image is expected
  to be located next to this script or in a known resource folder.

Usage
1. Ensure dependencies are installed:
   - pygame, numpy, numba, pygame-widgets
   Example:
     pip install pygame numpy numba pygame-widgets

2. Place a 30x30 (or larger) RGBA bird sprite named "bird.png" in the same
   directory as this script (or adjust the path in Arrow.__init__).

3. Run the script:
     python viscek_model.py

Controls
- Mouse: move to create a repulsion field if the mouse is within the
  repulsion radius of agents.
- SPACE: scatter all agents randomly across the screen.
- Sliders: adjust noise and interaction radius at runtime.

Performance notes
- Numba @jit(nopython=True) is used for speed in the compute loops. The first
  run will incur compilation overhead; subsequent frames will be much faster.
- Large numbers of agents or very large interaction radii will increase CPU
  load due to pairwise distance calculations. Consider reducing agent count or
  increasing delta_t / reducing FPS for less frequent updates.

Author / License
- Minimal educational example; adapt freely for research or learning.
"""

import pygame
import sys
import numpy as np
import pygame_widgets
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox
import numba as nb
from numba import jit, prange

# Simulation Parameters
delta_t = 0.1
number_of_arrows = 200 # Number of arrows in the simulation
interaction_radius = 50  # Radius within which arrows influence each other
noise = 10  # Random noise in direction change
repulsion_strength = 100  # Strength of repulsion from mouse
repulsion_radius = 100  # Radius within which arrows are repelled by the mouse

# Numba accelerated functions
@jit(nopython=True)
def compute_directions(positions, angles, interaction_radius):
    n = len(positions)
    new_angles = np.zeros(n)
    
    for i in range(n):
        sin_sum = 0.0
        cos_sum = 0.0
        count = 0
        
        for j in range(n):
            # Calculate distance between arrows
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist < interaction_radius:
                angle_rad = np.radians(angles[j])
                sin_sum += np.sin(angle_rad)
                cos_sum += np.cos(angle_rad)
                count += 1
        
        if count > 0:
            avg_angle = np.degrees(np.arctan2(sin_sum/count, cos_sum/count))
            new_angles[i] = avg_angle
        else:
            new_angles[i] = angles[i]
    
    return new_angles

@jit(nopython=True)
def apply_mouse_repulsion(positions, velocities, mouse_pos, repulsion_radius, repulsion_strength, delta_t):
    for i in range(len(positions)):
        dx = positions[i, 0] - mouse_pos[0]
        dy = positions[i, 1] - mouse_pos[1]
        distance_to_mouse = np.sqrt(dx*dx + dy*dy)
        
        if distance_to_mouse < repulsion_radius and distance_to_mouse > 0:
            repulsion_x = dx / distance_to_mouse * repulsion_strength
            repulsion_y = dy / distance_to_mouse * repulsion_strength
            velocities[i, 0] += repulsion_x * delta_t
            velocities[i, 1] += repulsion_y * delta_t
    
    return velocities

@jit(nopython=True)
def update_positions(positions, velocities, delta_t, width, height):
    for i in range(len(positions)):
        positions[i, 0] = (positions[i, 0] + velocities[i, 0] * delta_t) % width
        positions[i, 1] = (positions[i, 1] + velocities[i, 1] * delta_t) % height
    
    return positions

##############################################################################################################
# Viscek Model
##############################################################################################################


class Arrow:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle
        self.radius = 10
        self.pos = np.asarray((self.x, self.y))
        self.vel = np.asarray([np.cos(np.radians(self.angle)), np.sin(np.radians(self.angle))]) * 20
        
        # Load the bird image
        self.original_image = pygame.image.load("bird.png").convert_alpha()
        # You might want to scale the image to an appropriate size
        self.original_image = pygame.transform.scale(self.original_image, (30, 30))
        self.image = self.original_image
        
    def display(self, surface):
        # Rotate the image according to the bird's angle
        self.image = pygame.transform.rotate(self.original_image, -self.angle)
        
        # Get the rect for positioning (centered on the bird's position)
        rect = self.image.get_rect(center=(self.x, self.y))
        
        # Draw the image onto the surface
        surface.blit(self.image, rect)

##############################################################################################################
# All the Pygame related code goes below this line
##############################################################################################################
# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1000, 1000
FPS = 60
BACKGROUND_COLOR = (30, 30, 30)  # Dark gray

# Create screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Viscek Model Simulation")

# Clock to control frame rate
clock = pygame.time.Clock()

# Create sliders
noise_slider = Slider(screen, 50, 30, 200, 20, min=0, max=50, step=1, initial=noise)
noise_text = TextBox(screen, 260, 30, 100, 20, fontSize=16)
noise_text.disable()  # Make it read-only
noise_label = TextBox(screen, 50, 10, 100, 20, fontSize=16, textColour=(255, 255, 255), colour=BACKGROUND_COLOR)
noise_label.setText("Noise")

radius_slider = Slider(screen, 50, 80, 200, 20, min=10, max=150, step=1, initial=interaction_radius)
radius_text = TextBox(screen, 260, 80, 100, 20, fontSize=16)
radius_text.disable()  # Make it read-only
radius_label = TextBox(screen, 50, 60, 150, 20, fontSize=16, textColour=(255, 255, 255), colour=BACKGROUND_COLOR)
radius_label.setText("Interaction Radius")

# Initialize the Objects
list_of_arrows = []
for i in range(number_of_arrows):
    x = np.random.randint(0, WIDTH)
    y = np.random.randint(0, HEIGHT)
    angle = np.random.randint(0, 360)
    arrow = Arrow(x, y, angle)
    list_of_arrows.append(arrow)

def update(screen, interaction_radius, noise):
    mouse_x, mouse_y = pygame.mouse.get_pos()
    mouse_pos = np.array([mouse_x, mouse_y], dtype=np.float64)
    
    # Extract positions, angles, and velocities for Numba processing
    positions = np.array([[arrow.x, arrow.y] for arrow in list_of_arrows], dtype=np.float64)
    angles = np.array([arrow.angle for arrow in list_of_arrows], dtype=np.float64)
    velocities = np.array([arrow.vel for arrow in list_of_arrows], dtype=np.float64)
    
    # Compute new directions with Numba
    new_angles = compute_directions(positions, angles, interaction_radius)
    
    # Apply noise
    new_angles += np.random.uniform(-noise, noise, len(new_angles))
    
    # Update velocities based on new angles
    for i, angle in enumerate(new_angles):
        velocities[i] = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))]) * 20
    
    # Apply mouse repulsion with Numba
    velocities = apply_mouse_repulsion(positions, velocities, mouse_pos, repulsion_radius, repulsion_strength, delta_t)
    
    # Update positions with Numba
    positions = update_positions(positions, velocities, delta_t, WIDTH, HEIGHT)
    
    # Update arrow objects
    for i, arrow in enumerate(list_of_arrows):
        arrow.x = positions[i, 0]
        arrow.y = positions[i, 1]
        arrow.pos = positions[i]
        arrow.angle = new_angles[i]
        arrow.vel = velocities[i]
        
        # Display arrow
        arrow.display(screen)


def scatter_arrows():
    """Randomly repositions all arrows on the screen."""
    for arrow in list_of_arrows:
        # Set random position
        arrow.x = np.random.randint(0, WIDTH)
        arrow.y = np.random.randint(0, HEIGHT)
        arrow.pos = np.array([arrow.x, arrow.y])
        
        # Set random angle
        arrow.angle = np.random.randint(0, 360)
        arrow.vel = np.array([np.cos(np.radians(arrow.angle)), np.sin(np.radians(arrow.angle))]) * 20
                            

def main():
    running = True
    while running:
        # Event handling
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    scatter_arrows()

        
        # Get slider values
        current_noise = noise_slider.getValue()
        current_radius = radius_slider.getValue()
        
        # Update text boxes
        noise_text.setText(f"{current_noise:.1f}")
        radius_text.setText(f"{current_radius:.1f}")
        
        # Drawing
        screen.fill(BACKGROUND_COLOR)
        update(screen, current_radius, current_noise)
        
        # Update widgets
        pygame_widgets.update(events)
        
        # Update display
        pygame.display.flip()
        
        # Cap the frame rate
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
