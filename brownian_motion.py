"""
Brownian Motion Simulation - Pygame Visualization

This module simulates Brownian motion in 2D space with multiple particles (balls),
including a central "big ball" and numerous smaller particles that interact via
elastic collisions. The simulation uses Pygame for real-time visualization,
NumPy for numerical computations, and Numba for performance acceleration.
Particles bounce off walls and each other, with velocity updates based on
acceleration (simulating forces like gravity or random kicks).

Features
- Many particles (default 300) with elastic collisions (conserving momentum and energy).
- A central "big ball" that influences particle paths.
- Real-time Pygame rendering at 60 FPS.
- Velocity updates via acceleration for more realistic dynamics.
- Adjustable parameters like number of balls, dimensions, and delta_t.

Usage
1. Ensure dependencies are installed:
   - pygame, numpy, numba, pandas
   Example:
     pip install pygame numpy numba pandas

2. Run the script:
     python brownian_motion.py

   The simulation starts with a big ball in the center and smaller balls
   randomly placed, avoiding overlap. It runs until the window is closed.

Controls
- Close the window: Quit the simulation.
- No interactive controls; simulation runs automatically.

Performance notes
- Numba @jit accelerates collision and update functions for efficiency.
- Larger number_of_balls increases computation time; delta_t affects stability.
- Frame times are printed; for large N, consider reducing FPS or optimizing.

Author / License
- Educational simulation of Brownian motion and particle dynamics; adapt freely.
"""

import numpy as np
from numpy import random
import pandas as pd
from numba import jit
import os
import time
import pygame

number_of_balls = 300 # Number of balls in the simulation
dimensions = np.asarray([600,600])

delta_t = 5 # Update with the actual time step
#Bug Discovered. The particles go into the big ball if the timestep is bigger than 1

class Ball:
    """
    Class representing a ball in a 2D space.

    Attributes:
    - pos (numpy.ndarray): The position of the ball as a 2D numpy array [x, y].
    - vel (numpy.ndarray): The velocity of the ball as a 2D numpy array [vx, vy].
    - acc (numpy.ndarray): The acceleration of the ball as a 2D numpy array [ax, ay].

    Methods:
    - __init__(self, position, velocity, acceleration): Constructor to initialize the ball with
      a given position, velocity, and acceleration.
    - collide(self): Method to handle collisions with the walls and update the velocity accordingly.
    - update(self): Method to update the position, velocity, and handle collisions.

    Usage:
    1. Create an instance of the Ball class with initial position, velocity, and acceleration.
    2. Call the update method to simulate the motion of the ball, considering collisions.
    3. Access the attributes 'pos' and 'vel' to retrieve the current position and velocity.

    Note: This class assumes a 2D space, and collisions with the walls are handled by
          reversing the appropriate component of the velocity.

    Author: Sasi Mitra Behara
    Date: December 28, 2023
    """
    def __init__(self, position, velocity, acceleration, radius):
        # Taking Position, velocity, acceleration as inputs and making them object variables
        self.pos = np.asarray(position)
        self.vel = np.asarray(velocity)*5
        self.acc = acceleration
        self.radius = radius
        self.color = np.random.randint(0, 256, size=3)

    def update(self):
        """
        Update the position, velocity, and handle collisions.
        """
        self.check_wall_collision()
        self.pos, self.vel, self.acc =  updateParameters(self.pos, self.vel, self.acc)
    
    def check_wall_collision(self):
        '''Check if Collisions occurs with the walls and implements elastic collision with the walls'''
        self.vel = wall_collision(self.pos, self.vel, self.radius)

        
    def display(self):
        pygame.draw.circle(win, self.color, self.pos, self.radius)
        
        
@jit(nopython = True)       
def wall_collision(position, velocity, radius):
    # Find indices where position is less than or equal to 0 or greater than or equal to dimensions
    below_zero = position <= (0 + radius)
    above_dimensions = position >= (dimensions - radius)

    # Update velocities for the corresponding dimensions
    velocity[below_zero] = -velocity[below_zero]
    velocity[above_dimensions] = -velocity[above_dimensions]

    return velocity        

@jit(nopython=True)
def updateParameters(position, velocity, acceleration):
    velocity = velocity + acceleration * delta_t
    position = position + velocity * delta_t
    return position, velocity, acceleration

# Collision Logic for the Particles
@jit(nopython=True)
def collide(pos1, pos2, vel1, vel2, r1, r2):
    """
    Handle collision between two objects.

    Parameters:
    - pos1 (numpy.ndarray): Position of the first object.
    - pos2 (numpy.ndarray): Position of the second object.
    - vel1 (numpy.ndarray): Velocity of the first object.
    - vel2 (numpy.ndarray): Velocity of the second object.
    - r1 (float): Radius of the first object.
    - r2 (float): Radius of the second object.

    Returns:
    - numpy.ndarray: Updated velocity of the first object.
    - numpy.ndarray: Updated velocity of the second object.

    Collision Formula:
    - Calculate the distance between the two objects manually using Euclidean norm:
      r = np.sqrt(np.sum((pos1 - pos2)**2))

    - Check if a collision occurs (distance is less than or equal to COLLISION_DISTANCE):
      if r <= COLLISION_DISTANCE:
        - Update velocities using collision formula:
        vel1_new = vel1 - 2 * (np.sum((vel1 - vel2) * (pos1 - pos2)) / np.sum((pos1 - pos2)**2)) * (pos1 - pos2) * (r2**2 / (r1**2 + r2**2))
        vel2_new = vel2 - 2 * (np.sum((vel2 - vel1) * (pos2 - pos1)) / np.sum((pos2 - pos1)**2)) * (pos2 - pos1) * (r1**2 / (r1**2 + r2**2))

    The collision formula updates the velocities of two objects after a collision
    based on the conservation of linear momentum and kinetic energy.
    """
    
    # Named constant
    COLLISION_DISTANCE = (r1 + r2)**2

    # Calculate the distance between the two objects manually
    delta_pos = pos1 - pos2
    r = np.sum(delta_pos**2)

    # Check if a collision occurs (distance is less than or equal to COLLISION_DISTANCE)
    if r <= COLLISION_DISTANCE:
        # Update velocities using collision formula
        vel1_new = vel1 - 2 * (np.sum((vel1 - vel2) * delta_pos) / np.sum(delta_pos**2)) * delta_pos * (r2**2 / (r1**2 + r2**2))
        vel2_new = vel2 - 2 * (np.sum((vel2 - vel1) * (-delta_pos)) / np.sum(delta_pos**2)) * (-delta_pos) * (r1**2 / (r1**2 + r2**2))

        return vel1_new, vel2_new

    return vel1, vel2
  
@jit(nopython = True)        
def calculate_norm(array):
    return np.sqrt(np.sum(array**2))      
        
#########################The Drawing Function########################


def draw(frame_count): #This is the function we'll be using to draw each frame.
    win.fill((0,0,0))

    
    for i in range(number_of_balls):

        if i==(1-number_of_balls):
            pass
        else:
            for j in range(i+1, number_of_balls):
                a,b = collide(my_balls[i].pos, my_balls[j].pos, my_balls[i].vel, my_balls[j].vel, my_balls[i].radius, my_balls[j].radius)
                my_balls[i].vel = a
                my_balls[j].vel = b

        my_balls[i].update()
        
        my_balls[i].display()




def draw_path(frame_count): #This is the function we'll be using to draw each frame.
    
    
    for i in range(number_of_balls):

        if i==(1-number_of_balls):
            pass
        else:
            for j in range(i+1, number_of_balls):
                a,b = collide(my_balls[i].pos, my_balls[j].pos, my_balls[i].vel, my_balls[j].vel, my_balls[i].radius, my_balls[j].radius)
                my_balls[i].vel = a
                my_balls[j].vel = b

        my_balls[i].update()
        
        #my_balls[i].display()
    #my_balls[0].display()
    pygame.draw.circle(win, my_balls[0].color, my_balls[0].pos, 5)



################################Generating the Ball Objects ##################################

# Making a list called my_balls where the balls are stored
my_balls = []

big_ball_check = 0
# Iterating 'number_of_balls' times and making a new ball, and appending it to the list
for i in range(number_of_balls):
    if big_ball_check == 0:
        big_pos = dimensions/2
        vel = [0.0,0.0]
        acc = np.asarray([0,0])
        big_radius = 15
        big_ball = Ball(big_pos, vel, acc, big_radius) # Creating an instance of the big ball
        my_balls.append(big_ball)
        big_ball_check += 1
        mass = 0.1

    # Randomly generating position and velocity
    pos = [random.randint(0, dimensions[0]), random.randint(0, dimensions[1])]
    while calculate_norm(pos - big_pos) <= big_radius:
        pos = [random.randint(0, dimensions[0]), random.randint(0, dimensions[1])]
        

    vel = [random.random(), random.random()]
    acc = np.asarray([0,0])
    radius = 2
    mass = 0.05
    # Making a ball
    i_th_ball = Ball(pos, vel, acc, radius )  # Creating an instance of the Ball class
    
    # Appending the ball to the list
    my_balls.append(i_th_ball)


################################ The Actual Simulation ##################################

pygame.init() #Intializing the pygame engine


#Making a pygame window where we will be drawing stuff
win = pygame.display.set_mode(dimensions) 

'''fpsClock is basically a Clock that lets you control how fast the 
The program is going to be running'''
fpsClock = pygame.time.Clock() #Basically a clock.

run = True #Run is a variable that we can set to false when we want the window to close
frame_count = 0
while run: #As long as this while loop is true, we can see the window
    '''Pygame stores any interactions with the screen in a queue called 
	pygame.event.get(). We are going through that queue and looking for a specific
	event called pygame.QUIT, which is triggered when you press the x button in the
	window. When we set run to False, the while loop, and hence the window will 
	close'''

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        else:
            pass
    #Drawing to the screen each iteration
    frame_count += 1
    
    
    start_time = time.time()  # Record the start time for the frame generation
    draw(frame_count)
    #draw_path(frame_count) # This is for showing the path of the big ball


    end_time = time.time()  # Record the end time for the frame generation
  
    #Updating the display to reflect the changes made in the new frame
    pygame.display.flip()
    duration = end_time - start_time
    
    print(f"Frame number {frame_count + 1} has been generated in {duration:.4f} seconds.")
    #Telling the clock to pause the code for 1/60th of a second, and then go ahead
    fpsClock.tick(60)

#We're out of the while loop, so it assumes we're done with the code.
#Might as well close the engine we intialized at the start of the code.
pygame.quit()