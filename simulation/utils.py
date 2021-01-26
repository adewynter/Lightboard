# For Macs, uncomment if imageio gives you issues.:
#import matplotlib
# matplotlib.use("TkAgg")

import random
import torch
import imageio

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


def set_seeds(seed=123):
    """
    Set seeds for reproducibility. This will also help 
    explore other solutions to the same configuration.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def generate_positions(number_of_agents, environment_size_x, environment_size_y, environment, total_timestep, resource_max):
    """
    Define a function that generating new positions
    """
    position_x = np.zeros((number_of_agents, total_timestep))
    position_y = np.zeros((number_of_agents, total_timestep))
    position_x_coords = []
    position_y_coords = []
    for dummy_a in range(number_of_agents):
        position_x[dummy_a,
                   0] = np.random.uniform(-0.5, environment_size_x-0.5)
        position_y[dummy_a,
                   0] = np.random.uniform(-0.5, environment_size_y-0.5)
        while (environment[int(round(position_x[dummy_a][0]))]
               [int(round(position_y[dummy_a][0]))] != resource_max):
            position_x[dummy_a, 0] = \
                np.random.uniform(-0.5, environment_size_x-0.5)
            position_y[dummy_a, 0] = \
                np.random.uniform(-0.5, environment_size_y-0.5)
        position_x_coords.append(position_x[dummy_a, 0])
        position_y_coords.append(position_y[dummy_a, 0])
    return position_x, position_y, position_x_coords, position_y_coords


def render_movie(timedraw, number_of_agents, position_x, position_y,
                 environment, environment_size_x, environment_size_y, agent_radius,
                 resource_max, img_path="./output/Field_Drive.gif"):
    """
    Render an array as a gif.
    """
    whole_system = []
    for dummy_Td in tqdm(range(0, timedraw.shape[0]), desc="Rendering..."):
        system = generate_wholesystem(timedraw[dummy_Td],
                                      number_of_agents,
                                      position_x,
                                      position_y,
                                      environment[:, :, timedraw[dummy_Td]],
                                      environment_size_x,
                                      environment_size_y,
                                      agent_radius,
                                      resource_max)
        whole_system.append(system)

    imageio.mimsave(img_path, whole_system, fps=10)


def generate_wholesystem(timedraw, number_of_agents, position_x, position_y,
                         environment, environment_size_x, environment_size_y, agent_radius,
                         resource_max):
    """
    Draw the whole system, agents and the environment landscape
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(environment, cmap='gray', vmin=0, vmax=resource_max)
    for dummy_a in range(0, number_of_agents):
        y, x = generate_circle(position_x[dummy_a][timedraw],
                               position_y[dummy_a][timedraw],
                               agent_radius)
        ax.fill(x, y, facecolor='w', edgecolor='k')
        ax.axis('scaled')
        ax.set_xlim(-0.5, environment_size_x-0.5)
        ax.set_ylim(-0.5, environment_size_y-0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title('Random Walk Simulation' +
                  '\n at time = '+str(timedraw)+'', fontsize=32)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image


def generate_circle(position_x, position_y, radius):
    """
    Draw a beautiful circle
    """
    number_of_angle = 20
    theta = np.linspace(0, 2*np.pi, number_of_angle)
    x = position_x+radius*np.cos(theta)
    y = position_y+radius*np.sin(theta)
    return x, y
