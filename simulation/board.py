import os
import gym
import math
import random

import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict
from copy import deepcopy
from matplotlib import cm
from PIL import Image


IMPASSABLE_FOOD = 0  # TODO: does this make sense?


def update_shadow(position_x, position_y, shadow_in,
                  environment, environment_size_x, environment_size_y,
                  delta_time, relaxation_time, agent_influence_strength,
                  agent_influence_radius, number_of_agents, sigma):
    """
    Define the update rule for shadow
    """
    shadow_out = shadow_in*np.exp(-delta_time/relaxation_time)
    for dummy_a in range(0, number_of_agents):
        highest_x = int(
            np.round(position_x[dummy_a]+sigma*agent_influence_radius))
        lowest_x = int(
            np.round(position_x[dummy_a] - sigma*agent_influence_radius))
        highest_y = int(
            np.round(position_y[dummy_a]+sigma*agent_influence_radius))
        lowest_y = int(
            np.round(position_y[dummy_a] - sigma*agent_influence_radius))
        for dummy_x in range(max(0, lowest_x),
                             min(highest_x, environment_size_x)):
            for dummy_y in range(max(0, lowest_y),
                                 min(highest_y, environment_size_y)):
                dummy_r = np.sqrt((dummy_x-position_x[dummy_a])**2 +
                                  (dummy_y-position_y[dummy_a])**2)
                shadow_out[dummy_x, dummy_y] = \
                    shadow_out[dummy_x, dummy_y]+agent_influence_strength *\
                    np.exp(-dummy_r**2/(2*agent_influence_radius**2)) *\
                    delta_time
                if shadow_out[dummy_x, dummy_y] > environment[dummy_x, dummy_y]:
                    shadow_out[dummy_x,
                               dummy_y] = environment[dummy_x, dummy_y]
    return shadow_out


def get_velocity(position_x, position_y, agent_gradient_sensitivity, environment_influenced, agent_radius):
    """
    Get the velocity from agent response to the environment influenced
    """
    center_x = int(np.round(position_x))
    center_y = int(np.round(position_y))
    velocity_x = agent_gradient_sensitivity/(2*agent_radius) * (
        environment_influenced[center_x+agent_radius, center_y] - environment_influenced[center_x-agent_radius, center_y])
    velocity_y = agent_gradient_sensitivity/(2*agent_radius)*(
        environment_influenced[center_x, center_y+agent_radius] - environment_influenced[center_x, center_y-agent_radius])
    velocity_max = 5
    velocity = np.sqrt(velocity_x**2+velocity_y**2)
    if velocity > velocity_max:
        velocity_x = velocity_x/velocity*velocity_max
        velocity_y = velocity_y/velocity*velocity_max
    return velocity_x, velocity_y


def update_position(position_x, position_y, delta_time, diffusivity, environment_size_x, environment_size_y, agent_radius, agent_gradient_sensitivity, environment_influenced):
    """
    Define the next position dynamics
    """
    randomwalk_length = np.sqrt(2*delta_time*diffusivity)
    velocity_x, velocity_y = get_velocity(
        position_x, position_y, agent_gradient_sensitivity, environment_influenced, agent_radius)
    position_next_x = position_x + \
        np.random.normal(0, randomwalk_length) + velocity_x*delta_time
    position_next_y = position_y + \
        np.random.normal(0, randomwalk_length) + velocity_y*delta_time
    if position_next_x < (-0.5+agent_radius):
        position_next_x = 2*(-0.5+agent_radius)-position_next_x
    if position_next_x > (environment_size_x-0.5-agent_radius):
        position_next_x = 2*(environment_size_y-0.5 -
                             agent_radius)-position_next_x
    if position_next_y < (-0.5+agent_radius):
        position_next_y = 2*(-0.5+agent_radius)-position_next_y
    if position_next_y > (environment_size_y-0.5-agent_radius):
        position_next_y = 2*(environment_size_x-0.5 -
                             agent_radius)-position_next_y
    return position_next_x, position_next_y


class Cell():
    """
    A cell for the board. Contains information relevant to the experiment.

    Args:
        x (int): the x-position of the board (immutable).
        y (int): the y-position of the board (immutable).
        food (float): the amount of food on this cell.
        time (int): the current simulation time.
        has_agent (bool): whether an agent is present in this cell.
        is_blocking (bool): whether this is an impassable tile (mountains, holes, etc.) Useful for mazes
                            and more complex boards. Default to False for now.
    """

    def __init__(self, x, y, food, time, has_agent, is_impassable=False):
        super(Cell, self).__init__()
        self._time = time
        self._x = x
        self._y = y
        # TODO: These three variables should be accessible through a method.
        self._food = food
        self.has_agent = has_agent
        self.is_impassable = is_impassable

    def get_food(self):
        """
        Get the amount of food in the cell, if any.
        """
        if self.is_impassable:
            # TODO: should we also return IMPASSABLE_FOOD when it is occupied,
            # or should we trust the bot to eventually learn it can't move there?
            return IMPASSABLE_FOOD
        else:
            return self._food


class Board(gym.Env):

    DEFAULT_ENVS = ["none", "simple", "maze", "random"]
    _action_to_coordinates = OrderedDict({"0": [0, 0],
                                          "1": [-1, 0],
                                          "2": [1, 0],
                                          "3": [0, -1],
                                          "4": [0, 1]})  # TODO: diagonal movements

    """ 
    The lightboard class. It is a stateful object that records positions of the agents and food
    available on every cell.
    It is also able to optionally render a movie.
        
    Args:
        agent_positions (list) : A list of tuples (x,y) with the initial positions of the agents.
        x_dim (optional, int): the width of the board. Default: 640.
        y_dim (optional, int): the height of the board. Default: 640.
        I (optional, list): the resource profile. Its dimensions must match x_dim and y_dim. If not
                            specified, it will be randomly generated at start.
        tau (optional, float): the recovery rate. Default: 10
        kappa (optional, float): the consumption rate. Default: 20
        sigma (optional, float): the standard deviation. Default: 3
        preset (optional, str): whether to use one of the preset environments for impassable tiles.
                                For now, the options are ["maze", "random", "simple", "none"]. Default: "simple".
    """

    def __init__(self, agent_positions,
                 position_x,
                 position_y,
                 x_dim=640, y_dim=640,
                 I=None,
                 kappa=20, tau=10, sigma=3,
                 delta_time=0.1,
                 diffusivity=0,
                 agent_gradient_sensitivity=0.5,
                 agent_influence_strength=10,
                 agent_influence_radius=3,
                 agent_radius=1,
                 relaxation_time=12,
                 preset="simple"):

        # I. Init environment variables
        self._agent_positions = {str(i): deepcopy(j)
                                 for i, j in enumerate(agent_positions)}

        self.NUMBER_OF_AGENTS = len(agent_positions)
        self.ENVIRONMENT_SIZE_X = x_dim
        self.ENVIRONMENT_SIZE_Y = y_dim
        self.ENVIRONMENT = I
        self._I = I
        self.ENVIRONMENT_INFLUENCED = deepcopy(self.ENVIRONMENT)
        self.ENVIRONMENT_INFLUENCED_HISTORY = np.zeros((self.ENVIRONMENT_SIZE_X,
                                                        self.ENVIRONMENT_SIZE_Y,
                                                        position_x.shape[1]))
        self.ENVIRONMENT_INFLUENCED_HISTORY[:, :, 0] = I
        self.SHADOW = np.zeros((x_dim, y_dim))
        self.POSITION_X = position_x
        self.POSITION_Y = position_y
        self._x_dim = x_dim
        self._y_dim = y_dim
        self._grid = None

        self.AGENT_RADIUS = agent_radius
        self.AGENT_GRADIENT_SENSITIVITY = agent_gradient_sensitivity
        self.AGENT_INFLUENCE_STRENGTH = agent_influence_strength
        self.AGENT_INFLUENCE_RADIUS = agent_influence_radius

        self.RELAXATION_TIME = 12
        self.DELTA_TIME = delta_time
        self.DIFFUSIVITY = diffusivity
        self.SIGMA = sigma

        self._time = 1

        # II. Init dynamics
        self._max_food = self._I.max()  # For normalizing
        assert (self._x_dim ==
                self._I.shape[0]), "Dimension mismatch! Did you specify x_dim?"
        assert (self._y_dim ==
                self._I.shape[1]), "Dimension mismatch! Did you specify y_dim?"

        # Create a random grid if it's not specified.
        assert (preset in self.DEFAULT_ENVS), "Preset '{}' not found! Choices: {}".format(
            preset, self.DEFAULT_ENVS)
        if self._grid is None:
            self._grid = [[None for _ in range(self._x_dim)]
                          for _ in range(self._y_dim)]
            for i in range(self._x_dim):
                for j in range(self._y_dim):
                    has_agent = (i, j) in agent_positions
                    is_impassable = self._get_preset(i, j, preset)
                    food = self._I[i][j]
                    self._grid[i][j] = Cell(i, j,
                                            food=food,
                                            time=self._time,
                                            has_agent=has_agent,
                                            is_impassable=is_impassable)

    def step(self, agent_action, agent_id, update_time=False):
        """
        Return the next state at time t based on the action of  a given agent.
        Update the agent positions and return the new state of the board. 
        For now we will consider a reward of 0 if the move is unsuccessful, 1
        otherwise; and focus on food as observation.

        Args:
            agent_action (int): An action taken by a given agent.
            agent_id (int): The agent's ID for rendering and so forth. For now we assume 
                            that it is an integer in [0,..., n_agents -1].
            update_time (bool): Whether to update the time on the board. This should
                                be called by the last agent.

        Returns:
            A tuple of the form (observation, reward, is_done, is_successful)
        """
        # Convert integer to coordinates and get the new position.
        action_x, action_y = self._action_to_coordinates[str(agent_action)]
        old_x, old_y = self._agent_positions[str(
            agent_id)][0], self._agent_positions[str(agent_id)][1]

        is_done = False
        reward = 1
        observation = None  # This will be the food.
        # Step 1: Figure out what the next state of the agent will be.
        is_successful = self.is_movement_successful(
            old_x + action_x, old_y + action_y, old_x, old_y)
        if not is_successful:
            reward = 0
            new_x, new_y = old_x, old_y
        else:
            new_x, new_y = old_x + action_x, old_y + action_y
            self._grid[new_x][new_y].has_agent = True
            self._grid[old_x][old_y].has_agent = False
            self._agent_positions[str(agent_id)] = [new_x, new_y]

        observation = self.get_observation_from_coords(new_x, new_y)
        #reward *= max(0, self._grid[new_x][new_y]._food)

        # Step 2: update local food based on consumption rate,
        # and generate new observations of neighboring tiles.
        #old_food = self._grid[old_x][old_y]._food
        #self._grid[old_x][old_y]._food = max(old_food - self._kappa*old_food, 0)

        # Step 3: Finally, if done with all the agents, do
        # rendering/updating.
        if update_time:
            # Update the board state and add it to memory.
            self._update_board_state()

        return (observation, reward, is_done, {"success": is_successful})

    def _update_board_state(self):
        """
        Update the environment based on the dynamics until now.
        """
        for dummy_A in range(0, self.NUMBER_OF_AGENTS):
            self.POSITION_X[dummy_A, self._time], self.POSITION_Y[dummy_A, self._time] = update_position(self.POSITION_X[dummy_A, self._time-1],
                                                                                                         self.POSITION_Y[dummy_A,
                                                                                                                         self._time-1],
                                                                                                         self.DELTA_TIME,
                                                                                                         self.DIFFUSIVITY,
                                                                                                         self.ENVIRONMENT_SIZE_X,
                                                                                                         self.ENVIRONMENT_SIZE_Y,
                                                                                                         self.AGENT_RADIUS,
                                                                                                         self.AGENT_GRADIENT_SENSITIVITY,
                                                                                                         self.ENVIRONMENT_INFLUENCED)
            pos_x = int(self.POSITION_X[dummy_A, self._time])
            pos_y = int(self.POSITION_Y[dummy_A, self._time])
            self._agent_positions[str(dummy_A)] = [pos_x, pos_y]

        SHADOW = update_shadow(self.POSITION_X[:, self._time],
                               self.POSITION_Y[:, self._time],
                               self.SHADOW,
                               self.ENVIRONMENT,
                               self.ENVIRONMENT_SIZE_X,
                               self.ENVIRONMENT_SIZE_Y,
                               self.DELTA_TIME,
                               self.RELAXATION_TIME,
                               self.AGENT_INFLUENCE_STRENGTH,
                               self.AGENT_INFLUENCE_RADIUS,
                               self.NUMBER_OF_AGENTS,
                               self.SIGMA)
        self.ENVIRONMENT_INFLUENCED = self.ENVIRONMENT - SHADOW
        self.SHADOW = SHADOW
        self.ENVIRONMENT_INFLUENCED_HISTORY[:, :,
                                            self._time] = self.ENVIRONMENT_INFLUENCED
        self._time += 1

        # TODO: this can be removed since we are now just updating
        # within a small radius.
        for a in self._agent_positions.keys():
            for i in range(self._x_dim):
                for j in range(self._y_dim):
                    self._grid[i][j]._food = self.ENVIRONMENT_INFLUENCED[i][j]

    def get_observation_from_coords(self, x, y):
        obs = []
        for k, v in self._action_to_coordinates.items():
            v_x, v_y = v[0], v[1]
            if x + v_x < 0:
                obs.append(IMPASSABLE_FOOD)
            elif y + v_y < 0:
                obs.append(IMPASSABLE_FOOD)
            elif x + v_x >= self._x_dim:
                obs.append(IMPASSABLE_FOOD)
            elif y + v_y >= self._y_dim:
                obs.append(IMPASSABLE_FOOD)
            else:
                obs.append(self._grid[x + v_x][y + v_y].get_food())
        return obs

    def _get_preset(self, i, j, preset):
        """
        A way to generate some random environments based on the coordinates of a cell.
        """
        if preset == "maze":
            if i == j and i < self._x_dim/4:
                return True
            if i == j and j > self._y_dim/3:
                return True
            if j == self._x_dim - i and j < self._y_dim/4:
                return True
            return False
        elif preset == "random":
            return random.random() < 0.005
        elif preset == "simple":
            ctr_x, ctr_y = (self._x_dim - 1)/2, (self._y_dim - 1)/2
            radius = np.sqrt((i - ctr_x)**2 + (j - ctr_y)**2)
            if i > radius or j > radius:
                return True
            else:
                return False
        else:
            return False

    def is_movement_successful(self, new_x, new_y, old_x, old_y):
        """
        Test if the new coordinates are valid, based off the action taken
        by the agent.

        Args:
            new_x (int): new x-coordinate from the agent's action
            new_y (int): new y-coordinate from the agent's action
            old_x (int): old x-coordinate from the agent's action
            old_y (int): old y-coordinate from the agent's action

        Returns:
            Whether the next move is successful.
        """
        if new_x < 0 or new_x >= self._x_dim:
            return False
        if new_y < 0 or new_y >= self._y_dim:
            return False
        if self._grid[new_x][new_y].has_agent and new_x != old_x and old_y != new_y:
            return False
        if self._grid[new_x][new_y].is_impassable:
            return False
        return True
