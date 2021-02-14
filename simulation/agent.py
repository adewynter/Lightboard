import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class NeuralNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None):
        """ Simple two-layer neural network.
        """
        super(NeuralNetwork, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim*2
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.ac1 = nn.LeakyReLU()
        self.l2 = nn.Linear(hidden_dim, out_dim)
        self.ac2 = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.ac1(self.l1(x.float()))
        return self.ac2(self.l2(x))


class Agent():
    """ The agent (bot).

        Args:
            n_dof (int, optional): the degrees of freedom of the bot. In other words,
                                   whether it can only move sideways/up/downward, or also
                                   in diagonal. It includes the choice of not moving and the reward.
                                   Default: 1+1+4
            lr (float, optional): the learning rate at which to perform optimization.
            rho (float, optional): the momentum for the optimizer (SGD only).
    """

    def __init__(self, pos_x, pos_y, n_dof=6, lr=0.0001, rho=0.9):
        super(Agent, self).__init__()
        self._n_dof = n_dof
        self._model = NeuralNetwork(in_dim=self._n_dof,
                                    out_dim=self._n_dof - 1)
        self._optimizer = optim.SGD(
            self._model.parameters(), lr=lr, momentum=rho)
        self._loss = nn.MSELoss()
        self.position_x = int(pos_x)
        self.position_y = int(pos_y)
        self.past_reward = 0.
        self.action_to_position = {"0": [0, 0],
                                   "1": [-1, 0],
                                   "2": [1, 0],
                                   "3": [0, -1],
                                   "4": [0, 1]}
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def observation_to_label(self, obs):
        """
        Convert an observation into a label that can be used by our model.
        """
        return self.softmax(self.relu(torch.tensor(obs)))

    def act(self, state):
        """
        Act given a state.

        Args:
            state (list): The current state observed by the robot (all the visible tiles
                          along with their food).
        """
        self._model.eval()
        inp = torch.tensor([self.past_reward] + state)
        out = self._model(inp)
        return np.argmax(out.detach().cpu().numpy())

    def update(self, observation, reward, is_successful):
        """
        Update our agent.

        Args:
           observation (list): the observation for this agent (all the visible tiles)
           reward (float): the reward obtained by the model.
           is_successful (bool): whether the move was successful. Needed to
                                 update the internal coordinates of the model.
        """
        observation = [o/(max(observation) + 1e-6) for o in observation]
        self._model.train()
        self.past_reward = reward
        out = self._model(torch.tensor([reward] + observation))
        loss = self._loss(out, reward*self.observation_to_label(observation))
        loss.backward()
        _ = nn.utils.clip_grad_norm_(
            self._model.parameters(), 5.0)  # Needed to avoid NaNs
        self._optimizer.step()
        self._optimizer.zero_grad()

        if is_successful:
            out = np.argmax(out.detach().cpu().numpy())
            shift = self.action_to_position[str(int(out))]
            self.position_x = self.position_x + shift[0]
            self.position_y = self.position_y + shift[1]
