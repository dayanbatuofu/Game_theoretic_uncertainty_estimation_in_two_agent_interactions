import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

H = 128
input_shape = 4
output_shape = 5


def DQN(args):
    if args.dueling:
        model = DuelingDQN()
    else:
        model = DQNBase()
    return model


class DQNBase(nn.Module):
    """
    Basic DQN

    parameters
    ---------
    env         environment(openai gym)
    """

    def __init__(self):
        super(DQNBase, self).__init__()

        self.flatten = Flatten()

        self.fc = nn.Sequential(
            nn.Linear(input_shape, H),
            nn.ReLU(),
            nn.Linear(H, output_shape)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

    def _feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)

    def act(self, state, epsilon):
        """
        Parameters
        ----------
        state       torch.Tensor with appropritate device type
        epsilon     epsilon for epsilon-greedy
        """
        if random.random() > epsilon:  # NoisyNet does not use e-greedy
            with torch.no_grad():
                state = state.unsqueeze(0)
                q_value = self.forward(state)
                action = q_value.max(1)[1].item()
        else:
            action = random.randrange(output_shape)
        return action

    def best_action(self, state):
        """
        Parameters
        ----------
        state       torch.Tensor with appropritate device type

        """
        with torch.no_grad():
            state = state.unsqueeze(0)
            q_value = self.forward(state)
            action = q_value.max(1)[1].item()

        return action


class Policy(DQNBase):
    """
    Policy with only actors. This is used in supervised learning for NFSP.
    """

    def __init__(self):
        super(Policy, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_shape, H),
            nn.ReLU(),
            nn.Linear(H, output_shape),
            nn.Softmax(dim=1)
        )

    def act(self, state, mode):
        """
        Parameters
        ----------
        state       torch.Tensor with appropriate device type
        """
        with torch.no_grad():
            state = state.unsqueeze(0)
            distribution = self.forward(state)
            if mode == 0:
                m = Categorical(distribution)
                action = m.sample().item()
                # action = distribution.multinomial(1).item()
            else:
                action = distribution.argmax(1).item()
        return action


class DuelingDQN(DQNBase):
    """
    Dueling Network Architectures for Deep Reinforcement Learning
    https://arxiv.org/abs/1511.06581
    """

    def __init__(self):
        super(DuelingDQN, self).__init__()

        self.advantage = self.fc

        self.value = nn.Sequential(
            # nn.Linear(self._feature_size(), 32),
            # nn.ReLU(),
            # nn.Linear(32, 1)
            nn.Linear(input_shape, H),
            nn.ReLU(),
            nn.Linear(H, 1)
        )

    def forward(self, x):
        # x /= 255.
        # x = self.features(x)
        x = self.flatten(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(1, keepdim=True)


class cReLU(nn.Module):
    def forward(self, x):
        return torch.cat((F.relu(x), F.relu(-x)), dim=1)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
