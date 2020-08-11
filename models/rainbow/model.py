import math
import random
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

# H = 128
H1 = 128
H2 = 128
input_shape = 4
output_shape = 5


def DQN(env, args):
    global input_shape
    if args.encoding:
        input_shape = 5

    if args.c51:
        if args.dueling:
            model = CategoricalDuelingDQN(env, args.noisy, args.sigma_init,
                                          args.Vmin, args.Vmax, args.num_atoms, args.batch_size)
        else:
            model = CategoricalDQN(env, args.noisy, args.sigma_init,
                                   args.Vmin, args.Vmax, args.num_atoms, args.batch_size)
    else:
        if args.dueling:
            model = DuelingDQN(env, args.noisy, args.sigma_init)
        else:
            model = DQNBase(env, args.noisy, args.sigma_init)

    return model


class DQNBase(nn.Module):
    """
    Basic DQN + NoisyNet

    Noisy Networks for Exploration
    https://arxiv.org/abs/1706.10295
    
    parameters
    ---------
    env         environment(openai gym)
    noisy       boolean value for NoisyNet. 
                If this is set to True, self.Linear will be NoisyLinear module
    """

    class MyDataParallel(nn.DataParallel):
        def __getattr__(self, name):
            return getattr(self.module, name)


    def __init__(self, env, noisy, sigma_init):
        super(DQNBase, self).__init__()

        # self.input_shape = env.observation_space.shape
        # self.num_actions = env.action_space.n
        self.noisy = noisy

        if noisy:
            self.Linear = partial(NoisyLinear, sigma_init=sigma_init)
        else:
            self.Linear = nn.Linear

        self.flatten = Flatten()

        self.fc = nn.Sequential(
            self.Linear(input_shape, H1),
            nn.ReLU(),
            self.Linear(H1, output_shape)
        )

    def forward(self, x):
        # x = self.features(x)
        # x = self.flatten(x)
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
        if random.random() > epsilon or self.noisy:  # NoisyNet does not use e-greedy
            with torch.no_grad():
                state = state.unsqueeze(0)
                q_value = self.forward(state)
                action = q_value.max(1)[1].item()
        else:
            action = random.randrange(output_shape)
        return action

    def best_act(self, state):
        """
        Parameters
        ----------
        state       torch.Tensor with appropritate device type
        epsilon     epsilon for epsilon-greedy
        """

        with torch.no_grad():
            state = state.unsqueeze(0)
            q_value = self.forward(state)
            action = q_value.max(1)[1].item()

        return action

    def update_noisy_modules(self):
        if self.noisy:
            self.noisy_modules = [module for module in self.modules() if isinstance(module, NoisyLinear)]

    def sample_noise(self):
        for module in self.noisy_modules:
            module.sample_noise()

    def remove_noise(self):
        for module in self.noisy_modules:
            module.remove_noise()


class NFSP_Model(nn.Module):

    def __init__(self, action_num):
        super(NFSP_Model, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, H2),
            nn.ReLU(),
            nn.Linear(H2, action_num),
        )

    def forward(self, observation):
        out = self.fc(observation)
        return out

    def act(self, state):
        with torch.no_grad():
            state = state.unsqueeze(0)
            q_value = self.forward(state)
            action = q_value.max(1)[1].item()
        return action

    def load_nfsp_Q(self, checkpoint_path, num):
        # Hack to load models saved with GPU
        map_location = lambda storage, loc: storage
        # map_location = None
        checkpoint = torch.load(checkpoint_path, map_location)
        self.load_state_dict(checkpoint['p' + str(num) + '_model'])

    def load_nfsp_policy(self, checkpoint_path, num):
        map_location = lambda storage, loc: storage
        # map_location = None
        checkpoint = torch.load(checkpoint_path, map_location)
        self.load_state_dict(checkpoint['p' + str(num) + '_policy'])


class NFSP_Policy(NFSP_Model):
    """
    Policy with only actors. This is used in supervised learning for NFSP.
    """

    def __init__(self, action_num):
        super(NFSP_Policy, self).__init__(action_num)
        self.fc = nn.Sequential(
            nn.Linear(4, H2),
            nn.ReLU(),
            nn.Linear(H2, action_num),
            nn.Softmax(dim=1)
        )

    def act(self, state):
        """
        Parameters
        ----------
        state       torch.Tensor with appropritate device type
        """
        with torch.no_grad():
            state = state.unsqueeze(0)
            distribution = self.forward(state)
            # m = Categorical(distribution)
            # action = m.sample()
            action = distribution.argmax(1).item()

        return action

    def act_dist(self, state):
        """
        Parameters
        ----------
        state       torch.Tensor with appropritate device type
        """
        with torch.no_grad():
            state = state.unsqueeze(0)
            distribution = self.forward(state)

        return distribution


class DuelingDQN(DQNBase):
    """
    Dueling Network Architectures for Deep Reinforcement Learning
    https://arxiv.org/abs/1511.06581
    """

    def __init__(self, env, noisy, sigma_init):
        super(DuelingDQN, self).__init__(env, noisy, sigma_init)

        self.advantage = self.fc

        self.value = nn.Sequential(
            self.Linear(input_shape, H1),
            nn.ReLU(),
            self.Linear(H1, 1)
        )

    def forward(self, x):
        # x = self.features(x)
        # x = self.flatten(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean(1, keepdim=True)


class CategoricalDQN(DQNBase):
    """
    A Distributional Perspective on Reinforcement Learning
    https://arxiv.org/abs/1707.06887
    """

    def __init__(self, env, noisy, sigma_init, Vmin, Vmax, num_atoms, batch_size):
        super(CategoricalDQN, self).__init__(env, noisy, sigma_init)

        support = torch.linspace(Vmin, Vmax, num_atoms)
        offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size).long() \
            .unsqueeze(1).expand(batch_size, num_atoms)

        self.register_buffer('support', support)
        self.register_buffer('offset', offset)
        self.num_atoms = num_atoms

        self.fc = nn.Sequential(
            self.Linear(input_shape, H1),
            nn.ReLU(),
            self.Linear(H1, output_shape * self.num_atoms),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = self.features(x)
        # x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x.view(-1, self.num_atoms))
        x = x.view(-1, output_shape, self.num_atoms)
        return x

    def act(self, state, epsilon):
        """
        Parameters
        ----------
        state       torch.Tensor with appropritate device type
        epsilon     epsilon for epsilon-greedy
        """
        if random.random() > epsilon or self.noisy:  # NoisyNet does not use e-greedy
            with torch.no_grad():
                state = state.unsqueeze(0)
                q_dist = self.forward(state)
                q_value = (q_dist * self.support).sum(2)
                action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.num_actions)
        return action


class CategoricalDuelingDQN(CategoricalDQN):

    def __init__(self, env, noisy, sigma_init, Vmin, Vmax, num_atoms, batch_size):
        super(CategoricalDuelingDQN, self).__init__(env, noisy, sigma_init, Vmin, Vmax, num_atoms, batch_size)

        self.advantage = self.fc

        self.value = nn.Sequential(
            self.Linear(input_shape, H1),
            nn.ReLU(),
            self.Linear(H1, num_atoms)
        )

    def forward(self, x):
        # x = self.features(x)
        x = self.flatten(x)

        advantage = self.advantage(x).view(-1, output_shape, self.num_atoms)
        value = self.value(x).view(-1, 1, self.num_atoms)

        x = value + advantage - advantage.mean(1, keepdim=True)
        x = self.softmax(x.view(-1, self.num_atoms))
        x = x.view(-1, output_shape, self.num_atoms)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.register_buffer('sample_weight_in', torch.FloatTensor(in_features))
        self.register_buffer('sample_weight_out', torch.FloatTensor(out_features))
        self.register_buffer('sample_bias_out', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.sample_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.bias_sigma.size(0)))

    def sample_noise(self):
        self.sample_weight_in = self._scale_noise(self.sample_weight_in)
        self.sample_weight_out = self._scale_noise(self.sample_weight_out)
        self.sample_bias_out = self._scale_noise(self.sample_bias_out)

        self.weight_epsilon.copy_(self.sample_weight_out.ger(self.sample_weight_in))
        self.bias_epsilon.copy_(self.sample_bias_out)

    def _scale_noise(self, x):
        x = x.normal_()
        x = x.sign().mul(x.abs().sqrt())
        return x