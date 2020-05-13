import torch
import torch.nn as nn
import numpy as np
from hrl4in.utils.networks import AddBias

EPS = 1e-6


def atanh(x):
    return 0.5 * torch.log((1.0 + x + EPS) / (1.0 - x + EPS))


class CustomFixedCategorical(torch.distributions.Categorical):
    def sample(self, sample_shape=torch.Size()):
        return super().sample(sample_shape).unsqueeze(1)  # [time_steps, 1]

    def log_probs(self, actions):
        # actions: [time_steps, 1]
        return (super()
                .log_prob(actions.squeeze(1))  # [time_steps]
                .unsqueeze(1)  # [time_steps, 1]
                )

    def mode(self):
        return self.probs.argmax(dim=1, keepdim=True)  # [time_steps, 1]


class CategoricalNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        x = self.linear(x)
        return CustomFixedCategorical(logits=x)


class CustomFixedMultiCategorical(torch.distributions.Distribution):
    def __init__(self, logits_list):
        self.dists = [CustomFixedCategorical(logits=logits) for logits in logits_list]

    def sample(self, sample_shape=torch.Size()):
        # sample_list: a list of [time_steps, 1], length is n, the number of action categories
        sample_list = [dist.sample(sample_shape=sample_shape) for dist in self.dists]
        return torch.cat(sample_list, dim=1)  # [time_steps, n]

    def log_probs(self, actions):
        # actions: [time_steps, n]
        # log_probs_list: a list of [time_steps, 1], length is n
        log_probs_list = [dist.log_probs(actions[:, i:i+1]) for i, dist in enumerate(self.dists)]

        return (torch.cat(log_probs_list, dim=1)  # [time_steps, n]
                .sum(dim=1, keepdim=True))  # [time_steps, 1]

    def mode(self):
        # mode_list: a list of [time_steps, 1], length is n
        mode_list = [dist.mode() for dist in self.dists]
        return torch.cat(mode_list, dim=1)  # [time_steps, n]

    def entropy(self):
        # entropy_list: a list of [time_steps, 1], length is n
        entropy_list = [dist.entropy().unsqueeze(1) for dist in self.dists]
        return (torch.cat(entropy_list, dim=1)  # [time_steps, n]
                .mean(dim=1))  # [time_steps]


class MultiCategoricalNet(nn.Module):
    def __init__(self, num_inputs, num_outputs_list):
        super().__init__()

        self.linear_layers = nn.ModuleList([nn.Linear(num_inputs, num_outputs) for num_outputs in num_outputs_list])

        for linear_layer in self.linear_layers:
            nn.init.orthogonal_(linear_layer.weight, gain=0.01)
            nn.init.constant_(linear_layer.bias, 0.0)

    def forward(self, x):
        x = [linear_layer(x) for linear_layer in self.linear_layers]
        return CustomFixedMultiCategorical(logits_list=x)


class CustomFixedNormal(torch.distributions.Normal):
    def __init__(self, loc, scale, squash=False):
        super().__init__(loc, scale)
        self.squash = squash

    def sample(self, sample_shape=torch.Size()):
        actions = super().sample(sample_shape)  # [time_steps, action_dim]
        if self.squash:
            actions = torch.tanh(actions)
        return actions

    def squash_correction(self, actions):
        return torch.log((1 - torch.tanh(actions) ** 2 + EPS))

    def log_probs(self, actions):
        if self.squash:
            actions = atanh(actions)  # [time_steps, action_dim]
            log_probs = super().log_prob(actions)  # [time_steps, action_dim]
            log_probs -= self.squash_correction(actions)  # [time_steps, action_dim]
        else:
            log_probs = super().log_prob(actions)  # [time_steps, action_dim]
        return log_probs.sum(dim=1, keepdim=True)  # [time_steps, 1]

    def entropy(self):
        return (super().entropy().  # [time_steps, action_dim]
                mean(dim=1))  # [time_steps]

    def mode(self):
        if self.squash:
            return torch.tanh(self.mean)  # [time_steps, action_dim]
        else:
            return self.mean  # [time_steps, action_dim]


class DiagGaussianNet(nn.Module):
    def __init__(self,
                 num_inputs,
                 num_outputs,
                 action_space,
                 squash_mean=False,
                 squash_distribution=False,
                 initial_stddev=1 / 3.0,
                 min_stddev=0.0,
                 stddev_transform=torch.nn.functional.softplus
                 ):
        super().__init__()
        self.fc_mean = nn.Linear(num_inputs, num_outputs)
        self.action_space_mean = torch.nn.Parameter(
            torch.tensor((action_space.low + action_space.high) / 2.0, dtype=torch.float), requires_grad=False
        )
        self.action_space_magnitude = torch.nn.Parameter(
            torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float), requires_grad=False
        )
        self.squash_mean = squash_mean
        self.squash_distribution = squash_distribution

        # initial_stddev and min_stddev could be a float or a numpy array of floats
        initial_stddev_before_transform = torch.tensor(initial_stddev, dtype=torch.float)
        if stddev_transform == torch.exp:
            initial_stddev_before_transform = torch.log(initial_stddev_before_transform)
        elif stddev_transform == torch.nn.functional.softplus:
            initial_stddev_before_transform = torch.log(torch.exp(initial_stddev_before_transform) - 1.0)
        else:
            assert False, 'unknown stddev transform function'

        self.stddev_before_transform = AddBias(torch.ones(num_outputs) * initial_stddev_before_transform)
        self.stddev_transform = stddev_transform
        min_stddev = torch.ones(num_outputs) * torch.tensor(min_stddev, dtype=torch.float)
        self.min_stddev = torch.nn.Parameter(min_stddev, requires_grad=False)

        # nn.init.orthogonal_(self.fc_mean.weight)
        nn.init.orthogonal_(self.fc_mean.weight, gain=0.5)  # accommodate subgoal range [-2, 2]
        nn.init.constant_(self.fc_mean.bias, 0.0)

    def squash_to_action_spec(self, action):
        return self.action_space_mean + torch.tanh(action) * self.action_space_magnitude

    def forward(self, x, stddev=None):
        action_mean = self.fc_mean(x)
        if self.squash_mean and not self.squash_distribution:
            action_mean = self.squash_to_action_spec(action_mean)
        # print("action_mean", action_mean)

        if stddev is None:
            action_std_before_transform = self.stddev_before_transform(torch.zeros_like(action_mean))
            action_std = self.stddev_transform(action_std_before_transform)
            action_std = torch.max(action_std, self.min_stddev)
        else:
            # directly setting stddev from outside call; stddev has to be a single floating number
            action_std = torch.zeros_like(action_mean)
            action_std[:] = stddev
        return CustomFixedNormal(action_mean, action_std, squash=self.squash_distribution)
