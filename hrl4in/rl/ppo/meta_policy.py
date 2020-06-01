#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from hrl4in.utils.distributions import DiagGaussianNet, CategoricalNet
from hrl4in.utils.networks import Net


class MetaPolicy(nn.Module):
    def __init__(self,
                 observation_space,
                 subgoal_space,
                 use_action_masks=False,
                 action_masks_dim=3,
                 hidden_size=512,
                 cnn_layers_params=None,
                 initial_stddev=1.0 / 3.0,
                 min_stddev=0.0,
                 stddev_transform=torch.nn.functional.softplus):
        super().__init__()
        self.net = Net(
            observation_space=observation_space,
            hidden_size=hidden_size,
            cnn_layers_params=cnn_layers_params,
        )

        assert len(subgoal_space.shape) == 1, 'only supports one dimensional subgoal space'
        self.subgoal_distribution = DiagGaussianNet(self.net.output_size,
                                                    subgoal_space.shape[0],
                                                    subgoal_space,
                                                    squash_mean=True,
                                                    squash_distribution=False,
                                                    initial_stddev=initial_stddev,
                                                    min_stddev=min_stddev,
                                                    stddev_transform=stddev_transform)
        self.use_action_masks = use_action_masks

        # base + arm or base-only
        if self.use_action_masks:
            self.action_mask_distribution = CategoricalNet(self.net.output_size, action_masks_dim)

    def forward(self, *x):
        raise NotImplementedError

    def act(self, observations, rnn_hidden_states, masks, deterministic=False):
        value, actor_features, rnn_hidden_states = self.net(observations, rnn_hidden_states, masks)
        subgoal_distribution = self.subgoal_distribution(actor_features)

        if deterministic:
            subgoals = subgoal_distribution.mode()
        else:
            subgoals = subgoal_distribution.sample()
        subgoal_log_probs = subgoal_distribution.log_probs(subgoals)

        # print("mean", subgoal_distribution.loc)
        # print("std", subgoal_distribution.scale)

        if self.use_action_masks:
            action_mask_distribution = self.action_mask_distribution(actor_features)
            if deterministic:
                action_mask_indices = action_mask_distribution.mode()
            else:
                action_mask_indices = action_mask_distribution.sample()
            action_mask_log_probs = action_mask_distribution.log_probs(action_mask_indices)
        else:
            action_mask_indices = torch.zeros_like(subgoal_log_probs, dtype=torch.long)
            action_mask_log_probs = torch.zeros_like(subgoal_log_probs)

        return (
            value,
            subgoals,
            subgoal_log_probs,
            action_mask_indices,
            action_mask_log_probs,
            rnn_hidden_states,
        )

    def get_value(self, observations, rnn_hidden_states, masks):
        value, _, _ = self.net(observations, rnn_hidden_states, masks)
        return value

    def evaluate_actions(self,
                         observations,
                         rnn_hidden_states,
                         masks,
                         subgoals,
                         action_mask_indices
                         ):
        value, actor_features, rnn_hidden_states = self.net(observations, rnn_hidden_states, masks)
        subgoal_distribution = self.subgoal_distribution(actor_features)

        subgoal_log_probs = subgoal_distribution.log_probs(subgoals)
        subgoal_dist_entropy = subgoal_distribution.entropy()

        if self.use_action_masks:
            action_mask_distribution = self.action_mask_distribution(actor_features)
            action_mask_log_probs = action_mask_distribution.log_probs(action_mask_indices)
            action_mask_dist_entropy = action_mask_distribution.entropy()
        else:
            action_mask_log_probs = torch.zeros_like(subgoal_log_probs)
            action_mask_dist_entropy = torch.zeros_like(subgoal_dist_entropy)

        action_log_probs = subgoal_log_probs + action_mask_log_probs
        dist_entropy = subgoal_dist_entropy + action_mask_dist_entropy

        return (
            value,
            action_log_probs,
            dist_entropy,
            rnn_hidden_states
        )
