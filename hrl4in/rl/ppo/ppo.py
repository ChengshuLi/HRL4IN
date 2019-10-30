#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim
from IPython import embed

EPS_PPO = 1e-5


class PPO(nn.Module):
    def __init__(
            self,
            actor_critic,
            clip_param,
            ppo_epoch,
            num_mini_batch,
            value_loss_coef,
            entropy_coef,
            lr=None,
            eps=None,
            max_grad_norm=None,
            use_clipped_value_loss=True,
            is_meta_agent=False,
            normalize_advantage=True,
    ):

        super().__init__()

        self.actor_critic = actor_critic
        self.is_meta_agent = is_meta_agent
        self.normalize_advantage = normalize_advantage

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def forward(self, *x):
        raise NotImplementedError

    def update(self, rollouts, update=None):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        # TODO: meta_rollouts usually has too little data during each update, should we still normalize advantages?
        if self.normalize_advantage:
            if self.is_meta_agent:
                indices = torch.arange(rollouts.num_steps, device=rollouts.device)
                indices = indices.unsqueeze(1).repeat(1, rollouts.num_envs)
                valid_masks = indices < rollouts.valid_steps.unsqueeze(0)
                advantages = (advantages - advantages[valid_masks].mean()) / (advantages[valid_masks].std() + EPS_PPO)
            else:
                advantages = (advantages - advantages.mean()) / (advantages.std() + EPS_PPO)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch
            )

            for sample in data_generator:
                if self.is_meta_agent:
                    (
                        obs_batch,
                        recurrent_hidden_states_batch,
                        actions_batch,
                        action_mask_indices_batch,
                        value_preds_batch,
                        return_batch,
                        masks_batch,
                        old_action_log_probs_batch,
                        adv_targ_batch,
                        valid_masks_batch,
                    ) = sample
                else:
                    (
                        obs_batch,
                        recurrent_hidden_states_batch,
                        actions_batch,
                        value_preds_batch,
                        return_batch,
                        masks_batch,
                        old_action_log_probs_batch,
                        adv_targ_batch,
                    ) = sample

                # Reshape to do in a single forward pass for all steps
                if self.is_meta_agent:
                    (
                        values,
                        action_log_probs,
                        dist_entropy,
                        _,
                    ) = self.actor_critic.evaluate_actions(
                        obs_batch,
                        recurrent_hidden_states_batch,
                        masks_batch,
                        actions_batch,
                        action_mask_indices_batch,
                    )
                    values = values[valid_masks_batch]
                    action_log_probs = action_log_probs[valid_masks_batch]
                    dist_entropy = dist_entropy[valid_masks_batch]
                    value_preds_batch = value_preds_batch[valid_masks_batch]
                    return_batch = return_batch[valid_masks_batch]
                    old_action_log_probs_batch = old_action_log_probs_batch[valid_masks_batch]
                    adv_targ_batch = adv_targ_batch[valid_masks_batch]
                else:
                    (
                        values,
                        action_log_probs,
                        dist_entropy,
                        _,
                    ) = self.actor_critic.evaluate_actions(
                        obs_batch,
                        recurrent_hidden_states_batch,
                        masks_batch,
                        actions_batch,
                        update=update,
                    )

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * adv_targ_batch
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ_batch
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                                         torch.clamp(values - value_preds_batch, -self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()
                dist_entropy = dist_entropy.mean()

                self.optimizer.zero_grad()
                total_loss = (action_loss
                              + value_loss * self.value_loss_coef
                              - dist_entropy * self.entropy_coef)
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
