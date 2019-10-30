import torch
from collections import defaultdict
from IPython import embed

def _flatten_helper(t, n, tensor):
    return tensor.view(t * n, *tensor.size()[2:])


class RolloutStorage:
    def __init__(
            self,
            num_steps,
            num_envs,
            observation_space,
            action_space,
            recurrent_hidden_state_size,
    ):
        self.observations = {}

        for sensor in observation_space.spaces:
            self.observations[sensor] = torch.zeros(
                num_steps + 1,
                num_envs,
                *observation_space.spaces[sensor].shape
            )

        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_envs, recurrent_hidden_state_size
        )

        self.rewards = torch.zeros(num_steps, num_envs, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_envs, 1)
        self.returns = torch.zeros(num_steps + 1, num_envs, 1)

        self.action_log_probs = torch.zeros(num_steps, num_envs, 1)
        if action_space.__class__.__name__ == "Discrete":
            action_shape = 1
        else:
            action_shape = action_space.shape[0]

        self.actions = torch.zeros(num_steps, num_envs, action_shape)
        if action_space.__class__.__name__ == "Discrete":
            self.actions = self.actions.long()

        self.masks = torch.ones(num_steps + 1, num_envs, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        for sensor in self.observations:
            self.observations[sensor] = self.observations[sensor].to(device)

        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.device = device

    def insert(
            self,
            observations,
            recurrent_hidden_states,
            actions,
            action_log_probs,
            value_preds,
            rewards,
            masks,
    ):
        for sensor in observations:
            self.observations[sensor][self.step + 1].copy_(
                observations[sensor]
            )
        self.recurrent_hidden_states[self.step + 1].copy_(
            recurrent_hidden_states
        )
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        for sensor in self.observations:
            self.observations[sensor][0].copy_(self.observations[sensor][-1])

        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = (
                        self.rewards[step]
                        + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                        - self.value_preds[step]
                )
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = (
                        self.returns[step + 1] * gamma * self.masks[step + 1]
                        + self.rewards[step]
                )

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch)
        )
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = defaultdict(list)

            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]

                for sensor in self.observations:
                    observations_batch[sensor].append(
                        self.observations[sensor][:-1, ind]
                    )

                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind]
                )

                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind]
                )

                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch

            # These are all tensors of size (T, N, -1)
            for sensor in observations_batch:
                observations_batch[sensor] = torch.stack(observations_batch[sensor], 1)

            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1
            )
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1
            ).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            for sensor in observations_batch:
                observations_batch[sensor] = _flatten_helper(
                    T, N, observations_batch[sensor]
                )

            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(
                T, N, old_action_log_probs_batch
            )
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield (
                observations_batch,
                recurrent_hidden_states_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                old_action_log_probs_batch,
                adv_targ,
            )


class AsyncRolloutStorage:
    def __init__(
            self,
            num_steps,
            num_envs,
            observation_space,
            action_space,
            recurrent_hidden_state_size,
    ):
        self.observations = {}

        for sensor in observation_space.spaces:
            self.observations[sensor] = torch.zeros(
                num_steps + 1,
                num_envs,
                *observation_space.spaces[sensor].shape
            )

        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_envs, recurrent_hidden_state_size
        )

        self.rewards = torch.zeros(num_steps, num_envs, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_envs, 1)
        self.returns = torch.zeros(num_steps + 1, num_envs, 1)

        self.action_log_probs = torch.zeros(num_steps, num_envs, 1)
        if action_space.__class__.__name__ == "Discrete":
            action_shape = 1
        else:
            action_shape = action_space.shape[0]

        self.actions = torch.zeros(num_steps, num_envs, action_shape)
        if action_space.__class__.__name__ == "Discrete":
            self.actions = self.actions.long()

        self.action_mask_log_probs = torch.zeros(num_steps, num_envs, 1)
        self.action_mask_indices = torch.zeros(num_steps, num_envs, 1)
        self.action_mask_indices = self.action_mask_indices.long()

        self.masks = torch.ones(num_steps + 1, num_envs, 1)

        self.num_steps = num_steps
        self.num_envs = num_envs
        self.valid_steps = torch.zeros(num_envs, dtype=torch.long)
        self.env_indices = torch.arange(num_envs, dtype=torch.long)

    def to(self, device):
        for sensor in self.observations:
            self.observations[sensor] = self.observations[sensor].to(device)

        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.action_mask_log_probs = self.action_mask_log_probs.to(device)
        self.action_mask_indices = self.action_mask_indices.to(device)
        self.masks = self.masks.to(device)
        self.valid_steps = self.valid_steps.to(device)
        self.device = device

    def insert(
            self,
            insert_mask,
            observations,
            recurrent_hidden_states,
            actions,
            action_log_probs,
            action_mask_indices,
            action_mask_log_probs,
            value_preds,
            rewards,
            masks,
    ):
        for env_id in range(self.num_envs):
            if insert_mask[env_id]:
                for sensor in observations:
                    self.observations[sensor][self.valid_steps[env_id] + 1, env_id].copy_(
                        observations[sensor][env_id])
                self.recurrent_hidden_states[self.valid_steps[env_id] + 1, env_id].copy_(
                    recurrent_hidden_states[env_id])
                self.actions[self.valid_steps[env_id], env_id].copy_(actions[env_id])
                self.action_log_probs[self.valid_steps[env_id], env_id].copy_(action_log_probs[env_id])
                self.action_mask_indices[self.valid_steps[env_id], env_id].copy_(action_mask_indices[env_id])
                self.action_mask_log_probs[self.valid_steps[env_id], env_id].copy_(action_mask_log_probs[env_id])
                self.value_preds[self.valid_steps[env_id], env_id].copy_(value_preds[env_id])
                self.rewards[self.valid_steps[env_id], env_id].copy_(rewards[env_id])
                self.masks[self.valid_steps[env_id] + 1, env_id].copy_(masks[env_id])
                self.valid_steps[env_id] += 1

    def after_update(self):
        for sensor in self.observations:
            self.observations[sensor][0].copy_(self.observations[sensor][self.valid_steps, self.env_indices])

        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[self.valid_steps, self.env_indices])
        self.masks[0].copy_(self.masks[self.valid_steps, self.env_indices])
        self.valid_steps.zero_()

    def compute_returns(self, next_value, use_gae, gamma, tau):
        for env_id in range(self.num_envs):
            if use_gae:
                self.value_preds[self.valid_steps[env_id], env_id] = next_value[env_id]
                gae = 0
                for step in reversed(range(self.valid_steps[env_id])):
                    delta = (
                            self.rewards[step, env_id]
                            + gamma * self.value_preds[step + 1, env_id] * self.masks[step + 1, env_id]
                            - self.value_preds[step, env_id]
                    )
                    gae = delta + gamma * tau * self.masks[step + 1, env_id] * gae
                    self.returns[step, env_id] = gae + self.value_preds[step, env_id]
            else:
                self.returns[self.valid_steps[env_id], env_id] = next_value[env_id]
                for step in reversed(range(self.valid_steps[env_id])):
                    self.returns[step, env_id] = (
                            self.returns[step + 1, env_id] * gamma * self.masks[step + 1, env_id]
                            + self.rewards[step, env_id]
                    )

    def recurrent_generator(self, advantages, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch)
        )
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = defaultdict(list)

            recurrent_hidden_states_batch = []
            actions_batch = []
            action_mask_indices_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            valid_masks_batch = []

            max_valid_steps = 0
            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                valid_mask = (torch.arange(self.num_steps, device=self.device) < self.valid_steps[ind])
                max_valid_steps = max(max_valid_steps, self.valid_steps[ind].item())
                for sensor in self.observations:
                    observations_batch[sensor].append(
                        self.observations[sensor][:-1, ind]
                    )

                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind]
                )

                actions_batch.append(self.actions[:, ind])
                action_mask_indices_batch.append(self.action_mask_indices[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind] +
                    self.action_mask_log_probs[:, ind]
                )
                adv_targ.append(advantages[:, ind])
                valid_masks_batch.append(valid_mask)

            T, N = self.num_steps, num_envs_per_batch

            # These are all tensors of size (T, N, -1)
            for sensor in observations_batch:
                observations_batch[sensor] = torch.stack(observations_batch[sensor], 1)
            actions_batch = torch.stack(actions_batch, 1)
            action_mask_indices_batch = torch.stack(action_mask_indices_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ_batch = torch.stack(adv_targ, 1)
            valid_masks_batch = torch.stack(valid_masks_batch, 1)

            # hidden states is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(recurrent_hidden_states_batch, 1).view(N, -1)

            # Slice the (T, N, ...) tensors to (T', N, ...), where T' is the max valid steps across all parallel envs
            for sensor in observations_batch:
                observations_batch[sensor] = observations_batch[sensor][:max_valid_steps]
            actions_batch = actions_batch[:max_valid_steps]
            action_mask_indices_batch = action_mask_indices_batch[:max_valid_steps]
            value_preds_batch = value_preds_batch[:max_valid_steps]
            return_batch = return_batch[:max_valid_steps]
            masks_batch = masks_batch[:max_valid_steps]
            old_action_log_probs_batch = old_action_log_probs_batch[:max_valid_steps]
            adv_targ_batch = adv_targ_batch[:max_valid_steps]
            valid_masks_batch = valid_masks_batch[:max_valid_steps]

            # Flatten the (T', N, ...) tensors to (T' * N, ...)
            for sensor in observations_batch:
                observations_batch[sensor] = _flatten_helper(max_valid_steps, N, observations_batch[sensor])
            actions_batch = _flatten_helper(max_valid_steps, N, actions_batch)
            action_mask_indices_batch = _flatten_helper(max_valid_steps, N, action_mask_indices_batch)
            value_preds_batch = _flatten_helper(max_valid_steps, N, value_preds_batch)
            return_batch = _flatten_helper(max_valid_steps, N, return_batch)
            masks_batch = _flatten_helper(max_valid_steps, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(max_valid_steps, N, old_action_log_probs_batch)
            adv_targ_batch = _flatten_helper(max_valid_steps, N, adv_targ_batch)
            valid_masks_batch = _flatten_helper(max_valid_steps, N, valid_masks_batch)

            # # Flatten the (T, N, ...) tensors to (T * N, ...)
            # for sensor in observations_batch:
            #     observations_batch[sensor] = _flatten_helper(
            #         T, N, observations_batch[sensor]
            #     )
            #
            # actions_batch = _flatten_helper(T, N, actions_batch)
            # action_mask_indices_batch = _flatten_helper(T, N, action_mask_indices_batch)
            #
            # value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            # return_batch = _flatten_helper(T, N, return_batch)
            # masks_batch = _flatten_helper(T, N, masks_batch)
            # old_action_log_probs_batch = _flatten_helper(
            #     T, N, old_action_log_probs_batch
            # )
            # adv_targ_batch = _flatten_helper(T, N, adv_targ_batch)
            # valid_masks_batch = _flatten_helper(T, N, valid_masks_batch)

            yield (
                observations_batch,
                recurrent_hidden_states_batch,
                actions_batch,
                action_mask_indices_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                old_action_log_probs_batch,
                adv_targ_batch,
                valid_masks_batch,
            )
