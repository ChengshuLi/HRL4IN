#!/usr/bin/env python3

from time import time
from collections import deque
import random
import numpy as np
import argparse
import gym

import torch
from torch.utils.tensorboard import SummaryWriter

import hrl4in
from hrl4in.envs.toy_env.toy_env import ToyEnv
from hrl4in.utils.logging import logger
from hrl4in.rl.ppo import PPO, Policy, RolloutStorage, MetaPolicy, AsyncRolloutStorage
from hrl4in.utils.utils import *
from hrl4in.utils.args import *

from gibson2.envs.parallel_env import ParallelNavEnvironment

from IPython import embed
import matplotlib.pyplot as plt

def evaluate(args,
             envs,
             meta_actor_critic,
             actor_critic,
             sensor_mean,
             sensor_magnitude,
             auxiliary_sensor_mean,
             auxiliary_sensor_magnitude,
             action_mask_choices,
             subgoal_mask_choices,
             env_config,
             device,
             writer,
             update=0,
             count_steps=0,
             eval_only=False,
             visualize=False):
    env_width = env_config.get("width")
    env_height = env_config.get("height")
    env_door_row = env_config.get("door_row")
    env_door_col = env_config.get("door_col")

    observations = envs.reset()
    batch = batch_obs(observations)
    for sensor in batch:
        batch[sensor] = batch[sensor].to(device)

    episode_rewards = torch.zeros(envs._num_envs, 1, device=device)
    episode_success_rates = torch.zeros(envs._num_envs, 1, device=device)
    episode_lengths = torch.zeros(envs._num_envs, 1, device=device)
    episode_counts = torch.zeros(envs._num_envs, 1, device=device)
    current_episode_reward = torch.zeros(envs._num_envs, 1, device=device)

    subgoal_rewards = torch.zeros(envs._num_envs, 1, device=device)
    subgoal_success_rates = torch.zeros(envs._num_envs, 1, device=device)
    subgoal_lengths = torch.zeros(envs._num_envs, 1, device=device)
    subgoal_counts = torch.zeros(envs._num_envs, 1, device=device)
    current_subgoal_reward = torch.zeros(envs._num_envs, 1, device=device)

    current_meta_recurrent_hidden_states = torch.zeros(envs._num_envs, args.hidden_size, device=device)
    next_meta_recurrent_hidden_states = torch.zeros(envs._num_envs, args.hidden_size, device=device)
    recurrent_hidden_states = torch.zeros(envs._num_envs, args.hidden_size, device=device)
    subgoals_done = torch.zeros(envs._num_envs, 1, device=device)
    masks = torch.zeros(envs._num_envs, 1, device=device)

    current_subgoals = torch.zeros(batch["sensor"].shape, device=device)
    current_subgoals_steps = torch.zeros(envs._num_envs, 1, device=device)
    current_subgoal_masks = torch.zeros(batch["sensor"].shape, device=device)

    action_dim = envs.action_space.nvec.shape[0]
    current_action_masks = torch.zeros(envs._num_envs, action_dim, device=device)

    subgoal_round_to = torch.ones(batch["sensor"].shape, device=device)
    subgoal_round_to[:, 2] = np.pi / 2.0

    steps = 0
    while episode_counts.sum() < args.num_eval_episodes:
        with torch.no_grad():
            (
                _,
                subgoals,
                _,
                action_mask_indices,
                _,
                meta_recurrent_hidden_states,
            ) = meta_actor_critic.act(
                batch,
                current_meta_recurrent_hidden_states,
                masks,
                deterministic=False,
            )

            prev_observations_denormalized = batch["sensor"] * sensor_magnitude + sensor_mean
            prev_auxiliary_sensor_denormalized = batch["auxiliary_sensor"] * auxiliary_sensor_magnitude + auxiliary_sensor_mean

            if meta_actor_critic.use_action_masks:
                action_masks = action_mask_choices.index_select(0, action_mask_indices.squeeze(1))
                subgoal_masks = subgoal_mask_choices.index_select(0, action_mask_indices.squeeze(1))
            else:
                action_masks = torch.ones_like(current_action_masks)
                subgoal_masks = torch.ones_like(current_subgoal_masks)

            should_use_new_subgoals = (current_subgoals_steps == 0.0).float()
            current_subgoals = should_use_new_subgoals * subgoals + \
                               (1 - should_use_new_subgoals) * current_subgoals
            current_subgoal_masks = should_use_new_subgoals * subgoal_masks.float() + \
                                    (1 - should_use_new_subgoals) * current_subgoal_masks
            current_subgoals *= current_subgoal_masks
            current_action_masks = should_use_new_subgoals * action_masks + \
                                   (1 - should_use_new_subgoals) * current_action_masks
            next_meta_recurrent_hidden_states = should_use_new_subgoals * meta_recurrent_hidden_states + \
                                                (1 - should_use_new_subgoals) * next_meta_recurrent_hidden_states

            # compute ideal next state before observation is masked
            ideal_next_state = batch["sensor"] + current_subgoals
            ideal_next_state_denormalized = ideal_next_state * sensor_magnitude + sensor_mean
            ideal_next_state_denormalized[:, 2] = wrap_to_pi(ideal_next_state_denormalized[:, 2])
            ideal_next_state_denormalized_rounded = torch.round(
                ideal_next_state_denormalized / subgoal_round_to
            ) * subgoal_round_to
            # ideal_next_state_denormalized_rounded = torch.round(ideal_next_state_denormalized)

            # transform subgoal in global coordinate to subgoal in local coordinate
            theta = batch["sensor"][:, 2] * np.pi
            rotation_matrix = torch.zeros((envs._num_envs, 2, 2), device=device)
            rotation_matrix[:, 0, 0] = torch.cos(-theta)
            rotation_matrix[:, 0, 1] = -torch.sin(-theta)
            rotation_matrix[:, 1, 0] = torch.sin(-theta)
            rotation_matrix[:, 1, 1] = torch.cos(-theta)
            current_subgoals_rotated = torch.bmm(rotation_matrix, current_subgoals[:, 0:2].unsqueeze(2)).squeeze(2)

            # full subgoal with rotation
            current_subgoals_observation = current_subgoals.clone()
            current_subgoals_observation[:, 0:2] = current_subgoals_rotated

            batch["subgoal"] = current_subgoals_observation
            batch["subgoal_mask"] = current_subgoal_masks
            batch["action_mask"] = current_action_masks

            # values: [num_processes, 1]
            # actions: [num_processes, 1]
            # actions_log_probs: [num_processes, 1]
            # recurrent_hidden_states: [num_processes, hidden_size]
            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
            ) = actor_critic.act(
                batch,
                recurrent_hidden_states,
                1 - subgoals_done,
                deterministic=False,
            )
            actions_masked = actions * current_action_masks.long()

        actions_np = actions_masked.cpu().numpy()
        outputs = envs.step(actions_np)

        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
        next_obs = [info["last_observation"] if done else obs for obs, done, info in
                    zip(observations, dones, infos)]

        batch = batch_obs(observations)
        for sensor in batch:
            batch[sensor] = batch[sensor].to(device)

        next_obs_batch = batch_obs(next_obs)
        for sensor in next_obs_batch:
            next_obs_batch[sensor] = next_obs_batch[sensor].to(device)

        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        rewards = rewards.unsqueeze(1)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=device,
        )
        success_masks = torch.tensor(
            [[1.0] if done and "success" in info and info["success"] else [0.0]
             for done, info in zip(dones, infos)],
            dtype=torch.float,
            device=device
        )
        lengths = torch.tensor(
            [[float(info["episode_length"])] if done and "episode_length" in info else [0.0]
             for done, info in zip(dones, infos)],
            dtype=torch.float,
            device=device
        )

        current_episode_reward += rewards
        episode_rewards += (1 - masks) * current_episode_reward
        episode_success_rates += success_masks
        episode_lengths += lengths
        episode_counts += 1 - masks
        current_episode_reward *= masks

        current_subgoals_steps += 1
        steps += 1
        if masks[0][0] == 0.0:
            steps = 0

        observations_denormalized = next_obs_batch["sensor"] * sensor_magnitude + sensor_mean
        subgoals_diff = (ideal_next_state_denormalized_rounded - observations_denormalized) * current_subgoal_masks
        subgoals_diff[:, 2] = wrap_to_pi(subgoals_diff[:, 2])
        subgoals_distance = torch.abs(subgoals_diff)
        subgoals_achieved = torch.all(subgoals_distance < 0.01, dim=1, keepdim=True)
        subgoals_done = torch.cat([subgoals_achieved,  # subgoals achieved
                                   current_subgoals_steps == args.time_scale,  # subgoals time up
                                   (1 - masks).byte()], dim=1)  # episode is done
        subgoals_done = torch.any(subgoals_done, dim=1, keepdim=True).float()
        subgoals_achieved = subgoals_achieved.float()

        prev_potential = prev_observations_denormalized - ideal_next_state_denormalized_rounded
        prev_cos_similarity = torch.cos(prev_potential[:, 2])
        prev_potential[:, 2] = 0
        prev_potential = torch.norm(prev_potential * current_subgoal_masks, p=1, dim=1, keepdim=True)
        prev_potential -= (prev_cos_similarity * current_subgoal_masks[:, 2]).unsqueeze(1)

        current_potential = observations_denormalized - ideal_next_state_denormalized_rounded
        current_cos_similarity = torch.cos(current_potential[:, 2])
        current_potential[:, 2] = 0
        current_potential = torch.norm(current_potential * current_subgoal_masks, p=1, dim=1, keepdim=True)
        current_potential -= (current_cos_similarity * current_subgoal_masks[:, 2]).unsqueeze(1)

        intrinsic_reward = 0.0
        intrinsic_reward += prev_potential - current_potential

        current_subgoal_reward += intrinsic_reward
        subgoal_rewards += subgoals_done * current_subgoal_reward
        subgoal_success_rates += subgoals_achieved
        subgoal_lengths += subgoals_done * current_subgoals_steps
        subgoal_counts += subgoals_done
        current_subgoal_reward *= (1 - subgoals_done)

        current_subgoals = (ideal_next_state - next_obs_batch["sensor"]) * current_subgoal_masks
        current_subgoals[:, 2] = wrap_to_pi(current_subgoals[:, 2] * np.pi) / np.pi
        current_subgoals_steps = (1 - subgoals_done) * current_subgoals_steps
        current_meta_recurrent_hidden_states = subgoals_done * next_meta_recurrent_hidden_states + \
                                               (1 - subgoals_done) * current_meta_recurrent_hidden_states

        if visualize:
            global_map = np.array([[" " for _ in range(env_width)] for _ in range(env_height)])
            global_map[0, :] = "-"
            global_map[-1, :] = "-"
            global_map[:, 0] = "|"
            global_map[:, -1] = "|"
            global_map[:, env_door_col] = "|"
            global_map[env_door_row, env_door_col] = str(int(prev_observations_denormalized[0, 3]))

            global_map[int(prev_auxiliary_sensor_denormalized[0, 2]),
                       int(prev_auxiliary_sensor_denormalized[0, 3])] = "G"
            ideal_next_state_valid = 0 <= int(ideal_next_state_denormalized_rounded[0, 0]) < env_height and \
                                     0 <= int(ideal_next_state_denormalized_rounded[0, 1]) < env_width
            if ideal_next_state_valid:
                global_map[int(ideal_next_state_denormalized_rounded[0, 0]),
                           int(ideal_next_state_denormalized_rounded[0, 1])] = "S"
            else:
                print("ideal next state oob: {}".format(ideal_next_state_denormalized_rounded[0, 0:2]))

            agent_tokens = ["<", "v", ">", "^"]
            agent = agent_tokens[int(torch.round((prev_observations_denormalized[0, 2] + np.pi) / (np.pi / 2.0))) % 4]
            global_map[int(prev_observations_denormalized[0, 0]),
                       int(prev_observations_denormalized[0, 1])] = agent

            action_mapping = [
                {
                    0: "no op",
                    1: "turn left",
                    2: "turn right",
                    3: "move forward",
                },
                {
                    0: "no op",
                    1: "slide up",
                    2: "slide down"
                }
            ]
            for row in global_map:
                print("".join(row))
            for v, m in zip(actions_np[0], action_mapping):
                print(m[v])
            print("subgoals", current_subgoals[0])
            print("subgoal_masks", current_subgoal_masks[0])
            print("action_mask", current_action_masks[0])
            print("ideal_next_state_denormalized_rounded", ideal_next_state_denormalized_rounded[0])
            print("ideal orientation", agent_tokens[int(torch.round((ideal_next_state_denormalized_rounded[0, 2] + np.pi) / (np.pi / 2.0))) % 4])
            print("subgoals diff", subgoals_diff[0])
            print("subgoal achieved", subgoals_achieved[0])
            print("subgoal_step: %d" % current_subgoals_steps[0])
            print("total_step: %d" % steps)
            input("Press Enter to continue...")
            os.system("clear")
            # sleep(0.3)

    episode_reward_mean = (episode_rewards.sum() / episode_counts.sum()).item()
    episode_success_rate_mean = (episode_success_rates.sum() / episode_counts.sum()).item()
    episode_length_mean = (episode_lengths.sum() / episode_counts.sum()).item()

    subgoal_reward_mean = (subgoal_rewards.sum() / subgoal_counts.sum()).item()
    subgoal_success_rate_mean = (subgoal_success_rates.sum() / subgoal_counts.sum()).item()
    subgoal_length_mean = (subgoal_lengths.sum() / subgoal_counts.sum()).item()

    if eval_only:
        print("EVAL: num_eval_episodes: {}\treward: {:.3f}\t"
              "success_rate: {:.3f}\tepisode_length: {:.3f}".format(
            args.num_eval_episodes, episode_reward_mean, episode_success_rate_mean, episode_length_mean))
        print("EVAL: num_eval_episodes: {}\tsubgoal_reward: {:.3f}\t"
              "subgoal_success_rate: {:.3f}\tsubgoal_length: {:.3f}".format(
            args.num_eval_episodes, subgoal_reward_mean, subgoal_success_rate_mean, subgoal_length_mean))
    else:
        logger.info("EVAL: num_eval_episodes: {}\tupdate: {}\t"
                    "reward: {:.3f}\tsuccess_rate: {:.3f}\tepisode_length: {:.3f}".format(
            args.num_eval_episodes, update, episode_reward_mean, episode_success_rate_mean, episode_length_mean))
        logger.info("EVAL: num_eval_episodes: {}\tupdate: {}\t"
                    "subgoal_reward: {:.3f}\tsubgoal_success_rate: {:.3f}\tsubgoal_length: {:.3f}".format(
            args.num_eval_episodes, update, subgoal_reward_mean, subgoal_success_rate_mean, subgoal_length_mean))
        writer.add_scalar("eval/updates/reward", episode_reward_mean, global_step=update)
        writer.add_scalar("eval/updates/success_rate", episode_success_rate_mean, global_step=update)
        writer.add_scalar("eval/updates/episode_length", episode_length_mean, global_step=update)
        writer.add_scalar("eval/env_steps/reward", episode_reward_mean, global_step=count_steps)
        writer.add_scalar("eval/env_steps/success_rate", episode_success_rate_mean, global_step=count_steps)
        writer.add_scalar("eval/env_steps/episode_length", episode_length_mean, global_step=count_steps)
        writer.add_scalar("eval/updates/subgoal_reward", subgoal_reward_mean, global_step=update)
        writer.add_scalar("eval/updates/subgoal_success_rate", subgoal_success_rate_mean, global_step=update)
        writer.add_scalar("eval/updates/subgoal_length", subgoal_length_mean, global_step=update)
        writer.add_scalar("eval/env_steps/subgoal_reward", subgoal_reward_mean, global_step=count_steps)
        writer.add_scalar("eval/env_steps/subgoal_success_rate", subgoal_success_rate_mean, global_step=count_steps)
        writer.add_scalar("eval/env_steps/subgoal_length", subgoal_length_mean, global_step=count_steps)


def wrap_to_pi(theta):
    return theta - np.pi * 2 * torch.floor((theta + np.pi) / (np.pi * 2))


def plot_action_mask(plot_env,
                     meta_actor_critic,
                     hidden_size,
                     device
                     ):
    meta_recurrent_hidden_states = torch.zeros(1, hidden_size, device=device)
    masks = torch.zeros(1, 1, device=device)
    base_heat_map = np.zeros((plot_env.height, plot_env.width))
    arm_heat_map = np.zeros((plot_env.height, plot_env.width))

    plot_env.reset()
    plot_env.agent_orientation = 3
    plot_env.target_pos = np.array([3, 5])
    for col in range(1, plot_env.width - 1):
        if col == plot_env.width // 2:
            continue
        if col > plot_env.width // 2:
            plot_env.door_state = plot_env.door_max_state
        for row in range(1, plot_env.height - 1):
            plot_env.agent_pos = np.array([row, col])
            observations = [plot_env.get_state()]
            batch = batch_obs(observations)
            for sensor in batch:
                batch[sensor] = batch[sensor].to(device)

            with torch.no_grad():
                _, _, _, _, _, _, action_mask_probs = meta_actor_critic.act(
                    batch,
                    meta_recurrent_hidden_states,
                    masks,
                )
            base_heat_map[row, col] = action_mask_probs[0][0].item() + action_mask_probs[0][2].item()
            arm_heat_map[row, col] = action_mask_probs[0][1].item() + action_mask_probs[0][2].item()
            print(row, col, action_mask_probs)
    print(arm_heat_map)
    plt.figure(0)
    plt.imshow(base_heat_map, cmap='hot', interpolation='nearest')
    plt.figure(1)
    plt.imshow(arm_heat_map, cmap='hot', interpolation='nearest')
    plt.show()
    assert False


def main():
    parser = argparse.ArgumentParser()
    add_ppo_args(parser)
    add_env_args(parser)
    add_common_args(parser)
    add_hrl_args(parser)
    args = parser.parse_args()

    ckpt_folder, ckpt_path, start_epoch, start_env_step, summary_folder, log_file = \
        set_up_experiment_folder(args.experiment_folder, args.checkpoint_index)

    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:{}".format(args.pth_gpu_id))
    logger.add_filehandler(log_file)
    writer = SummaryWriter(log_dir=summary_folder)

    for p in sorted(list(vars(args))):
        logger.info("{}: {}".format(p, getattr(args, p)))

    config_file = os.path.join(os.path.dirname(hrl4in.__file__), 'envs/toy_env', args.config_file)
    assert os.path.isfile(config_file), "config file does not exist: {}".format(config_file)

    env_config = parse_config(config_file)
    for (k, v) in env_config.items():
        logger.info("{}: {}".format(k, v))

    def load_env(env_mode, device_idx):
        return ToyEnv(config_file=config_file,
                      should_normalize_observation=True,
                      automatic_reset=True,
                      visualize=False)

    sim_gpu_id = [int(gpu_id) for gpu_id in args.sim_gpu_id.split(",")]
    env_id_to_which_gpu = np.linspace(0,
                                      len(sim_gpu_id),
                                      num=args.num_train_processes + args.num_eval_processes,
                                      dtype=np.int,
                                      endpoint=False)
    train_envs = [lambda device_idx=sim_gpu_id[env_id_to_which_gpu[env_id]]: load_env("headless", device_idx)
                  for env_id in range(args.num_train_processes)]
    train_envs = ParallelNavEnvironment(train_envs, blocking=False)
    eval_envs = [lambda device_idx=sim_gpu_id[env_id_to_which_gpu[env_id]]: load_env("headless", device_idx)
                 for env_id in range(args.num_train_processes, args.num_train_processes + args.num_eval_processes - 1)]
    eval_envs += [lambda: load_env(args.env_mode, sim_gpu_id[env_id_to_which_gpu[-1]])]
    eval_envs = ParallelNavEnvironment(eval_envs, blocking=False)

    print(train_envs.observation_space, train_envs.action_space)

    dummy_env = ToyEnv(config_file=config_file,
                       should_normalize_observation=True,
                       automatic_reset=True,
                       visualize=False)
    sensor_min_max = dummy_env.observation_space_min_max["sensor"]
    sensor_mean = np.mean(sensor_min_max, axis=1)
    sensor_magnitude = (sensor_min_max[:, 1] - sensor_min_max[:, 0]) / 2.0
    sensor_mean = torch.tensor(sensor_mean, device=device, dtype=torch.float)
    sensor_magnitude = torch.tensor(sensor_magnitude, device=device, dtype=torch.float)

    auxiliary_sensor_min_max = dummy_env.observation_space_min_max["auxiliary_sensor"]
    auxiliary_sensor_mean = np.mean(auxiliary_sensor_min_max, axis=1)
    auxiliary_sensor_magnitude = (auxiliary_sensor_min_max[:, 1] - auxiliary_sensor_min_max[:, 0]) / 2.0
    auxiliary_sensor_mean = torch.tensor(auxiliary_sensor_mean, device=device, dtype=torch.float)
    auxiliary_sensor_magnitude = torch.tensor(auxiliary_sensor_magnitude, device=device, dtype=torch.float)
    del dummy_env

    action_dim = train_envs.action_space.nvec.shape[0]
    action_mask_choices = torch.zeros(3, action_dim, device=device)
    action_mask_choices[0, 0] = 1.0  # base only
    action_mask_choices[1, 1] = 1.0  # arm only
    action_mask_choices[2, :] = 1.0  # base and arm

    env_width = env_config.get("width")
    env_height = env_config.get("height")
    env_door_max_state = env_config.get("door_max_state")
    env_door_row = env_config.get("door_row")
    env_door_col = env_config.get("door_col")

    # (output_channel, kernel_size, stride, padding)
    cnn_layers_params = [(32, 3, 1, 1), (32, 3, 1, 1), (32, 3, 1, 1)]

    meta_observation_space = train_envs.observation_space
    sensor_space = train_envs.observation_space.spaces["sensor"]

    subgoal_mask_choices = torch.zeros(3, sensor_space.shape[0], device=device)
    subgoal_mask_choices[0, 0:3] = 1.0  # base only
    subgoal_mask_choices[1, 3] = 1.0    # arm only
    subgoal_mask_choices[2, :] = 1.0    # base and arm

    # subgoal_space = sensor_space
    subgoal_space = gym.spaces.Box(low=sensor_space.low * 2.0, high=sensor_space.high * 2.0, dtype=np.float32)

    rollout_observation_space = train_envs.observation_space.spaces.copy()
    rollout_observation_space["subgoal"] = subgoal_space
    rollout_observation_space["subgoal_mask"] = gym.spaces.Box(low=0, high=1,
                                                               shape=subgoal_space.shape, dtype=np.float32)
    rollout_observation_space["action_mask"] = gym.spaces.Box(low=0, high=1,
                                                              shape=(action_dim,), dtype=np.float32)

    observation_space = rollout_observation_space.copy()
    # TODO: should LL policy observe "global map"?
    if "global_map" in observation_space:
        del observation_space["global_map"]
    rollout_observation_space = gym.spaces.Dict(rollout_observation_space)
    observation_space = gym.spaces.Dict(observation_space)

    min_stddev = np.zeros(subgoal_space.shape[0])
    min_stddev[0:2] = 1.0 / (max(env_width, env_height) - 1)
    min_stddev[2] = 1.0 / 4  # four directions
    min_stddev[3] = 1.0 / (env_door_max_state - 1)

    initial_stddev = min_stddev
    # initial_stddev = min_stddev * 1.2

    meta_actor_critic = MetaPolicy(
        observation_space=meta_observation_space,
        subgoal_space=subgoal_space,
        use_action_masks=args.use_action_masks,
        action_masks_dim=action_mask_choices.shape[0],
        hidden_size=args.hidden_size,
        cnn_layers_params=cnn_layers_params,
        initial_stddev=initial_stddev,
        min_stddev=min_stddev,
        stddev_transform=torch.nn.functional.softplus,
    )
    meta_actor_critic.to(device)

    meta_agent = PPO(
        meta_actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.meta_lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        is_meta_agent=True,
        normalize_advantage=args.meta_agent_normalize_advantage
    )

    actor_critic = Policy(
        observation_space=observation_space,
        action_space=train_envs.action_space,
        hidden_size=args.hidden_size,
        cnn_layers_params=cnn_layers_params,
    )
    actor_critic.to(device)

    agent = PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        is_meta_agent=False,
        normalize_advantage=True,
    )
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device)
        agent.load_state_dict(ckpt["state_dict"])
        logger.info("loaded checkpoint: {}".format(ckpt_path))

        ckpt_path = os.path.join(os.path.dirname(ckpt_path), os.path.basename(ckpt_path).replace("ckpt", "meta_ckpt"))
        ckpt = torch.load(ckpt_path, map_location=device)
        meta_agent.load_state_dict(ckpt["state_dict"])
        logger.info("loaded checkpoint: {}".format(ckpt_path))

    logger.info(
        "agent number of parameters: {}".format(
            sum(param.numel() for param in agent.parameters())
        )
    )
    logger.info(
        "meta agent number of parameters: {}".format(
            sum(param.numel() for param in meta_agent.parameters())
        )
    )

    # plot_env = load_env('headless', sim_gpu_id[env_id_to_which_gpu[-1]])
    # plot_action_mask(plot_env,
    #                  meta_actor_critic,
    #                  args.hidden_size,
    #                  device,
    #                  )

    if args.eval_only:
        evaluate(args,
                 eval_envs,
                 meta_actor_critic,
                 actor_critic,
                 sensor_mean,
                 sensor_magnitude,
                 auxiliary_sensor_mean,
                 auxiliary_sensor_magnitude,
                 action_mask_choices,
                 subgoal_mask_choices,
                 env_config,
                 device,
                 writer,
                 update=0,
                 count_steps=0,
                 eval_only=True,
                 visualize=True)
        return

    observations = train_envs.reset()

    batch = batch_obs(observations)

    meta_rollouts = AsyncRolloutStorage(
        args.num_steps,
        train_envs._num_envs,
        meta_observation_space,
        subgoal_space,
        args.hidden_size,
    )

    for sensor in meta_rollouts.observations:
        meta_rollouts.observations[sensor][0].copy_(batch[sensor])
    meta_rollouts.to(device)

    rollouts = RolloutStorage(
        args.num_steps,
        train_envs._num_envs,
        rollout_observation_space,
        train_envs.action_space,
        args.hidden_size,
    )

    for sensor in rollouts.observations:
        if sensor in batch:
            rollouts.observations[sensor][0].copy_(batch[sensor])
    rollouts.to(device)

    episode_rewards = torch.zeros(train_envs._num_envs, 1, device=device)
    episode_success_rates = torch.zeros(train_envs._num_envs, 1, device=device)
    episode_lengths = torch.zeros(train_envs._num_envs, 1, device=device)
    episode_counts = torch.zeros(train_envs._num_envs, 1, device=device)
    current_episode_reward = torch.zeros(train_envs._num_envs, 1, device=device)

    subgoal_rewards = torch.zeros(train_envs._num_envs, 1, device=device)
    subgoal_success_rates = torch.zeros(train_envs._num_envs, 1, device=device)
    subgoal_lengths = torch.zeros(train_envs._num_envs, 1, device=device)
    subgoal_counts = torch.zeros(train_envs._num_envs, 1, device=device)
    current_subgoal_reward = torch.zeros(train_envs._num_envs, 1, device=device)

    window_episode_reward = deque()
    window_episode_success_rates = deque()
    window_episode_lengths = deque()
    window_episode_counts = deque()

    window_subgoal_reward = deque()
    window_subgoal_success_rates = deque()
    window_subgoal_lengths = deque()
    window_subgoal_counts = deque()

    current_subgoals = torch.zeros(batch["sensor"].shape, device=device)
    current_subgoal_log_probs = torch.zeros(train_envs._num_envs, 1, device=device)
    current_meta_values = torch.zeros(train_envs._num_envs, 1, device=device)
    current_subgoals_steps = torch.zeros(train_envs._num_envs, 1, device=device)
    current_subgoals_cumulative_rewards = torch.zeros(train_envs._num_envs, 1, device=device)
    current_subgoals_penalty = torch.zeros(train_envs._num_envs, 1, device=device)
    subgoal_oob_penalty = -0.0
    subgoal_on_wall_penalty = -0.0
    original_subgoals = torch.zeros(batch["sensor"].shape, device=device)
    subgoal_round_to = torch.ones(batch["sensor"].shape, device=device)
    subgoal_round_to[:, 2] = np.pi / 2.0

    # hard-coded subgoal mask and observation mask
    current_subgoal_masks = torch.zeros(batch["sensor"].shape, device=device)
    current_action_masks = torch.zeros(train_envs._num_envs, action_dim, device=device)
    current_action_mask_indices = torch.zeros(train_envs._num_envs, 1, dtype=torch.long, device=device)
    current_action_mask_log_probs = torch.zeros(train_envs._num_envs, 1, device=device)
    next_meta_recurrent_hidden_states = torch.zeros(train_envs._num_envs, args.hidden_size, device=device)

    t_start = time()
    env_time = 0
    pth_time = 0
    count_steps = 0

    for update in range(start_epoch, args.num_updates):
        update_lr(agent.optimizer,
                  args.lr,
                  update,
                  args.num_updates,
                  args.use_linear_lr_decay,
                  0)
        update_lr(meta_agent.optimizer,
                  args.meta_lr,
                  update,
                  args.num_updates,
                  args.use_linear_lr_decay,
                  args.freeze_lr_n_updates)

        agent.clip_param = args.clip_param * (1 - update / args.num_updates)

        # collect num_steps tuples for each environment
        for step in range(args.num_steps):
            t_sample_action = time()
            with torch.no_grad():
                step_observation = {
                    k: v[step] for k, v in rollouts.observations.items()
                }
                meta_step_observation = {
                    k: v[meta_rollouts.valid_steps, meta_rollouts.env_indices] for k, v in
                    meta_rollouts.observations.items()
                }

                (
                    meta_values,
                    subgoals,
                    subgoal_log_probs,
                    action_mask_indices,
                    action_mask_log_probs,
                    meta_recurrent_hidden_states,
                ) = meta_actor_critic.act(
                    meta_step_observation,
                    meta_rollouts.recurrent_hidden_states[meta_rollouts.valid_steps, meta_rollouts.env_indices],
                    meta_rollouts.masks[meta_rollouts.valid_steps, meta_rollouts.env_indices],
                )

                prev_observations_denormalized = step_observation["sensor"] * sensor_magnitude + sensor_mean

                if args.use_action_masks:
                    action_masks = action_mask_choices.index_select(0, action_mask_indices.squeeze(1))
                    subgoal_masks = subgoal_mask_choices.index_select(0, action_mask_indices.squeeze(1))
                else:
                    action_masks = torch.ones_like(current_action_masks)
                    subgoal_masks = torch.ones_like(current_subgoal_masks)

                should_use_new_subgoals = (current_subgoals_steps == 0.0).float()
                current_subgoals = should_use_new_subgoals * subgoals + \
                                   (1 - should_use_new_subgoals) * current_subgoals
                current_subgoal_log_probs = should_use_new_subgoals * subgoal_log_probs + \
                                            (1 - should_use_new_subgoals) * current_subgoal_log_probs
                current_meta_values = should_use_new_subgoals * meta_values + \
                                      (1 - should_use_new_subgoals) * current_meta_values
                original_subgoals = should_use_new_subgoals * subgoals + \
                                    (1 - should_use_new_subgoals) * original_subgoals
                current_subgoal_masks = should_use_new_subgoals * subgoal_masks.float() + \
                                        (1 - should_use_new_subgoals) * current_subgoal_masks
                current_action_masks = should_use_new_subgoals * action_masks + \
                                       (1 - should_use_new_subgoals) * current_action_masks
                current_action_mask_indices = should_use_new_subgoals.long() * action_mask_indices + \
                                              (1 - should_use_new_subgoals.long()) * current_action_mask_indices
                current_action_mask_log_probs = should_use_new_subgoals * action_mask_log_probs + \
                                                (1 - should_use_new_subgoals) * current_action_mask_log_probs
                current_subgoals *= current_subgoal_masks
                next_meta_recurrent_hidden_states = should_use_new_subgoals * meta_recurrent_hidden_states + \
                                                    (1 - should_use_new_subgoals) * next_meta_recurrent_hidden_states

                # compute ideal next state before observation is masked
                ideal_next_state = step_observation["sensor"] + current_subgoals
                ideal_next_state_denormalized = ideal_next_state * sensor_magnitude + sensor_mean
                ideal_next_state_denormalized[:, 2] = wrap_to_pi(ideal_next_state_denormalized[:, 2])

                # ideal_next_state_denormalized_rounded = torch.round(ideal_next_state_denormalized)
                ideal_next_state_denormalized_rounded = torch.round(
                    ideal_next_state_denormalized / subgoal_round_to
                ) * subgoal_round_to

                ideal_next_state_oob = (ideal_next_state_denormalized_rounded[:, 0] < 0) | \
                                       (ideal_next_state_denormalized_rounded[:, 0] >= env_height) | \
                                       (ideal_next_state_denormalized_rounded[:, 1] < 0) | \
                                       (ideal_next_state_denormalized_rounded[:, 1] >= env_width)
                ideal_next_state_on_wall = (ideal_next_state_denormalized_rounded[:, 0] == 0) | \
                                           (ideal_next_state_denormalized_rounded[:, 0] == env_height - 1) | \
                                           (ideal_next_state_denormalized_rounded[:, 1] == 0) | \
                                           (ideal_next_state_denormalized_rounded[:, 1] == env_width - 1) | \
                                           ((ideal_next_state_denormalized_rounded[:, 1] == env_door_col) &
                                            (ideal_next_state_denormalized_rounded[:, 0] != env_door_row))
                new_subgoal_penalty = subgoal_oob_penalty * ideal_next_state_oob.float() + \
                                      subgoal_on_wall_penalty * ideal_next_state_on_wall.float()
                current_subgoals_penalty = should_use_new_subgoals * new_subgoal_penalty.unsqueeze(1) + \
                                           (1 - should_use_new_subgoals) * current_subgoals_penalty

                # transform subgoal in global coordinate to subgoal in local coordinate
                theta = step_observation["sensor"][:, 2] * np.pi
                rotation_matrix = torch.zeros((train_envs._num_envs, 2, 2), device=device)
                rotation_matrix[:, 0, 0] = torch.cos(-theta)
                rotation_matrix[:, 0, 1] = -torch.sin(-theta)
                rotation_matrix[:, 1, 0] = torch.sin(-theta)
                rotation_matrix[:, 1, 1] = torch.cos(-theta)
                current_subgoals_rotated = torch.bmm(rotation_matrix, current_subgoals[:, 0:2].unsqueeze(2)).squeeze(2)

                # full subgoal with rotation
                current_subgoals_observation = current_subgoals
                current_subgoals_observation[:, 0:2] = current_subgoals_rotated

                # mask observation and add current subgoal
                step_observation["subgoal"] = current_subgoals_observation
                step_observation["subgoal_mask"] = current_subgoal_masks
                step_observation["action_mask"] = current_action_masks

                rollouts.observations["subgoal"][step] = current_subgoals_observation
                rollouts.observations["subgoal_mask"][step] = current_subgoal_masks
                rollouts.observations["action_mask"][step] = current_action_masks

                # values: [num_processes, 1]
                # actions: [num_processes, 1]
                # actions_log_probs: [num_processes, 1]
                # recurrent_hidden_states: [num_processes, hidden_size]
                (
                    values,
                    actions,
                    actions_log_probs,
                    recurrent_hidden_states,
                ) = actor_critic.act(
                    step_observation,
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step],
                )
                actions_masked = actions * current_action_masks.long()

            pth_time += time() - t_sample_action

            t_step_env = time()

            actions_np = actions_masked.cpu().numpy()

            # outputs:
            # [
            #     (observation, reward, done, info),
            #     ...
            #     ...
            #     (observation, reward, done, info),
            # ]
            # len(outputs) == num_processes
            outputs = train_envs.step(actions_np)
            observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

            # because of auto reset, we have to get the next observation from info if this episode is done
            next_obs = [info["last_observation"] if done else obs for obs, done, info in
                        zip(observations, dones, infos)]

            env_time += time() - t_step_env

            t_update_stats = time()
            batch = batch_obs(observations)
            for sensor in batch:
                batch[sensor] = batch[sensor].to(device)

            next_obs_batch = batch_obs(next_obs)
            for sensor in next_obs_batch:
                next_obs_batch[sensor] = next_obs_batch[sensor].to(device)

            rewards = torch.tensor(rewards,
                                   dtype=torch.float,
                                   device=device)
            rewards = rewards.unsqueeze(1)
            masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=device
            )
            success_masks = torch.tensor(
                [[1.0] if done and "success" in info and info["success"] else [0.0]
                 for done, info in zip(dones, infos)],
                dtype=torch.float,
                device=device
            )
            lengths = torch.tensor(
                [[float(info["episode_length"])] if done and "episode_length" in info else [0.0]
                 for done, info in zip(dones, infos)],
                dtype=torch.float,
                device=device
            )

            current_episode_reward += rewards
            episode_rewards += (1 - masks) * current_episode_reward
            episode_success_rates += success_masks
            episode_lengths += lengths
            episode_counts += 1 - masks
            current_episode_reward *= masks

            current_subgoals_steps += 1
            current_subgoals_cumulative_rewards += rewards

            observations_denormalized = next_obs_batch["sensor"] * sensor_magnitude + sensor_mean
            subgoals_diff = (ideal_next_state_denormalized_rounded - observations_denormalized) * current_subgoal_masks
            subgoals_diff[:, 2] = wrap_to_pi(subgoals_diff[:, 2])
            subgoals_distance = torch.abs(subgoals_diff)
            subgoals_achieved = torch.all(subgoals_distance < 0.01, dim=1, keepdim=True)
            subgoals_done = torch.cat([subgoals_achieved,  # subgoals achieved
                                       current_subgoals_steps == args.time_scale,  # subgoals time up
                                       (1 - masks).byte()], dim=1)  # episode is done
            subgoals_done = torch.any(subgoals_done, dim=1, keepdim=True).float()
            subgoals_achieved = subgoals_achieved.float()
            failed_subgoals_penalty = -0.0
            current_subgoals_penalty += failed_subgoals_penalty * subgoals_done * (1.0 - subgoals_achieved)

            meta_rollouts.insert(
                subgoals_done,
                batch,
                next_meta_recurrent_hidden_states,
                original_subgoals,
                current_subgoal_log_probs,
                current_action_mask_indices,
                current_action_mask_log_probs,
                current_meta_values,
                current_subgoals_cumulative_rewards + current_subgoals_penalty,
                masks,
            )

            prev_potential = prev_observations_denormalized - ideal_next_state_denormalized_rounded
            prev_cos_similarity = torch.cos(prev_potential[:, 2])
            prev_potential[:, 2] = 0
            prev_potential = torch.norm(prev_potential * current_subgoal_masks, p=1, dim=1, keepdim=True)
            prev_potential -= (prev_cos_similarity * current_subgoal_masks[:, 2]).unsqueeze(1)

            current_potential = observations_denormalized - ideal_next_state_denormalized_rounded
            current_cos_similarity = torch.cos(current_potential[:, 2])
            current_potential[:, 2] = 0
            current_potential = torch.norm(current_potential * current_subgoal_masks, p=1, dim=1, keepdim=True)
            current_potential -= (current_cos_similarity * current_subgoal_masks[:, 2]).unsqueeze(1)

            intrinsic_reward = 0.0
            intrinsic_reward += prev_potential - current_potential

            # print('-' * 50)
            # print("obs pos", prev_observations_denormalized[0, :2])
            # print("ideal_next_state pos", ideal_next_state_denormalized[0, :2])
            # print("ideal next state rounded pos", ideal_next_state_denormalized_rounded[0, :2])
            # print("next obs pos", observations_denormalized[0, :2])
            #
            # print("obs ori", prev_observations_denormalized[0, 2])
            # print("ideal_next_state ori", ideal_next_state_denormalized[0, 2])
            # print("ideal next state rounded ori", ideal_next_state_denormalized_rounded[0, 2])
            # print("next obs ori", observations_denormalized[0, 2])
            #
            # print("obs door", prev_observations_denormalized[0, 3])
            # print("ideal_next_state door", ideal_next_state_denormalized[0, 3])
            # print("ideal next state rounded door", ideal_next_state_denormalized_rounded[0, 3])
            # print("next obs door", observations_denormalized[0, 3])
            #
            # print("actions", actions[0])
            # print("intrinsic_reward", intrinsic_reward[0])
            # print("extrinsic_reward", (current_subgoals_cumulative_rewards + current_subgoals_penalty)[0])
            # embed()

            current_subgoal_reward += intrinsic_reward
            subgoal_rewards += subgoals_done * current_subgoal_reward
            subgoal_success_rates += subgoals_achieved
            subgoal_lengths += subgoals_done * current_subgoals_steps
            subgoal_counts += subgoals_done
            current_subgoal_reward *= (1 - subgoals_done)

            # s_t+1 - batch["rgb"]: [num_processes, 256, 256, 3],
            # s_t+1 - batch["depth"]: [num_processes, 256, 256, 1]
            # s_t+1 - batch["pointgoal"]: [num_processes, 2]
            # h_t+1 - recurrent_hidden_states: [num_processes, hidden_size]
            # a_t - actions: [num_processes. 1]
            # a_t - action_log_probs: [num_processes. 1]
            # V(s_t) - values: [num_processes. 1]
            # r_t - rewards: [num_processes. 1]
            # mask_t+1 - masks: [num_processes. 1]
            rollouts.insert(
                batch,
                recurrent_hidden_states,
                actions,
                actions_log_probs,
                values,
                intrinsic_reward,
                1 - subgoals_done,
            )

            current_subgoals = (ideal_next_state - next_obs_batch["sensor"]) * current_subgoal_masks
            current_subgoals[:, 2] = wrap_to_pi(current_subgoals[:, 2] * np.pi) / np.pi
            current_subgoals_steps = (1 - subgoals_done) * current_subgoals_steps
            current_subgoals_cumulative_rewards = (1 - subgoals_done) * current_subgoals_cumulative_rewards

            count_steps += train_envs._num_envs
            pth_time += time() - t_update_stats

        if len(window_episode_reward) == args.perf_window_size:
            window_episode_reward.popleft()
            window_episode_success_rates.popleft()
            window_episode_lengths.popleft()
            window_episode_counts.popleft()
            window_subgoal_reward.popleft()
            window_subgoal_success_rates.popleft()
            window_subgoal_lengths.popleft()
            window_subgoal_counts.popleft()
        window_episode_reward.append(episode_rewards.clone())
        window_episode_success_rates.append(episode_success_rates.clone())
        window_episode_lengths.append(episode_lengths.clone())
        window_episode_counts.append(episode_counts.clone())
        window_subgoal_reward.append(subgoal_rewards.clone())
        window_subgoal_success_rates.append(subgoal_success_rates.clone())
        window_subgoal_lengths.append(subgoal_lengths.clone())
        window_subgoal_counts.append(subgoal_counts.clone())

        t_update_model = time()

        with torch.no_grad():
            last_meta_observation = {
                k: v[meta_rollouts.valid_steps, meta_rollouts.env_indices]
                for k, v in meta_rollouts.observations.items()
            }
            next_meta_value = meta_actor_critic.get_value(
                last_meta_observation,
                meta_rollouts.recurrent_hidden_states[meta_rollouts.valid_steps, meta_rollouts.env_indices],
                meta_rollouts.masks[meta_rollouts.valid_steps, meta_rollouts.env_indices],
            ).detach()

            last_observation = {
                k: v[-1].clone() for k, v in rollouts.observations.items()
            }
            # transform subgoal in global coordinate to subgoal in local coordinate
            theta = rollouts.observations["sensor"][-1][:, 2] * np.pi
            rotation_matrix = torch.zeros((train_envs._num_envs, 2, 2), device=device)
            rotation_matrix[:, 0, 0] = torch.cos(-theta)
            rotation_matrix[:, 0, 1] = -torch.sin(-theta)
            rotation_matrix[:, 1, 0] = torch.sin(-theta)
            rotation_matrix[:, 1, 1] = torch.cos(-theta)
            current_subgoals_rotated = torch.bmm(rotation_matrix, current_subgoals[:, 0:2].unsqueeze(2)).squeeze(2)

            # full subgoal with rotation
            current_subgoals_observation = current_subgoals.clone()
            current_subgoals_observation[:, 0:2] = current_subgoals_rotated

            last_observation["subgoal"] = current_subgoals_observation
            last_observation["subgoal_mask"] = current_subgoal_masks
            last_observation["action_mask"] = current_action_masks

            next_value = actor_critic.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1],
            ).detach()

        meta_rollouts.compute_returns(
            next_meta_value, args.use_gae, args.meta_gamma, args.tau
        )

        # V(s_t+num_steps) - next_value: [num_processes, 1]
        rollouts.compute_returns(
            next_value, args.use_gae, args.gamma, args.tau
        )

        meta_value_loss, subgoal_loss, meta_dist_entropy = meta_agent.update(meta_rollouts)
        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        meta_rollouts.after_update()
        rollouts.after_update()
        pth_time += time() - t_update_model

        # log stats
        if update > 0 and update % args.log_interval == 0:
            logger.info(
                "update: {}\tenv_steps: {}\tenv_steps_per_sec: {:.3f}\tenv-time: {:.3f}s\tpth-time: {:.3f}s".format(
                    update, count_steps, count_steps / (time() - t_start), env_time, pth_time
                )
            )
            logger.info(
                "update: {}\tenv_steps: {}\tvalue_loss: {:.3f}\taction_loss: {:.3f}\tdist_entropy: {:.3f}".format(
                    update, count_steps, value_loss, action_loss, dist_entropy
                )
            )
            logger.info(
                "update: {}\tenv_steps: {}\tmeta_value_loss: {:.3f}\tsubgoal_loss: {:.3f}\t"
                "meta_dist_entropy: {:.3f}".format(
                    update, count_steps, meta_value_loss, subgoal_loss, meta_dist_entropy
                )
            )
            writer.add_scalar("time/env_step_per_second", count_steps / (time() - t_start), global_step=update)
            writer.add_scalar("time/env_time_per_update", env_time / update, global_step=update)
            writer.add_scalar("time/pth_time_per_update", pth_time / update, global_step=update)
            writer.add_scalar("time/env_steps_per_update", count_steps / update,
                              global_step=update)
            writer.add_scalar("losses/value_loss", value_loss, global_step=update)
            writer.add_scalar("losses/action_loss", action_loss, global_step=update)
            writer.add_scalar("losses/dist_entropy", dist_entropy, global_step=update)
            writer.add_scalar("losses/meta_value_loss", meta_value_loss, global_step=update)
            writer.add_scalar("losses/subgoal_loss", subgoal_loss, global_step=update)
            writer.add_scalar("losses/meta_dist_entropy", meta_dist_entropy, global_step=update)

            window_rewards = (window_episode_reward[-1] - window_episode_reward[0]).sum()
            window_success_rates = (window_episode_success_rates[-1] - window_episode_success_rates[0]).sum()
            window_lengths = (window_episode_lengths[-1] - window_episode_lengths[0]).sum()
            window_counts = (window_episode_counts[-1] - window_episode_counts[0]).sum()
            if window_counts > 0:
                reward_mean = (window_rewards / window_counts).item()
                success_rate_mean = (window_success_rates / window_counts).item()
                lengths_mean = (window_lengths / window_counts).item()
                logger.info(
                    "average window size {}\treward: {:3f}\tsuccess_rate: {:3f}\tepisode length: {:3f}".format(
                        len(window_episode_reward),
                        reward_mean,
                        success_rate_mean,
                        lengths_mean,
                    )
                )
                writer.add_scalar("train/updates/reward", reward_mean, global_step=update)
                writer.add_scalar("train/updates/success_rate", success_rate_mean, global_step=update)
                writer.add_scalar("train/updates/episode_length", lengths_mean, global_step=update)
                writer.add_scalar("train/env_steps/reward", reward_mean, global_step=count_steps)
                writer.add_scalar("train/env_steps/success_rate", success_rate_mean, global_step=count_steps)
                writer.add_scalar("train/env_steps/episode_length", lengths_mean, global_step=count_steps)
            else:
                logger.info("No episodes finish in current window")

            window_rewards = (window_subgoal_reward[-1] - window_subgoal_reward[0]).sum()
            window_success_rates = (window_subgoal_success_rates[-1] - window_subgoal_success_rates[0]).sum()
            window_lengths = (window_subgoal_lengths[-1] - window_subgoal_lengths[0]).sum()
            window_counts = (window_subgoal_counts[-1] - window_subgoal_counts[0]).sum()

            if window_counts > 0:
                reward_mean = (window_rewards / window_counts).item()
                success_rate_mean = (window_success_rates / window_counts).item()
                lengths_mean = (window_lengths / window_counts).item()
                logger.info(
                    "window_size: {}\tsubgoal_reward: {:3f}\tsubgoal_success_rate: {:3f}\tsubgoal_length: {:3f}".format(
                        len(window_subgoal_reward),
                        reward_mean,
                        success_rate_mean,
                        lengths_mean,
                    )
                )
                writer.add_scalar("train/updates/subgoal_reward", reward_mean, global_step=update)
                writer.add_scalar("train/updates/subgoal_success_rate", success_rate_mean, global_step=update)
                writer.add_scalar("train/updates/subgoal_length", lengths_mean, global_step=update)
                writer.add_scalar("train/env_steps/subgoal_reward", reward_mean, global_step=count_steps)
                writer.add_scalar("train/env_steps/subgoal_success_rate", success_rate_mean, global_step=count_steps)
                writer.add_scalar("train/env_steps/subgoal_length", lengths_mean, global_step=count_steps)
            else:
                logger.info("No subgoals finish in current window")

        # checkpoint model
        if update > 0 and update % args.checkpoint_interval == 0:
            checkpoint = {"state_dict": agent.state_dict()}
            torch.save(
                checkpoint,
                os.path.join(
                    ckpt_folder,
                    "ckpt.{}.pth".format(update),
                ),
            )

            checkpoint = {"state_dict": meta_agent.state_dict()}
            torch.save(
                checkpoint,
                os.path.join(
                    ckpt_folder,
                    "meta_ckpt.{}.pth".format(update),
                ),
            )

        # if update > 0 and update % args.eval_interval == 0:
        #     evaluate(args,
        #              eval_envs,
        #              meta_actor_critic,
        #              actor_critic,
        #              sensor_mean,
        #              sensor_magnitude,
        #              auxiliary_sensor_mean,
        #              auxiliary_sensor_magnitude,
        #              action_mask_choices,
        #              subgoal_mask_choices,
        #              env_config,
        #              device,
        #              writer,
        #              update=update,
        #              count_steps=count_steps,
        #              eval_only=False,
        #              visualize=False)


if __name__ == "__main__":
    main()
