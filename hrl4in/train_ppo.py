#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from time import time
from collections import deque
import random
import numpy as np
import sys
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

import hrl4in
from hrl4in.envs.toy_env.toy_env import ToyEnv
from hrl4in.utils.logging import logger
from hrl4in.rl.ppo import PPO, Policy, RolloutStorage
from hrl4in.utils.utils import *
from hrl4in.utils.args import *

import gibson2
from gibson2.envs.parallel_env import ParallelNavEnvironment
from gibson2.envs.locomotor_env import NavigateEnv, NavigateRandomEnv, InteractiveNavigateEnv



def evaluate(envs,
             actor_critic,
             hidden_size,
             num_eval_episodes,
             device,
             writer,
             update=0,
             count_steps=0,
             eval_only=False):
    observations = envs.reset()
    batch = batch_obs(observations)
    for sensor in batch:
        batch[sensor] = batch[sensor].to(device)

    episode_rewards = torch.zeros(envs._num_envs, 1, device=device)
    episode_success_rates = torch.zeros(envs._num_envs, 1, device=device)
    episode_lengths = torch.zeros(envs._num_envs, 1, device=device)
    episode_collision_steps = torch.zeros(envs._num_envs, 1, device=device)
    episode_total_energy_costs = torch.zeros(envs._num_envs, 1, device=device)
    episode_avg_energy_costs = torch.zeros(envs._num_envs, 1, device=device)
    episode_stage_open_door = torch.zeros(envs._num_envs, 1, device=device)
    episode_stage_to_target = torch.zeros(envs._num_envs, 1, device=device)

    episode_counts = torch.zeros(envs._num_envs, 1, device=device)
    current_episode_reward = torch.zeros(envs._num_envs, 1, device=device)

    recurrent_hidden_states = torch.zeros(envs._num_envs, hidden_size, device=device)
    masks = torch.zeros(envs._num_envs, 1, device=device)

    while episode_counts.sum() < num_eval_episodes:
        with torch.no_grad():
            _, actions, _, recurrent_hidden_states = actor_critic.act(
                batch,
                recurrent_hidden_states,
                masks,
                deterministic=True,
                update=0,
            )
        actions_np = actions.cpu().numpy()
        outputs = envs.step(actions_np)

        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        batch = batch_obs(observations)
        for sensor in batch:
            batch[sensor] = batch[sensor].to(device)
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
        collision_steps = torch.tensor(
            [[float(info["collision_step"])] if done and "collision_step" in info else [0.0]
             for done, info in zip(dones, infos)],
            dtype=torch.float,
            device=device
        )
        total_energy_cost = torch.tensor(
            [[float(info["energy_cost"])] if done and "energy_cost" in info else [0.0]
             for done, info in zip(dones, infos)],
            dtype=torch.float,
            device=device
        )
        avg_energy_cost = torch.tensor(
            [[float(info["energy_cost"]) / float(info["episode_length"])]
             if done and "energy_cost" in info and "episode_length" in info
             else [0.0]
             for done, info in zip(dones, infos)],
            dtype=torch.float,
            device=device
        )
        stage_open_door = torch.tensor(
            [[float(info["stage"] >= 1)] if done and "stage" in info else [0.0]
             for done, info in zip(dones, infos)],
            dtype=torch.float,
            device=device
        )
        stage_to_target = torch.tensor(
            [[float(info["stage"] >= 2)] if done and "stage" in info else [0.0]
             for done, info in zip(dones, infos)],
            dtype=torch.float,
            device=device
        )

        current_episode_reward += rewards
        episode_rewards += (1 - masks) * current_episode_reward
        episode_success_rates += success_masks
        episode_lengths += lengths
        episode_collision_steps += collision_steps
        episode_total_energy_costs += total_energy_cost
        episode_avg_energy_costs += avg_energy_cost
        episode_stage_open_door += stage_open_door
        episode_stage_to_target += stage_to_target
        episode_counts += 1 - masks
        current_episode_reward *= masks

    episode_reward_mean = (episode_rewards.sum() / episode_counts.sum()).item()
    episode_success_rate_mean = (episode_success_rates.sum() / episode_counts.sum()).item()
    episode_length_mean = (episode_lengths.sum() / episode_counts.sum()).item()
    episode_collision_step_mean = (episode_collision_steps.sum() / episode_counts.sum()).item()
    episode_total_energy_cost_mean = (episode_total_energy_costs.sum() / episode_counts.sum()).item()
    episode_avg_energy_cost_mean = (episode_avg_energy_costs.sum() / episode_counts.sum()).item()
    episode_stage_open_door_mean = (episode_stage_open_door.sum() / episode_counts.sum()).item()
    episode_stage_to_target_mean = (episode_stage_to_target.sum() / episode_counts.sum()).item()

    if eval_only:
        print("EVAL: num_eval_episodes: {}\treward: {:.3f}\t"
              "success_rate: {:.3f}\tepisode_length: {:.3f}\tcollision_step: {:.3f}\t"
              "total_energy_cost: {:.3f}\tavg_energy_cost: {:.3f}\t"
              "stage_open_door: {:.3f}\tstage_to_target: {:.3f}".format(
            num_eval_episodes, episode_reward_mean, episode_success_rate_mean, episode_length_mean,
            episode_collision_step_mean, episode_total_energy_cost_mean, episode_avg_energy_cost_mean,
            episode_stage_open_door_mean, episode_stage_to_target_mean,
        ))
    else:
        logger.info("EVAL: num_eval_episodes: {}\tupdate: {}\t"
                    "reward: {:.3f}\tsuccess_rate: {:.3f}\tepisode_length: {:.3f}\tcollision_step: {:.3f}".format(
            num_eval_episodes, update, episode_reward_mean, episode_success_rate_mean, episode_length_mean,
            episode_collision_step_mean))
        writer.add_scalar("eval/updates/reward", episode_reward_mean, global_step=update)
        writer.add_scalar("eval/updates/success_rate", episode_success_rate_mean, global_step=update)
        writer.add_scalar("eval/updates/episode_length", episode_length_mean, global_step=update)
        writer.add_scalar("eval/updates/collision_step", episode_collision_step_mean, global_step=update)
        writer.add_scalar("eval/updates/total_energy_cost", episode_total_energy_cost_mean, global_step=update)
        writer.add_scalar("eval/updates/avg_energy_cost", episode_avg_energy_cost_mean, global_step=update)
        writer.add_scalar("eval/updates/stage_open_door", episode_stage_open_door_mean, global_step=update)
        writer.add_scalar("eval/updates/stage_to_target", episode_stage_to_target_mean, global_step=update)

        writer.add_scalar("eval/env_steps/reward", episode_reward_mean, global_step=count_steps)
        writer.add_scalar("eval/env_steps/success_rate", episode_success_rate_mean, global_step=count_steps)
        writer.add_scalar("eval/env_steps/episode_length", episode_length_mean, global_step=count_steps)
        writer.add_scalar("eval/env_steps/collision_step", episode_collision_step_mean, global_step=count_steps)
        writer.add_scalar("eval/env_steps/total_energy_cost", episode_total_energy_cost_mean, global_step=count_steps)
        writer.add_scalar("eval/env_steps/avg_energy_cost", episode_avg_energy_cost_mean, global_step=count_steps)
        writer.add_scalar("eval/env_steps/stage_open_door", episode_stage_open_door_mean, global_step=count_steps)
        writer.add_scalar("eval/env_steps/stage_to_target", episode_stage_to_target_mean, global_step=count_steps)


def main():
    parser = argparse.ArgumentParser()
    add_ppo_args(parser)
    add_env_args(parser)
    add_common_args(parser)
    args = parser.parse_args()

    ckpt_folder, ckpt_path, start_epoch, start_env_step, summary_folder, log_file = \
        set_up_experiment_folder(args.experiment_folder, args.checkpoint_index)

    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:{}".format(args.pth_gpu_id))
    logger.add_filehandler(log_file)

    if not args.eval_only:
        writer = SummaryWriter(log_dir=summary_folder)
    else:
        writer = None

    for p in sorted(list(vars(args))):
        logger.info("{}: {}".format(p, getattr(args, p)))

    if args.env_type == "gibson" or args.env_type == "interactive_gibson":
        config_file = os.path.join(os.path.dirname(gibson2.__file__), "../examples/configs", args.config_file)
    elif args.env_type == "toy":
        config_file = os.path.join(os.path.dirname(hrl4in.__file__), 'envs/toy_env', args.config_file)

    assert os.path.isfile(config_file), "config file does not exist: {}".format(config_file)

    for (k, v) in parse_config(config_file).items():
        logger.info("{}: {}".format(k, v))

    def load_env(env_mode, device_idx):
        if args.env_type == "gibson":
            if args.random_position:
                return NavigateRandomEnv(config_file=config_file,
                                         mode=env_mode,
                                         action_timestep=args.action_timestep,
                                         physics_timestep=args.physics_timestep,
                                         random_height=args.random_height,
                                         automatic_reset=True,
                                         device_idx=device_idx)
            else:
                return NavigateEnv(config_file=config_file,
                                   mode=env_mode,
                                   action_timestep=args.action_timestep,
                                   physics_timestep=args.physics_timestep,
                                   automatic_reset=True,
                                   device_idx=device_idx)
        elif args.env_type == "interactive_gibson":
            return InteractiveNavigateEnv(config_file=config_file,
                                          mode=env_mode,
                                          action_timestep=args.action_timestep,
                                          physics_timestep=args.physics_timestep,
                                          automatic_reset=True,
                                          random_position=args.random_position,
                                          device_idx=device_idx)
        elif args.env_type == "toy":
            return ToyEnv(config_file=config_file,
                          should_normalize_state=True,
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

    # (output_channel, kernel_size, stride, padding)
    if args.env_type == "gibson" or args.env_type == "interactive_gibson":
        cnn_layers_params = [(32, 8, 4, 0), (64, 4, 2, 0), (64, 3, 1, 0)]
    elif args.env_type == "toy":
        cnn_layers_params = [(32, 3, 1, 1), (32, 3, 1, 1), (32, 3, 1, 1)]

    actor_critic = Policy(
        observation_space=train_envs.observation_space,
        action_space=train_envs.action_space,
        hidden_size=args.hidden_size,
        cnn_layers_params=cnn_layers_params,
        initial_stddev=args.action_init_std_dev,
        min_stddev=args.action_min_std_dev,
        stddev_anneal_schedule=args.action_std_dev_anneal_schedule,
        stddev_transform=torch.nn.functional.softplus,
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
        use_clipped_value_loss=True
    )
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device)
        agent.load_state_dict(ckpt["state_dict"])
        logger.info("loaded checkpoing: {}".format(ckpt_path))

    logger.info(
        "agent number of parameters: {}".format(
            sum(param.numel() for param in agent.parameters())
        )
    )

    if args.eval_only:
        evaluate(eval_envs,
                 actor_critic,
                 args.hidden_size,
                 args.num_eval_episodes,
                 device,
                 writer,
                 update=0,
                 count_steps=0,
                 eval_only=True)
        return

    observations = train_envs.reset()

    batch = batch_obs(observations)

    rollouts = RolloutStorage(
        args.num_steps,
        train_envs._num_envs,
        train_envs.observation_space,
        train_envs.action_space,
        args.hidden_size,
    )

    for sensor in rollouts.observations:
        rollouts.observations[sensor][0].copy_(batch[sensor])
    rollouts.to(device)

    episode_rewards = torch.zeros(train_envs._num_envs, 1)
    episode_success_rates = torch.zeros(train_envs._num_envs, 1)
    episode_lengths = torch.zeros(train_envs._num_envs, 1)
    episode_collision_steps = torch.zeros(train_envs._num_envs, 1)
    episode_total_energy_costs = torch.zeros(train_envs._num_envs, 1, device=device)
    episode_avg_energy_costs = torch.zeros(train_envs._num_envs, 1, device=device)
    episode_stage_open_doors = torch.zeros(train_envs._num_envs, 1, device=device)
    episode_stage_to_targets = torch.zeros(train_envs._num_envs, 1, device=device)
    episode_counts = torch.zeros(train_envs._num_envs, 1)
    current_episode_reward = torch.zeros(train_envs._num_envs, 1)

    window_episode_reward = deque()
    window_episode_success_rates = deque()
    window_episode_lengths = deque()
    window_episode_collision_steps = deque()
    window_episode_total_energy_costs = deque()
    window_episode_avg_energy_costs = deque()
    window_episode_stage_open_doors = deque()
    window_episode_stage_to_targets = deque()
    window_episode_counts = deque()

    t_start = time()
    env_time = 0
    pth_time = 0
    count_steps = start_env_step

    for update in range(start_epoch, args.num_updates):
        update_lr(agent.optimizer, args.lr, update, args.num_updates, args.use_linear_lr_decay, 0)

        agent.clip_param = args.clip_param * (1 - update / args.num_updates)

        # collect num_steps tuples for each environment
        for step in range(args.num_steps):
            t_sample_action = time()
            # sample actions
            with torch.no_grad():
                step_observation = {
                    k: v[step] for k, v in rollouts.observations.items()
                }

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
                    update=update,
                )
            pth_time += time() - t_sample_action

            t_step_env = time()

            actions_np = actions.cpu().numpy()
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
            env_time += time() - t_step_env

            t_update_stats = time()
            batch = batch_obs(observations)
            rewards = torch.tensor(rewards, dtype=torch.float)
            rewards = rewards.unsqueeze(1)
            masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones], dtype=torch.float
            )
            success_masks = torch.tensor(
                [[1.0] if done and "success" in info and info["success"] else [0.0]
                 for done, info in zip(dones, infos)],
                dtype=torch.float
            )
            lengths = torch.tensor(
                [[float(info["episode_length"])] if done and "episode_length" in info else [0.0]
                 for done, info in zip(dones, infos)],
                dtype=torch.float,
            )
            collision_steps = torch.tensor(
                [[float(info["collision_step"])] if done and "collision_step" in info else [0.0]
                 for done, info in zip(dones, infos)],
                dtype=torch.float,
            )
            total_energy_cost = torch.tensor(
                [[float(info["energy_cost"])] if done and "energy_cost" in info else [0.0]
                 for done, info in zip(dones, infos)],
                dtype=torch.float,
                device=device
            )
            avg_energy_cost = torch.tensor(
                [[float(info["energy_cost"]) / float(info["episode_length"])]
                 if done and "energy_cost" in info and "episode_length" in info
                 else [0.0]
                 for done, info in zip(dones, infos)],
                dtype=torch.float,
                device=device
            )
            stage_open_door = torch.tensor(
                [[float(info["stage"] >= 1)] if done and "stage" in info else [0.0]
                 for done, info in zip(dones, infos)],
                dtype=torch.float,
                device=device
            )
            stage_to_target = torch.tensor(
                [[float(info["stage"] >= 2)] if done and "stage" in info else [0.0]
                 for done, info in zip(dones, infos)],
                dtype=torch.float,
                device=device
            )

            current_episode_reward += rewards
            episode_rewards += (1 - masks) * current_episode_reward
            episode_success_rates += success_masks
            episode_lengths += lengths
            episode_collision_steps += collision_steps
            episode_total_energy_costs += total_energy_cost
            episode_avg_energy_costs += avg_energy_cost
            episode_stage_open_doors += stage_open_door
            episode_stage_to_targets += stage_to_target
            episode_counts += 1 - masks
            current_episode_reward *= masks

            # s_t+1 - batch["rgb"]: [num_processes, 256, 256, 3],
            # s_t+1 - batch["depth"]: [num_processes, 256, 256, 1]
            # s_t+1 - batch["pointgoal"]: [num_processes, 2]
            # h_t+1 - recurrent_hidden_states: [num_processes, hidden_size]
            # a_t - actions: [num_processes. 1]
            # a_t - action_log_probs: [num_processes. 1]
            # V(s_t) - values: [num_processes. 1]
            # r_t - rewards: [num_processes. 1]
            # mask_t+1 - masks: [[num_processes. 1]
            rollouts.insert(
                batch,
                recurrent_hidden_states,
                actions,
                actions_log_probs,
                values,
                rewards,
                masks,
            )

            count_steps += train_envs._num_envs
            pth_time += time() - t_update_stats

        if len(window_episode_reward) == args.perf_window_size:
            window_episode_reward.popleft()
            window_episode_success_rates.popleft()
            window_episode_lengths.popleft()
            window_episode_collision_steps.popleft()
            window_episode_total_energy_costs.popleft()
            window_episode_avg_energy_costs.popleft()
            window_episode_stage_open_doors.popleft()
            window_episode_stage_to_targets.popleft()
            window_episode_counts.popleft()
        window_episode_reward.append(episode_rewards.clone())
        window_episode_success_rates.append(episode_success_rates.clone())
        window_episode_lengths.append(episode_lengths.clone())
        window_episode_collision_steps.append(episode_collision_steps.clone())
        window_episode_total_energy_costs.append(episode_total_energy_costs.clone())
        window_episode_avg_energy_costs.append(episode_avg_energy_costs.clone())
        window_episode_stage_open_doors.append(episode_stage_open_doors.clone())
        window_episode_stage_to_targets.append(episode_stage_to_targets.clone())
        window_episode_counts.append(episode_counts.clone())

        t_update_model = time()
        with torch.no_grad():
            last_observation = {
                k: v[-1] for k, v in rollouts.observations.items()
            }
            next_value = actor_critic.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1],
            ).detach()

        # V(s_t+num_steps) - next_value: [num_processes, 1]
        rollouts.compute_returns(
            next_value, args.use_gae, args.gamma, args.tau
        )

        value_loss, action_loss, dist_entropy = agent.update(rollouts, update=update)

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
            writer.add_scalar("time/env_step_per_second", count_steps / (time() - t_start), global_step=update)
            writer.add_scalar("time/env_time_per_update", env_time / update, global_step=update)
            writer.add_scalar("time/pth_time_per_update", pth_time / update, global_step=update)
            writer.add_scalar("time/env_steps_per_update", count_steps / update,
                              global_step=update)
            writer.add_scalar("losses/value_loss", value_loss, global_step=update)
            writer.add_scalar("losses/action_loss", action_loss, global_step=update)
            writer.add_scalar("losses/dist_entropy", dist_entropy, global_step=update)

            window_rewards = (window_episode_reward[-1] - window_episode_reward[0]).sum()
            window_success_rates = (window_episode_success_rates[-1] - window_episode_success_rates[0]).sum()
            window_lengths = (window_episode_lengths[-1] - window_episode_lengths[0]).sum()
            window_collision_steps = (window_episode_collision_steps[-1] - window_episode_collision_steps[0]).sum()
            window_total_energy_costs = (window_episode_total_energy_costs[-1] - window_episode_total_energy_costs[0]).sum()
            window_avg_energy_costs = (window_episode_avg_energy_costs[-1] - window_episode_avg_energy_costs[0]).sum()
            window_stage_open_doors = (window_episode_stage_open_doors[-1] - window_episode_stage_open_doors[0]).sum()
            window_stage_to_targets = (window_episode_stage_to_targets[-1] - window_episode_stage_to_targets[0]).sum()
            window_counts = (window_episode_counts[-1] - window_episode_counts[0]).sum()

            if window_counts > 0:
                reward_mean = (window_rewards / window_counts).item()
                success_rate_mean = (window_success_rates / window_counts).item()
                lengths_mean = (window_lengths / window_counts).item()
                collision_steps_mean = (window_collision_steps / window_counts).item()
                total_energy_costs_mean = (window_total_energy_costs / window_counts).item()
                avg_energy_costs_mean = (window_avg_energy_costs / window_counts).item()
                stage_open_doors_mean = (window_stage_open_doors / window_counts).item()
                stage_to_targets_mean = (window_stage_to_targets / window_counts).item()

                logger.info(
                    "average window size {}\treward: {:.3f}\tsuccess_rate: {:.3f}\tepisode length: {:.3f}\t"
                    "collision_step: {:.3f}\ttotal_energy_cost: {:.3f}\tavg_energy_cost: {:.3f}\t"
                    "stage_open_door: {:.3f}\tstage_to_target: {:.3f}".format(
                        len(window_episode_reward),
                        reward_mean,
                        success_rate_mean,
                        lengths_mean,
                        collision_steps_mean,
                        total_energy_costs_mean,
                        avg_energy_costs_mean,
                        stage_open_doors_mean,
                        stage_to_targets_mean,
                    )
                )
                writer.add_scalar("train/updates/reward", reward_mean, global_step=update)
                writer.add_scalar("train/updates/success_rate", success_rate_mean, global_step=update)
                writer.add_scalar("train/updates/episode_length", lengths_mean, global_step=update)
                writer.add_scalar("train/updates/collision_step", collision_steps_mean, global_step=update)
                writer.add_scalar("train/updates/total_energy_cost", total_energy_costs_mean, global_step=update)
                writer.add_scalar("train/updates/avg_energy_cost", avg_energy_costs_mean, global_step=update)
                writer.add_scalar("train/updates/stage_open_door", stage_open_doors_mean, global_step=update)
                writer.add_scalar("train/updates/stage_to_target", stage_to_targets_mean, global_step=update)

                writer.add_scalar("train/env_steps/reward", reward_mean, global_step=count_steps)
                writer.add_scalar("train/env_steps/success_rate", success_rate_mean, global_step=count_steps)
                writer.add_scalar("train/env_steps/episode_length", lengths_mean, global_step=count_steps)
                writer.add_scalar("train/env_steps/collision_step", collision_steps_mean, global_step=count_steps)
                writer.add_scalar("train/env_steps/total_energy_cost", total_energy_costs_mean, global_step=count_steps)
                writer.add_scalar("train/env_steps/avg_energy_cost", avg_energy_costs_mean, global_step=count_steps)
                writer.add_scalar("train/env_steps/stage_open_door", stage_open_doors_mean, global_step=count_steps)
                writer.add_scalar("train/env_steps/stage_to_target", stage_to_targets_mean, global_step=count_steps)
            else:
                logger.info("No episodes finish in current window")

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

        if update > 0 and update % args.eval_interval == 0:
            evaluate(eval_envs,
                     actor_critic,
                     args.hidden_size,
                     args.num_eval_episodes,
                     device,
                     writer,
                     update=update,
                     count_steps=count_steps,
                     eval_only=False)


if __name__ == "__main__":
    main()
