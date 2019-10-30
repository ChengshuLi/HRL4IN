import numpy as np
import os
from datetime import datetime
from collections import defaultdict
import torch
import yaml


def parse_config(config):
    with open(config, 'r') as f:
        config_data = yaml.load(f)
    return config_data


def update_lr(optimizer, initial_lr, update, num_updates, use_linear_lr_decay, freeze_lr_n_updates):
    if update < freeze_lr_n_updates:
        lr = 0.0
    elif use_linear_lr_decay:
        lr = initial_lr * (1.0 - (update / float(num_updates)))
    else:
        lr = initial_lr
    assert lr >= 0.0, 'lr is negative'

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def batch_obs(observations):
    batch = defaultdict(list)

    for obs in observations:
        for sensor in obs:
            batch[sensor].append(obs[sensor])

    for sensor in batch:
        batch[sensor] = torch.tensor(np.array(batch[sensor]), dtype=torch.float)
    return batch


def rotate_torch_vector(vector, roll, pitch, yaw):
    num_envs = vector.shape[0]
    device = vector.device
    rot_x = torch.zeros((num_envs, 3, 3), device=device)
    rot_x[:, 0, 0] = 1.0
    rot_x[:, 0, 1] = 0.0
    rot_x[:, 0, 2] = 0.0
    rot_x[:, 1, 0] = 0.0
    rot_x[:, 1, 1] = torch.cos(-roll)
    rot_x[:, 1, 2] = -torch.sin(-roll)
    rot_x[:, 2, 0] = 0.0
    rot_x[:, 2, 1] = torch.sin(-roll)
    rot_x[:, 2, 2] = torch.cos(-roll)

    rot_y = torch.zeros((num_envs, 3, 3), device=device)
    rot_y[:, 0, 0] = torch.cos(-pitch)
    rot_y[:, 0, 1] = 0.0
    rot_y[:, 0, 2] = torch.sin(-pitch)
    rot_y[:, 1, 0] = 0.0
    rot_y[:, 1, 1] = 1.0
    rot_y[:, 1, 2] = 0.0
    rot_y[:, 2, 0] = -torch.sin(-pitch)
    rot_y[:, 2, 1] = 0.0
    rot_y[:, 2, 2] = torch.cos(-pitch)

    rot_z = torch.zeros((num_envs, 3, 3), device=device)
    rot_z[:, 0, 0] = torch.cos(-yaw)
    rot_z[:, 0, 1] = -torch.sin(-yaw)
    rot_z[:, 0, 2] = 0.0
    rot_z[:, 1, 0] = torch.sin(-yaw)
    rot_z[:, 1, 1] = torch.cos(-yaw)
    rot_z[:, 1, 2] = 0.0
    rot_z[:, 2, 0] = 0.0
    rot_z[:, 2, 1] = 0.0
    rot_z[:, 2, 2] = 1.0

    vector = vector.unsqueeze(2)
    vector = torch.bmm(rot_x, vector)
    vector = torch.bmm(rot_y, vector)
    vector = torch.bmm(rot_z, vector)
    vector = vector.squeeze(2)
    return vector


def set_up_experiment_folder(folder, ckpt_idx):
    if not os.path.isdir(folder):
        os.makedirs(folder)

    ckpt_folder = os.path.join(folder, 'ckpt')
    if not os.path.isdir(ckpt_folder):
        os.makedirs(ckpt_folder)

    # assume checkpoint are named after "ckpt.{num_update}.pth"
    try:
        all_ckpt_indices = [int(file.split('.')[1]) for file in os.listdir(ckpt_folder)]
    except:
        assert False, 'checkpoint files should be named as "ckpt.{num_update}.pth"'

    # load a specified checkpoint
    if ckpt_idx != -1:
        assert ckpt_idx in all_ckpt_indices, 'ckpt idx requested {} does not exist'.format(ckpt_idx)
    # load the last checkpoint
    else:
        if len(all_ckpt_indices) > 0:
            ckpt_idx = max(all_ckpt_indices)

    # if an existing checkpoint is found
    if ckpt_idx != -1:
        start_epoch = ckpt_idx + 1
        ckpt_path = os.path.join(ckpt_folder, 'ckpt.{}.pth'.format(ckpt_idx))
    else:
        start_epoch = 0
        ckpt_path = None

    summary_folder = os.path.join(folder, 'summary')
    if not os.path.isdir(summary_folder):
        os.makedirs(summary_folder)

    log_file = os.path.join(folder, 'log')

    start_env_step = 0
    if os.path.isfile(log_file) and ckpt_idx != -1:
        with open(log_file) as f:
            lines = [line.strip() for line in f.readlines() if ('update: {}'.format(ckpt_idx)) in line]
            if len(lines) > 0:
                tokens = lines[0].split('\t')
                if len(tokens) > 1 and 'env_step' in tokens[1] and len(tokens[1].split()) > 1:
                    try:
                        start_env_step = int(tokens[1].split()[1])
                    except ValueError:
                        start_env_step = 0
        os.rename(log_file,
                  log_file + '_' + datetime.fromtimestamp(os.path.getctime(log_file)).strftime('%Y-%m-%d_%H:%M:%S'))

    return ckpt_folder, ckpt_path, start_epoch, start_env_step, summary_folder, log_file
