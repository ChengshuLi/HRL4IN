import numpy as np
import gym, gym.spaces
import time
import os
from collections import OrderedDict
import yaml

EPS = 1e-5


def parse_config(config):
    with open(config, 'r') as f:
        config_data = yaml.load(f)
    return config_data


class ToyEnv:
    def __init__(self,
                 config_file,
                 should_normalize_observation=True,
                 automatic_reset=True,
                 visualize=False):

        self.config = parse_config(config_file)
        self.load_map()

        # action_space
        # base: stop(no-op), turn left, turn right, forward
        # arm: stop(no-op), slide up, slide down
        self.action_space = gym.spaces.MultiDiscrete([4, 3])

        self.direction = {
            0: np.array([0, 1]),  # facing east
            1: np.array([-1, 0]),  # facing north
            2: np.array([0, -1]),  # facing west
            3: np.array([1, 0]),  # facing south
        }
        self.num_agent_orientation = len(self.direction)

        # door state
        self.door_pos = np.array([self.door_row, self.door_col])
        self.door_min_state = 1
        self.door_max_state = self.config.get('door_max_state', 1)
        assert self.door_max_state >= 1, 'door_max_state has to be greater than 0'

        # observation_space
        self.outputs = self.config.get('outputs', ['global_map'])
        self.local_map_range = self.config.get('local_map_range', 5)
        assert self.local_map_range % 2 == 1, 'local_map_range has to be an odd number'

        observation_space = OrderedDict()
        observation_space_min_max = OrderedDict()
        if 'sensor' in self.outputs:
            self.sensor_dim = 4
            # [agent_row, agent_col, agent_theta, door_state]
            self.sensor_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.sensor_dim,), dtype=np.float32)
            observation_space['sensor'] = self.sensor_space
            observation_space_min_max['sensor'] = np.array([
                [0.0, self.height - 1],
                [0.0, self.width - 1],
                [-np.pi - EPS, np.pi + EPS],
                [self.door_min_state, self.door_max_state],
            ])
        if 'auxiliary_sensor' in self.outputs:
            # [sin(agent_theta), cos(agent_theta), target_row, target_col, door_row, door_col,
            # is_left_room, is_door_front, is_door_front_and_facing_door,
            # target_row_local, target_col_local, door_row_local, door_col_local]
            self.auxiliary_sensor_dim = 9
            self.auxiliary_sensor_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.auxiliary_sensor_dim,),
                                                         dtype=np.float32)
            observation_space['auxiliary_sensor'] = self.auxiliary_sensor_space
            observation_space_min_max['auxiliary_sensor'] = np.array([
                [-1.0, 1.0],
                [-1.0, 1.0],
                [0.0, self.height - 1],
                [0.0, self.width - 1],
                [0.0, self.height - 1],
                [0.0, self.width - 1],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
                # [-(max(self.height, self.width) - 1), max(self.height, self.width) - 1],
                # [-(max(self.height, self.width) - 1), max(self.height, self.width) - 1],
                # [-(max(self.height, self.width) - 1), max(self.height, self.width) - 1],
                # [-(max(self.height, self.width) - 1), max(self.height, self.width) - 1],
            ])
        if 'global_map' in self.outputs:
            # 3 channels: channel 1: map
            #             channel 2: robot position and orientation (1, 2, 3, 4), and goal position (-1)
            #             channel 3: door state 1-5, 5 = open, 1 = closed
            self.global_map_space = gym.spaces.Box(low=-1.0, high=1.0,
                                                   shape=(3, self.height, self.width),
                                                   dtype=np.float32)
            observation_space['global_map'] = self.global_map_space
            observation_space_min_max['global_map'] = np.array([
                [0.0, 2.0],
                [-1.0, self.num_agent_orientation + 1],
                [0.0, self.door_max_state],
            ], dtype=np.float).reshape(3, 2, 1)
        if 'local_map' in self.outputs:
            # ego centric map, doesn't consider rotation
            # 4 channels: channel 1: map
            #             channel 2: robot position and orientation (1, 2, 3, 4), and goal position (-1)
            #             channel 3: door state 1-5, 5 = open, 1 = closed
            #             channel 4: validity map, 1 = this position is within the valid range of global map
            self.local_map_space = gym.spaces.Box(low=-1.0, high=1.0,
                                                  shape=(4, self.local_map_range, self.local_map_range),
                                                  dtype=np.float32)
            observation_space['local_map'] = self.local_map_space
            observation_space_min_max['local_map'] = np.array([
                [0.0, 2.0],
                [-1.0, self.num_agent_orientation + 1],
                [0.0, self.door_max_state],
                [0.0, 1.0],
            ], dtype=np.float).reshape(4, 2, 1)

        self.observation_space = gym.spaces.Dict(observation_space)
        self.observation_space_min_max = observation_space_min_max

        # traversable tiles for random initialization
        self.traversable_tiles_left = np.transpose(np.where(self.map[:, :self.door_col] == 0))
        self.traversable_tiles_right = np.transpose(np.where(self.map[:, (self.door_col + 1):] == 0))

        # termination and reward
        self.max_step = self.config.get('max_step', 500)
        self.sparse_reward = self.config.get('sparse_reward', False)

        # training-related and visualization
        self.should_normalize_observation = should_normalize_observation
        self.automatic_reset = automatic_reset
        self.visualize = visualize

        self.reset_env()

    def load_map(self):
        self.height = self.config.get('height', 7)
        self.width = self.config.get('width', 7)

        self.door_row = self.config.get('door_row', 3)
        self.door_col = self.config.get('door_col', 3)

        self.map = np.zeros((self.height, self.width))
        self.map[0, :] = 1
        self.map[-1, :] = 1
        self.map[:, 0] = 1
        self.map[:, -1] = 1  # add walls

        self.map[:, self.door_col] = 1  # add door wall
        self.map[self.door_row, self.door_col] = 2  # add door

    def reset(self):
        self.reset_env()
        return self.get_observation()

    def normalize_observation(self, observation):
        for key in observation:
            if len(observation[key].shape) == 1:
                assert np.all(observation[key] >= self.observation_space_min_max[key][:, 0]) and \
                       np.all(observation[key] <= self.observation_space_min_max[key][:, 1]), \
                       'observation out of bound: {}'.format(key)
                obs_mean = np.mean(self.observation_space_min_max[key], axis=1)
                obs_mag = (self.observation_space_min_max[key][:, 1] -
                           self.observation_space_min_max[key][:, 0]) / 2.0
            elif len(observation[key].shape) == 3:
                assert np.all(observation[key] >= self.observation_space_min_max[key][:, 0:1, :]) and \
                       np.all(observation[key] <= self.observation_space_min_max[key][:, 1:2, :]), \
                       'observation out of bound: {}'.format(observation)
                obs_mean = np.mean(self.observation_space_min_max[key], axis=1, keepdims=True)
                obs_mag = (self.observation_space_min_max[key][:, 1:2, :] -
                           self.observation_space_min_max[key][:, 0:1, :]) / 2.0
            else:
                assert False, 'unexpected observation shape'
            observation[key] = (observation[key] - obs_mean) / obs_mag
        return observation

    def wrap_to_pi(self, theta):
        return theta - np.pi * 2 * np.floor((theta + np.pi) / (np.pi * 2))

    def get_global_map(self):
        global_map = np.zeros((3, self.height, self.width), dtype=np.float32)
        global_map[0] = self.map
        global_map[1, self.agent_pos[0], self.agent_pos[1]] = self.agent_orientation + 1
        global_map[1, self.target_pos[0], self.target_pos[1]] = -1
        global_map[2, self.door_pos[0], self.door_pos[1]] = self.door_state
        return global_map

    def get_observation(self):
        observation = OrderedDict()
        if 'sensor' in self.outputs:
            sensor = np.zeros(self.sensor_dim, dtype=np.float32)
            theta = self.wrap_to_pi(self.agent_orientation * np.pi / 2)
            sensor[0:2] = self.agent_pos
            sensor[2] = theta
            sensor[3] = float(self.door_state)
            observation['sensor'] = sensor
        if 'auxiliary_sensor' in self.outputs:
            auxiliary_sensor = np.zeros(self.auxiliary_sensor_dim, dtype=np.float32)
            theta = self.wrap_to_pi(self.agent_orientation * np.pi / 2)
            auxiliary_sensor[0] = np.sin(theta)
            auxiliary_sensor[1] = np.cos(theta)
            auxiliary_sensor[2:4] = self.target_pos
            auxiliary_sensor[4:6] = self.door_pos
            auxiliary_sensor[6] = float(self.agent_pos[1] < self.door_pos[1])
            auxiliary_sensor[7] = float(self.agent_pos[0] == self.door_row
                                        and self.agent_pos[1] == (self.door_col - 1))
            auxiliary_sensor[8] = float(self.agent_pos[0] == self.door_row
                                        and self.agent_pos[1] == (self.door_col - 1)
                                        and self.agent_orientation == 0)
            # rotation_matrix = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])
            # auxiliary_sensor[9:11] = rotation_matrix.dot(self.target_pos - self.agent_pos)
            # auxiliary_sensor[11:13] = rotation_matrix.dot(self.door_pos - self.agent_pos)
            observation['auxiliary_sensor'] = auxiliary_sensor
        if 'global_map' in self.outputs:
            observation['global_map'] = self.get_global_map()
        if 'local_map' in self.outputs:
            global_map = self.get_global_map()
            local_map = np.zeros((4, self.local_map_range, self.local_map_range), dtype=np.float32)

            row_start = max(0, self.agent_pos[0] - self.local_map_range // 2)
            row_end = min(self.height, self.agent_pos[0] + self.local_map_range // 2 + 1)
            col_start = max(0, self.agent_pos[1] - self.local_map_range // 2)
            col_end = min(self.width, self.agent_pos[1] + self.local_map_range // 2 + 1)

            local_row_start = row_start - (self.agent_pos[0] - self.local_map_range // 2)
            local_row_end = local_row_start + (row_end - row_start)
            local_col_start = col_start - (self.agent_pos[1] - self.local_map_range // 2)
            local_col_end = local_col_start + (col_end - col_start)
            local_map[:3, local_row_start:local_row_end, local_col_start:local_col_end] = \
                global_map[:, row_start:row_end, col_start:col_end]
            local_map[3, local_row_start:local_row_end, local_col_start:local_col_end] = 1

            observation['local_map'] = local_map
        if self.should_normalize_observation:
            observation = self.normalize_observation(observation)

        return observation

    def get_l1_dist(self, a, b):
        return np.sum(np.abs(a - b))

    def get_potential(self):
        potential = 0.0
        # in the left room
        if self.agent_pos[1] < self.door_col:
            potential += self.get_l1_dist(self.agent_pos, self.door_pos)
            potential += self.door_max_state - self.door_state
            potential += self.get_l1_dist(self.door_pos, self.target_pos)

        # in the right room
        else:
            potential = self.get_l1_dist(self.agent_pos, self.target_pos)
        return potential

    def get_reward(self, action):
        reward = 0.0
        new_normalized_potential = self.get_potential() / self.initial_potential
        potential_reward = self.normalized_potential - new_normalized_potential
        self.normalized_potential = new_normalized_potential

        # new_potential = self.get_potential()
        # potential_reward = self.potential - new_potential
        # self.potential = new_potential

        if not self.sparse_reward:
            reward += potential_reward

        # slack reward
        # reward -= 0.01
        reward -= 0.001 * np.sum(action != 0)

        if np.array_equal(self.agent_pos, self.target_pos):
            success_reward = 10.0
            reward += success_reward
        return reward

    def get_done(self):
        done, info = False, {}
        if np.array_equal(self.agent_pos, self.target_pos):
            done = True
            info['success'] = True
        elif self.n_step >= self.max_step:
            done = True
            info['success'] = False
        if done:
            info['episode_length'] = self.n_step

        return done, info

    def reset_env(self):
        self.agent_pos = self.traversable_tiles_left[np.random.randint(0, len(self.traversable_tiles_left))]
        self.agent_orientation = np.random.randint(4)

        self.target_pos = self.traversable_tiles_right[np.random.randint(0, len(self.traversable_tiles_right))].copy()
        self.target_pos[1] += (self.door_col + 1)

        self.door_state = self.door_min_state
        self.n_step = 0
        self.normalized_potential = 1.0
        self.initial_potential = self.get_potential()
        # self.potential = self.get_potential()

    def simulation_step(self, action):
        # locomotion
        if action[0] == 1:  # turn left
            self.agent_orientation = (self.agent_orientation + 1) % self.num_agent_orientation
        elif action[0] == 2:  # turn right
            self.agent_orientation = (self.agent_orientation - 1) % self.num_agent_orientation
        elif action[0] == 3:  # move forward
            next_pos = self.agent_pos + self.direction[self.agent_orientation]
            empty_space = self.map[next_pos[0], next_pos[1]] == 0
            open_door = self.map[next_pos[0], next_pos[1]] == 2 and self.door_state == self.door_max_state
            if empty_space or open_door:
                self.agent_pos = next_pos

        # manipulation
        if action[1] == 1 or action[1] == 2:
            next_pos = self.agent_pos + self.direction[self.agent_orientation]
            if self.map[next_pos[0], next_pos[1]] == 2:  # door
                if action[1] == 1:
                    self.door_state += 1
                else:
                    self.door_state -= 1
            self.door_state = np.clip(self.door_state, self.door_min_state, self.door_max_state)

    def step(self, action):
        self.n_step += 1
        if self.visualize:
            self.print_observation()
        self.simulation_step(action)
        if self.visualize:
            self.print_action(action)
            self.print_observation()
            time.sleep(1)
            os.system('clear')

        observation = self.get_observation()

        reward = self.get_reward(action)
        done, info = self.get_done()

        if done and self.automatic_reset:
            info['last_observation'] = observation
            observation = self.reset()
        return observation, reward, done, info

    def print_action(self, action):
        action_mapping = [
            {
                0: 'no op',
                1: 'turn left',
                2: 'turn right',
                3: 'move forward',
            },
            {
                0: 'no op',
                1: 'slide up',
                2: 'slide down'
            }
        ]
        for v, m in zip(action, action_mapping):
            print(m[v])

    def print_observation(self):
        vis_map = np.zeros((self.height, self.width), dtype=np.int)
        vis_map[self.agent_pos[0], self.agent_pos[1]] = self.agent_orientation + 1
        vis_map[self.target_pos[0], self.target_pos[1]] = -1
        vis_map_str = str(vis_map)
        vis_map_str = (vis_map_str.replace("-1", " x").replace("1", ">").
                       replace("2", "^").replace("3", "<").replace("4", "v"))
        print(vis_map_str)


if __name__ == "__main__":
    env = ToyEnv('map.yaml', visualize=False)
    potentials = []
    for i in range(1000):
        print("episode: {}".format(i))
        observation = env.reset()
        potentials.append(env.get_potential())
        # env.agent_pos = np.array([1, 2])
        # env.agent_orientation = 3
        # while True:
        #     action = env.action_space.sample()
        #     action = [3, 1]
        #     observation, _, done, _ = env.step(action)
        #     if done:
        #         break
    print(np.mean(potentials))