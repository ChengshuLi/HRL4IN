import torch
import torch.nn as nn
import numpy as np


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(0))

    def forward(self, x):
        return x + self._bias


class Net(nn.Module):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self,
                 observation_space,
                 hidden_size,
                 single_branch_size=256,
                 cnn_layers_params=None,
                 ):
        super().__init__()

        if "sensor" in observation_space.spaces:
            self._n_non_vis_sensor = observation_space.spaces["sensor"].shape[0]
        else:
            self._n_non_vis_sensor = 0

        if "auxiliary_sensor" in observation_space.spaces:
            self._n_auxiliary_sensor = observation_space.spaces["auxiliary_sensor"].shape[0]
        else:
            self._n_auxiliary_sensor = 0

        if "scan" in observation_space.spaces:
            self._n_scan = observation_space.spaces["scan"].shape[0]
        else:
            self._n_scan = 0

        if "subgoal" in observation_space.spaces:
            self._n_subgoal = observation_space.spaces["subgoal"].shape[0]
        else:
            self._n_subgoal = 0

        if "subgoal_mask" in observation_space.spaces:
            self._n_subgoal_mask = observation_space.spaces["subgoal_mask"].shape[0]
        else:
            self._n_subgoal_mask = 0

        if "action_mask" in observation_space.spaces:
            self._n_action_mask = observation_space.spaces["action_mask"].shape[0]
        else:
            self._n_action_mask = 0

        self._n_additional_rnn_input = (
                self._n_non_vis_sensor +
                self._n_auxiliary_sensor +
                self._n_subgoal +
                self._n_subgoal_mask +
                self._n_action_mask +
                self._n_scan
        )
        self._hidden_size = hidden_size
        self._single_branch_size = single_branch_size

        if self._n_additional_rnn_input != 0:
            self.feature_linear = nn.Sequential(
                nn.Linear(self._n_additional_rnn_input, self._single_branch_size),
                nn.ReLU()
            )

        if cnn_layers_params is None:
            self._cnn_layers_params = [(32, 8, 4, 0), (64, 4, 2, 0), (64, 3, 1, 0)]
        else:
            self._cnn_layers_params = cnn_layers_params
        self.cnn = self._init_perception_model(observation_space)

        self._rnn_input_size = 0
        if not self.is_blind:
            self._rnn_input_size += single_branch_size
        if self._n_additional_rnn_input != 0:
            self._rnn_input_size += single_branch_size

        assert self._rnn_input_size != 0, "the network has no input"

        self.rnn = nn.GRU(self._rnn_input_size, self._hidden_size)
        self.critic_linear = nn.Linear(self._hidden_size, 1)

        self.layer_init()
        self.train()

    def _init_perception_model(self, observation_space):
        if "rgb" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
        else:
            self._n_input_rgb = 0

        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
        else:
            self._n_input_depth = 0

        if "global_map" in observation_space.spaces:
            self._n_input_global_map = observation_space.spaces["global_map"].shape[0]
        else:
            self._n_input_global_map = 0

        if "local_map" in observation_space.spaces:
            self._n_input_local_map = observation_space.spaces["local_map"].shape[0]
        else:
            self._n_input_local_map = 0

        if self._n_input_rgb > 0:
            cnn_dims = np.array(observation_space.spaces["rgb"].shape[:2], dtype=np.float32)
        elif self._n_input_depth > 0:
            cnn_dims = np.array(observation_space.spaces["depth"].shape[:2], dtype=np.float32)
        elif self._n_input_global_map > 0:
            cnn_dims = np.array(observation_space.spaces["global_map"].shape[1:3], dtype=np.float32)
        elif self._n_input_local_map > 0:
            cnn_dims = np.array(observation_space.spaces["local_map"].shape[1:3], dtype=np.float32)

        if self.is_blind:
            return nn.Sequential()
        else:
            for _, kernel_size, stride, padding in self._cnn_layers_params:
                cnn_dims = self._conv_output_dim(
                    dimension=cnn_dims,
                    padding=np.array([padding, padding], dtype=np.float32),
                    dilation=np.array([1, 1], dtype=np.float32),
                    kernel_size=np.array([kernel_size, kernel_size], dtype=np.float32),
                    stride=np.array([stride, stride], dtype=np.float32),
                )

            cnn_layers = []
            prev_out_channels = None
            for i, (out_channels, kernel_size, stride, padding) in enumerate(self._cnn_layers_params):
                if i == 0:
                    in_channels = self._n_input_rgb + self._n_input_depth + \
                                  self._n_input_global_map + self._n_input_local_map
                else:
                    in_channels = prev_out_channels
                cnn_layers.append(nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ))
                if i != len(self._cnn_layers_params) - 1:
                    cnn_layers.append(nn.ReLU())
                prev_out_channels = out_channels

            cnn_layers += [
                Flatten(),
                nn.Linear(self._cnn_layers_params[-1][0] * cnn_dims[0] * cnn_dims[1],
                          self._single_branch_size),
                nn.ReLU(),
            ]
            return nn.Sequential(*cnn_layers)

    def _conv_output_dim(self, dimension, padding, dilation, kernel_size, stride):
        """Calculates the output height and width based on the input
        height and width to the convolution layer.

        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(np.floor(
                    ((dimension[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i]) + 1
                ))
            )
        return tuple(out_dimension)

    @property
    def output_size(self):
        return self._hidden_size

    def layer_init(self):
        for layer in self.cnn:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                # nn.init.orthogonal_(layer.weight, nn.init.calculate_gain("relu"))
                nn.init.orthogonal_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, val=0)

        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

        nn.init.orthogonal_(self.critic_linear.weight, gain=1)
        nn.init.constant_(self.critic_linear.bias, val=0)

    def forward_rnn(self, x, hidden_states, masks):
        if x.size(0) == hidden_states.size(0):
            assert hidden_states.size(0) == masks.size(0)
            x, hidden_states = self.rnn(
                x.unsqueeze(0), (hidden_states * masks).unsqueeze(0)
            )
            x = x.squeeze(0)
            hidden_states = hidden_states.squeeze(0)
        else:
            # x is a (T, N, -1) tensor flattened to (T * N, -1)
            n = hidden_states.size(0)
            t = int(x.size(0) / n)

            # unflatten
            x = x.view(t, n, x.size(1))
            masks = masks.view(t, n)

            # steps in sequence which have zero for any agent. Assume t=0 has
            # a zero in it.
            has_zeros = (
                (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()
            )

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                has_zeros = [has_zeros.item() + 1]  # handle scalar
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [t]

            hidden_states = hidden_states.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # process steps that don't have any zeros in masks together
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hidden_states = self.rnn(
                    x[start_idx:end_idx],
                    hidden_states * masks[start_idx].view(1, -1, 1),
                )

                outputs.append(rnn_scores)

            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            x = x.view(t * n, -1)  # flatten
            hidden_states = hidden_states.squeeze(0)

        return x, hidden_states

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth + self._n_input_global_map + self._n_input_local_map == 0

    def forward_perception_model(self, observations):
        cnn_input = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            cnn_input.append(rgb_observations)

        if self._n_input_depth > 0:
            depth_observations = observations["depth"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            cnn_input.append(depth_observations)

        if self._n_input_global_map > 0:
            global_map_observations = observations["global_map"]
            # global_map is already in dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            cnn_input.append(global_map_observations)

        if self._n_input_local_map > 0:
            local_map_observations = observations["local_map"]
            cnn_input.append(local_map_observations)

        cnn_input = torch.cat(cnn_input, dim=1)
        return self.cnn(cnn_input)

    def forward(self, observations, rnn_hidden_states, masks):
        if self._n_additional_rnn_input > 0:
            additional_rnn_input = []
            if self._n_non_vis_sensor > 0:
                additional_rnn_input.append(observations["sensor"])
            if self._n_auxiliary_sensor > 0:
                additional_rnn_input.append(observations["auxiliary_sensor"])
            if self._n_scan > 0:
                additional_rnn_input.append(observations["scan"])
            if self._n_subgoal > 0:
                additional_rnn_input.append(observations["subgoal"])
            if self._n_subgoal_mask > 0:
                additional_rnn_input.append(observations["subgoal_mask"])
            if self._n_action_mask > 0:
                additional_rnn_input.append(observations["action_mask"])
            x = torch.cat(additional_rnn_input, dim=1)
            x = self.feature_linear(x)

        if not self.is_blind:
            perception_embed = self.forward_perception_model(observations)
            if self._n_additional_rnn_input > 0:
                x = torch.cat([perception_embed, x], dim=1)
            else:
                x = perception_embed

        x, rnn_hidden_states = self.forward_rnn(x, rnn_hidden_states, masks)

        return self.critic_linear(x), x, rnn_hidden_states
