import math
from abc import ABC, abstractmethod
import torch
import numpy as np
import torch_utils


class MuZeroNetwork:
    def __new__(cls, config):
        if config.network == "fullyconnected":
            return MuZeroFullyConnectedNetwork(
                config.observation_shape,
                config.stacked_observations,
                len(config.action_space),
                config.encoding_size,
                config.fc_reward_layers,
                config.fc_value_layers,
                config.fc_policy_layers,
                config.fc_representation_layers,
                config.fc_dynamics_layers,
                config.support_size,
            )
        elif config.network == "resnet":
            return MuZeroResidualNetwork(
                config.observation_shape,
                config.stacked_observations,
                len(config.action_space),
                config.blocks,
                config.channels,
                config.reduced_channels_reward,
                config.reduced_channels_value,
                config.reduced_channels_policy,
                config.resnet_fc_reward_layers,
                config.resnet_fc_value_layers,
                config.resnet_fc_policy_layers,
                config.support_size,
                config.downsample,
            )
        else:
            raise NotImplementedError(
                'The network parameter should be "fullyconnected" or "resnet".'
            )


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict


class AbstractNetwork(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def initial_inference(self, observation):
        pass

    @abstractmethod
    def recurrent_inference(self, encoded_state, action):
        pass

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)


##################################
######## Fully Connected #########


class MuZeroFullyConnectedNetwork(AbstractNetwork):
    def __init__(
            self,
            observation_shape,
            stacked_observations,
            action_space_size,
            encoding_size,
            fc_reward_layers,
            fc_value_layers,
            fc_policy_layers,
            fc_representation_layers,
            fc_dynamics_layers,
            support_size,
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1

        self.representation_network = torch.nn.DataParallel(
            mlp(
                observation_shape[0]
                * observation_shape[1]
                * observation_shape[2]
                * (stacked_observations + 1)
                + stacked_observations * observation_shape[1] * observation_shape[2],
                fc_representation_layers,
                encoding_size,
            )
        )

        self.dynamics_encoded_state_network = torch.nn.DataParallel(
            mlp(
                encoding_size + self.action_space_size,
                fc_dynamics_layers,
                encoding_size,
            )
        )
        self.dynamics_reward_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_reward_layers, self.full_support_size)
        )

        self.prediction_policy_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_policy_layers, self.action_space_size)
        )
        self.prediction_value_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_value_layers, self.full_support_size)
        )

    def prediction(self, encoded_state):
        policy_logits = self.prediction_policy_network(encoded_state)
        value = self.prediction_value_network(encoded_state)
        return policy_logits, value

    def representation(self, observation):
        encoded_state = self.representation_network(
            observation.view(observation.shape[0], -1)
        )
        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_encoded_state = encoded_state.min(1, keepdim=True)[0]
        max_encoded_state = encoded_state.max(1, keepdim=True)[0]
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (
                                           encoded_state - min_encoded_state
                                   ) / scale_encoded_state
        return encoded_state_normalized

    def dynamics(self, encoded_state, action):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = (
            torch.zeros((action.shape[0], self.action_space_size))
                .to(action.device)
                .float()
        )
        action_one_hot.scatter_(1, action.long(), 1.0)
        x = torch.cat((encoded_state, action_one_hot), dim=1)

        next_encoded_state = self.dynamics_encoded_state_network(x)

        reward = self.dynamics_reward_network(next_encoded_state)

        # Scale encoded state between [0, 1] (See paper appendix Training)
        min_next_encoded_state = next_encoded_state.min(1, keepdim=True)[0]
        max_next_encoded_state = next_encoded_state.max(1, keepdim=True)[0]
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (
                                                next_encoded_state - min_next_encoded_state
                                        ) / scale_next_encoded_state

        return next_encoded_state_normalized, reward

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
        # reward equal to 0 for consistency
        reward = torch.log(
            (
                torch.zeros(1, self.full_support_size)
                    .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
                    .repeat(len(observation), 1)
                    .to(observation.device)
            )
        )

        return (
            value,
            reward,
            policy_logits,
            encoded_state,
        )

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state


###### End Fully Connected #######
##################################


##################################
############# ResNet #############


def conv3x3(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


# Residual block
class ResidualBlock(torch.nn.Module):
    def __init__(self, num_channels, stride=1):
        super().__init__()
        self.conv1 = conv3x3(num_channels, num_channels, stride)
        self.bn1 = torch.nn.BatchNorm2d(num_channels)
        self.conv2 = conv3x3(num_channels, num_channels)
        self.bn2 = torch.nn.BatchNorm2d(num_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.nn.functional.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = torch.nn.functional.relu(out)
        return out


# Downsample observations before representation network (See paper appendix Network Architecture)
class DownSample(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.resblocks1 = torch.nn.ModuleList(
            [ResidualBlock(out_channels // 2) for _ in range(2)]
        )
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.resblocks2 = torch.nn.ModuleList(
            [ResidualBlock(out_channels) for _ in range(3)]
        )
        self.pooling1 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks3 = torch.nn.ModuleList(
            [ResidualBlock(out_channels) for _ in range(3)]
        )
        self.pooling2 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        for block in self.resblocks1:
            x = block(x)
        x = self.conv2(x)
        for block in self.resblocks2:
            x = block(x)
        x = self.pooling1(x)
        for block in self.resblocks3:
            x = block(x)
        x = self.pooling2(x)
        return x


class DownsampleCNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, h_w):
        super().__init__()
        mid_channels = (in_channels + out_channels) // 2
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, mid_channels, kernel_size=h_w[0] * 2, stride=4, padding=2
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d(h_w)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x


class RepresentationNetwork(torch.nn.Module):
    """

    Example: (3, 96, 96) * 4 = (12, 96, 96) -> (128, 6, 6)

    """

    def __init__(
            self,
            observation_shape,
            stacked_observations,
            num_blocks,
            num_channels,
            downsample,
    ):
        super().__init__()
        print(observation_shape[0], stacked_observations, observation_shape[0] * stacked_observations
              + stacked_observations)
        self.downsample = downsample
        if self.downsample:
            if self.downsample == "resnet":
                self.downsample_net = DownSample(
                    observation_shape[0] * stacked_observations,
                    num_channels,
                )
            elif self.downsample == "CNN":
                self.downsample_net = DownsampleCNN(
                    observation_shape[0] * stacked_observations,
                    num_channels,
                    (
                        math.ceil(observation_shape[1] / 16),
                        math.ceil(observation_shape[2] / 16),
                    ),
                )
            else:
                raise NotImplementedError('downsample should be "resnet" or "CNN".')
        self.conv = conv3x3(
            observation_shape[0] * (stacked_observations + 1) + stacked_observations,
            num_channels,
        )
        self.bn = torch.nn.BatchNorm2d(num_channels)
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

    def forward(self, x):
        if self.downsample:
            x = self.downsample_net(x)
        else:
            x = self.conv(x)
            x = self.bn(x)
            x = torch.nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)
        return x


class DynamicsNetwork(torch.nn.Module):
    def __init__(
            self,
            num_blocks,
            num_channels,
            num_output_channels,
            reduced_channels_reward,
            fc_reward_layers,
            full_support_size,
            block_output_size_reward,
    ):
        super().__init__()

        self.conv = conv3x3(num_channels, num_output_channels)
        self.bn = torch.nn.BatchNorm2d(num_output_channels)
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_output_channels) for _ in range(num_blocks)]
        )

        self.conv1x1_reward = torch.nn.Conv2d(
            num_output_channels, reduced_channels_reward, 1
        )
        self.block_output_size_reward = block_output_size_reward
        self.fc = mlp(
            self.block_output_size_reward, fc_reward_layers, full_support_size,
        )

    def forward(self, x):
        # print('FWD', x.shape)
        x = self.conv(x)
        x = self.bn(x)
        x = torch.nn.functional.relu(x)
        for block in self.resblocks:
            x = block(x)
        state = x
        x = self.conv1x1_reward(x)
        x = x.view(-1, self.block_output_size_reward)
        reward = self.fc(x)
        return state, reward


class PredictionNetwork(torch.nn.Module):
    def __init__(
            self,
            action_space_size,
            num_blocks,
            num_channels,
            reduced_channels_value,
            reduced_channels_policy,
            fc_value_layers,
            fc_policy_layers,
            full_support_size,
            block_output_size_value,
            block_output_size_policy,
    ):
        super().__init__()
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

        self.conv1x1_value = torch.nn.Conv2d(num_channels, reduced_channels_value, 1)
        self.conv1x1_policy = torch.nn.Conv2d(num_channels, reduced_channels_policy, 1)
        self.block_output_size_value = block_output_size_value
        self.block_output_size_policy = block_output_size_policy
        self.fc_value = mlp(
            self.block_output_size_value, fc_value_layers, full_support_size
        )
        self.fc_policy = mlp(
            self.block_output_size_policy, fc_policy_layers, action_space_size * 2,
        )

        # Maybe we can try to output this....
        # self.log_stds = nn.Parameter(torch.zeros(1, action_space_size))

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
        value = self.conv1x1_value(x)
        policy = self.conv1x1_policy(x)
        value = value.view(-1, self.block_output_size_value)
        policy = policy.view(-1, self.block_output_size_policy)
        value = self.fc_value(value)
        policy = self.fc_policy(policy)
        return policy, value


class PredictionNetworkGaussian(torch.nn.Module):
    """
        Gaussian Policy for the continuous space.

    """

    @staticmethod
    def calculate_log_pi(log_stds, noises, actions):
        gaussian_log_probs = (-0.5 * noises.pow(2) - log_stds).sum(
            dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)

        return gaussian_log_probs - torch.log(
            1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

    @staticmethod
    def reparameterize(means, log_stds):
        noises = torch.randn_like(means)
        us = means + noises * log_stds.exp()
        actions = torch.tanh(us)
        return actions, PredictionNetworkGaussian.calculate_log_pi(log_stds, noises, actions)

    @staticmethod
    def reparameterize_n(means, log_stds, n):
        means_n = means.reshape(means.shape[0], 1, -1).repeat(1, n, 1)
        log_stds_n = log_stds.reshape(log_stds.shape[0], 1, -1).repeat(1, n, 1)
        noises = torch.randn_like(means_n)
        us = means_n + noises * log_stds_n.exp()
        actions = torch.tanh(us)
        return actions

    @staticmethod
    def reparameterize_clip(means, log_stds):
        noises = torch.randn_like(means)
        us = means + noises * log_stds.exp()
        actions = torch.clip(us, -1, 1)
        return actions, PredictionNetworkGaussian.calculate_log_pi(log_stds, noises, actions)

    @staticmethod
    def atanh(x):
        return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))

    @staticmethod
    def evaluate_log_pi(means, log_stds, actions):
        # print("LOG_STDS", log_stds)
        noises = (PredictionNetworkGaussian.atanh(actions) - means) / (log_stds.exp() + 1e-8)
        return PredictionNetworkGaussian.calculate_log_pi(log_stds, noises, actions)

    def __init__(
            self,
            action_space_size,
            num_blocks,
            num_channels,
            reduced_channels_value,
            reduced_channels_policy,
            fc_value_layers,
            fc_policy_layers,
            full_support_size,
            block_output_size_value,
            block_output_size_policy,
    ):
        super().__init__()
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

        self.conv1x1_value = torch.nn.Conv2d(num_channels, reduced_channels_value, 1)
        self.conv1x1_policy = torch.nn.Conv2d(num_channels, reduced_channels_policy, 1)
        self.block_output_size_value = block_output_size_value
        self.block_output_size_policy = block_output_size_policy
        self.fc_value = mlp(
            self.block_output_size_value, fc_value_layers, full_support_size
        )
        self.fc_policy = mlp(
            self.block_output_size_policy, fc_policy_layers, action_space_size * 2,
        )

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
        value = self.conv1x1_value(x)
        policy = self.conv1x1_policy(x)
        value = value.view(-1, self.block_output_size_value)
        policy = policy.view(-1, self.block_output_size_policy)
        value = self.fc_value(value)
        policy = self.fc_policy(policy)  # [mu, log_stds.]

        return policy, value


class MuZeroResidualNetwork(AbstractNetwork):
    def __init__(
            self,
            observation_shape,
            stacked_observations,
            action_space_size,
            num_blocks,
            num_channels,
            reduced_channels_reward,
            reduced_channels_value,
            reduced_channels_policy,
            fc_reward_layers,
            fc_value_layers,
            fc_policy_layers,
            fc_action_encoder_layers,
            fc_action_encoder_dim,
            support_size,
            downsample,
            bn_mt=0.1,
            proj_hid=256,
            proj_out=256,
            pred_hid=64,
            pred_out=256,
            init_zero=False,
            state_norm=False
    ):
        super().__init__()
        self.proj_hid = proj_hid
        self.proj_out = proj_out
        self.pred_hid = pred_hid
        self.pred_out = pred_out
        self.init_zero = init_zero
        self.state_norm = state_norm

        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1
        block_output_size_reward = (
            (
                    reduced_channels_reward
                    * math.ceil(observation_shape[1] / 16)
                    * math.ceil(observation_shape[2] / 16)
            )
            if downsample
            else (reduced_channels_reward * observation_shape[1] * observation_shape[2])
        )

        block_output_size_value = (
            (
                    reduced_channels_value
                    * math.ceil(observation_shape[1] / 16)
                    * math.ceil(observation_shape[2] / 16)
            )
            if downsample
            else (reduced_channels_value * observation_shape[1] * observation_shape[2])
        )

        block_output_size_policy = (
            (
                    reduced_channels_policy
                    * math.ceil(observation_shape[1] / 16)
                    * math.ceil(observation_shape[2] / 16)
            )
            if downsample
            else (reduced_channels_policy * observation_shape[1] * observation_shape[2])
        )

        self.representation_network =  RepresentationNetwork(
                observation_shape,
                stacked_observations,
                num_blocks,
                num_channels,
                downsample,
            )


        self.dynamics_network = DynamicsNetwork(
                num_blocks,
                num_channels + fc_action_encoder_dim,
                num_channels,
                reduced_channels_reward,
                fc_reward_layers,
                self.full_support_size,
                block_output_size_reward,
            )


        self.prediction_network =  PredictionNetwork(
                action_space_size,
                num_blocks,
                num_channels,
                reduced_channels_value,
                reduced_channels_policy,
                fc_value_layers,
                fc_policy_layers,
                self.full_support_size,
                block_output_size_value,
                block_output_size_policy,
            )


        self.action_encoder = mlp(action_space_size, fc_action_encoder_layers, fc_action_encoder_dim)


        in_dim = num_channels * math.ceil(observation_shape[1] / 16) * math.ceil(observation_shape[2] / 16)
        self.projection_in_dim = in_dim
        self.projection = nn.Sequential(
            nn.Linear(self.projection_in_dim, self.proj_hid),
            nn.BatchNorm1d(self.proj_hid),
            nn.ReLU(),
            nn.Linear(self.proj_hid, self.proj_hid),
            nn.BatchNorm1d(self.proj_hid),
            nn.ReLU(),
            nn.Linear(self.proj_hid, self.proj_out),
            nn.BatchNorm1d(self.proj_out)
        )

        self.projection_head = nn.Sequential(
            nn.Linear(self.proj_out, self.pred_hid),
            nn.BatchNorm1d(self.pred_hid),
            nn.ReLU(),
            nn.Linear(self.pred_hid, self.pred_out),
        )

    def sample_mixed_actions(self, policy, config):
        n_batchsize = policy.shape[0]
        n_policy_action = config.mcts_num_policy_samples
        n_random_action = config.mcts_num_random_samples

        # RETURNING, [BATCHSIZE, N, ACTION_DIM]
        random_actions = config.sample_random_actions_fast(n_random_action * n_batchsize)
        random_actions = random_actions.reshape(n_batchsize, -1, config.action_space_size)

        action_dim = policy.shape[-1] // 2
        policy = policy.reshape(-1, policy.shape[-1])

        policy_action = PredictionNetworkGaussian.reparameterize_n(
            means=policy[:, :action_dim], log_stds=policy[:, action_dim:], n=n_policy_action
        )

        policy_action = torch_utils.tensor_to_numpy(policy_action)
        return np.concatenate([policy_action, random_actions], axis=1)

    def sample_actions(self, policy, n):
        action_dim = policy.shape[-1] // 2
        policy = policy.reshape(-1, policy.shape[-1])
        policy = torch.cat([policy for _ in range(n)], dim=0)

        samples = PredictionNetworkGaussian.reparameterize(means=policy[:, :action_dim],
                                                           log_stds=policy[:, action_dim:])

        samples = torch_utils.tensor_to_numpy(samples[0])
        return samples

    def get_policy_entropy(self, policy):
        action_dim = policy.shape[-1] // 2
        log_stds = policy[:, action_dim:]
        return log_stds.mean()

    def evaluate_log_pi(self, hidden_state, target_all_actions):
        """

        :param hidden_state:
        :param target_all_actions:
        :return:
        """
        policy, _ = self.prediction(hidden_state)

        branch_dim = target_all_actions.shape[1]
        action_dim = policy.shape[-1] // 2

        means = policy[:, :action_dim]          # [Batchsize, Action_dim]
        log_stds = policy[:, action_dim:]       # [Batchsize, Action_dim]
        # print("MEAN", means.shape, "LOG_STDS", log_stds.shape)

        means = means.unsqueeze(1).repeat(1, branch_dim, 1)       #[]
        log_stds = log_stds.unsqueeze(1).repeat(1, branch_dim, 1)

        return PredictionNetworkGaussian.evaluate_log_pi(means, log_stds, target_all_actions)

    def prediction(self, encoded_state):
        policy, value = self.prediction_network(encoded_state)
        return policy, value

    def representation(self, observation):
        assert observation.max() < 10, "We assume that the observation is already pre-processed."

        encoded_state = self.representation_network(observation)

        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_encoded_state = (
            encoded_state.view(
                -1,
                encoded_state.shape[1],
                encoded_state.shape[2] * encoded_state.shape[3],
            )
                .min(2, keepdim=True)[0]
                .unsqueeze(-1)
        )
        max_encoded_state = (
            encoded_state.view(
                -1,
                encoded_state.shape[1],
                encoded_state.shape[2] * encoded_state.shape[3],
            )
                .max(2, keepdim=True)[0]
                .unsqueeze(-1)
        )
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (
                                           encoded_state - min_encoded_state
                                   ) / scale_encoded_state
        return encoded_state_normalized

    def dynamics(self, encoded_state, action):
        """

               :param encoded_state: [Batchsize, Encoded_channel_dim, Encoded_w, Encoded_h]
               :param action: shape: [Batchsize, Action_Dim]
               :return:
        """

        encoded_action = self.action_encoder.forward(action)
        expanded_action = encoded_action.reshape(*encoded_action.shape, 1, 1) \
            .repeat(1, 1, encoded_state.shape[2], encoded_state.shape[3])

        x = torch.cat((encoded_state, expanded_action), dim=1)

        next_encoded_state, reward = self.dynamics_network(x)

        # Scale encoded state between [0, 1] (See paper appendix Training)
        min_next_encoded_state = (
            next_encoded_state.view(
                -1,
                next_encoded_state.shape[1],
                next_encoded_state.shape[2] * next_encoded_state.shape[3],
            )
                .min(2, keepdim=True)[0]
                .unsqueeze(-1)
        )
        max_next_encoded_state = (
            next_encoded_state.view(
                -1,
                next_encoded_state.shape[1],
                next_encoded_state.shape[2] * next_encoded_state.shape[3],
            )
                .max(2, keepdim=True)[0]
                .unsqueeze(-1)
        )
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (
                                                next_encoded_state - min_next_encoded_state
                                        ) / scale_next_encoded_state
        return next_encoded_state_normalized, reward

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
        # reward equal to 0 for consistency
        reward =  (torch.zeros(1, self.full_support_size)
                    .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
                    .repeat(len(observation), 1)
                    .to(observation.device))
        return (
            value,
            reward,
            policy_logits,
            encoded_state,
        )

    def recurrent_inference(self, encoded_state, action):
        """

        :param encoded_state: [Batchsize, Encoded_channel_dim, Encoded_w, Encoded_h]
        :param action: shape: [Batchsize, Action_Dim]
        :return:
        """

        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state

    def contrastive_loss(self, x1, x2):
        out1 = x1.view(-1, self.projection_in_dim)

        z1 = self.projection(out1)
        p1 = self.projection_head(z1)

        out2 = x2.view(-1, self.projection_in_dim)
        z2 = self.projection(out2)
        p2 = self.projection_head(z2)

        d1 = self.d(p1, z2) / 2.
        d2 = self.d(p2, z1) / 2.

        return d1 + d2

    def project(self, hidden_state, with_grad=True):
        hidden_state = hidden_state.view(-1, self.projection_in_dim)
        proj = self.projection(hidden_state)

        # with grad, use proj_head
        if with_grad:
            proj = self.projection_head(proj)
            return proj
        else:
            return proj.detach()


########### End ResNet ###########
##################################
import torch.nn as nn


def mlp(
        input_size,
        layer_sizes,
        output_size,
        output_activation=torch.nn.Identity,
        momentum=0.1,
        activation=torch.nn.ELU,
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        if i < len(sizes) - 2:
            act = activation
            layers += [nn.Linear(sizes[i], sizes[i + 1]),
                       nn.BatchNorm1d(sizes[i + 1], momentum=momentum),
                       act()]
        else:
            act = output_activation
            layers += [nn.Linear(sizes[i], sizes[i + 1]),
                       act()]

    return torch.nn.Sequential(*layers)


if __name__ == '__main__':
    import numpy as np

    network = MuZeroResidualNetwork(observation_shape=(3, 96, 96),
                                    stacked_observations=4,
                                    action_space_size=4,
                                    num_blocks=3,
                                    num_channels=128,  # Hidden state = (128, 96, 96)
                                    reduced_channels_reward=256,
                                    reduced_channels_value=256,
                                    reduced_channels_policy=256,
                                    fc_reward_layers=[256, 256],
                                    fc_value_layers=[256, 256],
                                    fc_policy_layers=[256, 256],
                                    fc_action_encoder_layers=[128, ],
                                    fc_action_encoder_dim=8,
                                    support_size=300,
                                    downsample="resnet")

    # By default, the WH size will reduce by 16

    x = torch.randn((5, 16, 96, 96))
    x = network.representation(x)
    print(x.shape)
    policy_logits, value = network.prediction(x)
    xn = network.recurrent_inference(x, torch.randn(5, 4))
    # print(policy_logits.shape, value.shape)
    print(xn[3].shape)