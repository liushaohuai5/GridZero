import numpy as np

from model.network import MuZeroResidualNetwork
import ray


@ray.remote(num_gpus=0.3)
class RZeroModelWorker:
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
    ):
        super().__init__()
        self.network =  MuZeroResidualNetwork(observation_shape,
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
            downsample).cuda()

        self.network.eval()

    def get_weights(self):
        return self.network.get_weights()

    def set_weights(self, weights):
        self.network.set_weights(weights)

    def sample_actions(self, policy, n):
        return self.network.sample_actions(policy, n)

    def prediction(self, encoded_state):
        return self.network.prediction(encoded_state)

    def representation(self, observation):
        assert observation.max() < 10, "We assume that the observation is already pre-processed."
        return self.network.representation(observation)

    def dynamics(self, encoded_state, action):
        """

               :param encoded_state: [Batchsize, Encoded_channel_dim, Encoded_w, Encoded_h]
               :param action: shape: [Batchsize, Action_Dim]
               :return:
        """
        return self.network.dynamics(encoded_state, action)

    def sample_mixed_actions(self, policy, config):
        return self.network.sample_mixed_actions(policy, config)

    def initial_inference(self, observation):
        return self.network.initial_inference(observation)

    def recurrent_inference(self, encoded_state, action):
        """

        :param encoded_state: [Batchsize, Encoded_channel_dim, Encoded_w, Encoded_h]
        :param action: shape: [Batchsize, Action_Dim]
        :return:
        """
        return self.network.recurrent_inference(encoded_state, action)

    def contrastive_loss(self, x1, x2):
        return self.network.contrastive_loss(x1, x2)

    def project(self, hidden_state, with_grad=True):
        return self.network.project(hidden_state, with_grad)

import torch_utils

@ray.remote(num_gpus=0.4)
class NumpyRZeroModelWorker:
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
    ):
        super().__init__()
        self.network =  MuZeroResidualNetwork(observation_shape,
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
            downsample).cuda()

        self.network.eval()

    def get_weights(self):
        return self.network.get_weights()

    def set_weights(self, weights):
        self.network.set_weights(weights)

    def sample_actions(self, policy, n):
        return self.network.sample_actions(policy, n)

    def prediction(self, encoded_state):

        return self.network.prediction(encoded_state)

    def representation(self, observation):
        assert observation.max() < 10, "We assume that the observation is already pre-processed."
        observation = torch_utils.numpy_to_tensor(observation)
        return self.network.representation(observation)

    def dynamics(self, encoded_state, action):
        """

               :param encoded_state: [Batchsize, Encoded_channel_dim, Encoded_w, Encoded_h]
               :param action: shape: [Batchsize, Action_Dim]
               :return:
        """
        return self.network.dynamics(encoded_state, action)

    def sample_mixed_actions(self, policy, config):
        policy = np.array(policy)
        policy = torch_utils.numpy_to_tensor(policy)
        actions = self.network.sample_mixed_actions(policy, config)
        return actions

    def initial_inference(self, observation):
        observation = np.array(observation)
        observation = torch_utils.numpy_to_tensor(observation)
        a, b, c, d = self.network.initial_inference(observation)
        return torch_utils.tensor_to_numpy(a), torch_utils.tensor_to_numpy(b), \
               torch_utils.tensor_to_numpy(c), torch_utils.tensor_to_numpy(d)

    def recurrent_inference(self, encoded_state, action):
        """

        :param encoded_state: [Batchsize, Encoded_channel_dim, Encoded_w, Encoded_h]
        :param action: shape: [Batchsize, Action_Dim]
        :return:
        """
        encoded_state = np.array(encoded_state)
        action = np.array(action)
        encoded_state = torch_utils.numpy_to_tensor(encoded_state)
        action = torch_utils.numpy_to_tensor(action)
        a, b, c, d = self.network.recurrent_inference(encoded_state, action)
        return torch_utils.tensor_to_numpy(a), torch_utils.tensor_to_numpy(b), \
               torch_utils.tensor_to_numpy(c), torch_utils.tensor_to_numpy(d)

    def contrastive_loss(self, x1, x2):
        return self.network.contrastive_loss(x1, x2)

    def project(self, hidden_state, with_grad=True):
        return self.network.project(hidden_state, with_grad)
