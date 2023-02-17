import time

import torch
import torch_utils
import numpy as np
import mcts_tree.cytree as tree
from game.dmcontrol import make_dmcontrol_lowdim
import os

def run_multi(observations, model, config):
    root_num = observations.shape[0]

    with torch.no_grad():
        model.eval()

        pb_c_base, pb_c_init, discount = config.pb_c_base,config.pb_c_init, config.discount

        hidden_states_pool = []  # [NODE_ID, BATCHSIZE, H_DIM], CUDA_TENSORS
        actions_pool = []        # [NODE_ID, BATCHSIZE, N_ACTION, ACTION_DIM], CUDA_TENSORS>

        _, root_reward, policy_info, roots_hidden_state = \
           model.initial_inference(torch_utils.numpy_to_tensor(observations))

        # root_reward shape          [256]
        # roots_hidden_state_shape = [[256, h]]

        hidden_states_pool.append(torch_utils.tensor_to_numpy(roots_hidden_state))
        actions_pool.append(model.sample_mixed_actions(policy_info, config))
        hidden_state_idx_1 = 0

        n_total_actions = config.mcts_num_policy_samples + config.mcts_num_random_samples
        roots = tree.Roots(root_num, n_total_actions, config.num_simulations)
        noises = [np.random.dirichlet([config.root_dirichlet_alpha] * config.action_space_size).astype(
            np.float32).tolist() for _ in range(root_num)]

        roots.prepare(config.root_exploration_fraction, noises, root_reward.reshape(-1).tolist())

        min_max_stats_lst = tree.MinMaxStatsList(root_num)
        min_max_stats_lst.set_delta(0.01)

        for index_simulation in range(config.num_simulations):
            hidden_states = []
            selected_actions = []
            results = tree.ResultsWrapper(root_num)
            data_idxes_0, data_idxes_1, last_actions = \
                tree.multi_traverse(roots, pb_c_base, pb_c_init, discount, min_max_stats_lst, results)

            ptr = 0
            for idx_0, idx_1 in zip(data_idxes_0, data_idxes_1):
                hidden_states.append(hidden_states_pool[idx_1][idx_0])
                selected_actions.append(actions_pool[idx_1][idx_0][last_actions[ptr]])
                ptr += 1

            hidden_states = torch.from_numpy(np.asarray(hidden_states)).to('cuda').float()
            selected_actions = torch.from_numpy(np.asarray(selected_actions)).to('cuda').float()
            # print('SA', selected_actions.shape)

            leaves_value, leaves_reward, leaves_policy, leaves_hidden_state = \
                model.recurrent_inference(hidden_states, selected_actions)

            leaves_reward = leaves_reward.reshape(-1).tolist()
            leaves_value = leaves_value.reshape(-1).tolist()

            # Update the database
            hidden_states_pool.append(torch_utils.tensor_to_numpy(leaves_hidden_state))
            actions_pool.append(model.sample_mixed_actions(policy_info, config))
            hidden_state_idx_1 += 1

            # Back-propagate the reward information.
            tree.multi_back_propagate(hidden_state_idx_1, discount, leaves_reward,
                                      leaves_value, min_max_stats_lst, results)

    return roots.get_values(), roots.get_distributions(), actions_pool[0]

class DMControlExperientConfig:
    def __init__(self, env_id, task_name):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization
        self.env_id = env_id
        self.task_name = task_name

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = 4  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available
        self.total_cpus = 96

        ### Game
        self.dummy_game = make_dmcontrol_lowdim(env_id, task_name, 0)
        self.observation_shape = (3, 96, 96)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space_size = 1            #  Fixed list of all possible actions. You should only edit the length
        self.players = list(range(1))  # List of players. You should only edit the length
        self.stacked_observations = 4  # How many observations are stacked together.
        self.env_action_space = self.dummy_game.action_space

        ### Self-Play
        self.num_workers = 8  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = True
        self.max_moves = 250  # Maximum number of moves if game is not finished before
        self.num_simulations = 50  # Number of future moves self-simulated
        self.discount = 0.997  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time
        self.model_lr = 0.001

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.mlp_obs_shape = self.dummy_game.observation_space.shape[0]
        self.mlp_action_shape = self.dummy_game.action_space.shape[0]
        self.mlp_hidden_shape = 8
        self.mlp_proj_shape = 8

        # MCTS Action Samples
        self.mcts_num_policy_samples = 6    # Number of actions that will be sampled from the current policy.
        self.mcts_num_random_samples = 2    # Number of actions that will be sampled randomly.

        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = int(100e3)  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 256  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 100  # TODO(): Set back to 1e3, Number of training steps before using the model for self-playing
        # self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available
        self.save_interval = 10000

        self.target_update_interval = 200
        self.selfplay_update_interval = 100

        self.optimizer = "SGD"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 0.0001  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.01  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 30e3

        ### Replay Buffer
        self.replay_buffer_size = 300  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 2  # Number of game moves to keep for every batch element
        self.td_steps = 5  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 1  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = True

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step

        self.ratio_lower_bound = 1.0
        self.ratio_upper_bound = 1.0    # Desired training steps per self played step ratio. Equivalent to a synchronous version,
                                        # training can take much longer. Set it to None to disable it

        # Loss functions
        self.policy_loss_coeff = 0.05 #1.0
        self.reward_loss_coeff = 1.0
        self.value_loss_coeff  = 0.25
        self.entropy_loss_coeff = 0
        self.consistency_loss_coeff = 2.0
        self.max_grad_norm = 10

        self.selfplay_model_serve_num = 4
        self.model_worker_serve_num = 2
        self.batch_worker_num = 22

    def get_network_config(self):
        config = {
            'observation_shape':        self.observation_shape,
            'stacked_observations':     self.stacked_observations,
            'action_space_size':        self.action_space_size,
        }
        return config

    def new_game(self, seed=0):
        return  make_dmcontrol_lowdim(self.env_id, self.task_name, seed=seed)

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 50e3:
            return 1.0
        elif trained_steps < 75e3:
            return 0.5
        else:
            return 0.25

    def sample_random_actions(self, n):
        # acts = [-0.6, -0.4, -0.2, -0.1, 0, 0.1, 0.2, 0.4, 0.6]
        # import random
        # random.shuffle(acts)
        actions = [self.env_action_space.sample() for _ in range(n)]
        return np.array(actions)#.reshape(-1, 1)

    def sample_random_actions_fast(self, n):
        return np.random.randn(n, self.action_space_size)



if __name__ == '__main__':
    from algo.state_based_ez.rzero import MLPModel
    config = DMControlExperientConfig('cartpole', 'swingup')
    model = MLPModel(config).cuda()
    x = time.time()
    counter = 0
    while True:
        observations = np.random.randn(2560, 5)

        v, d, a = run_multi(observations, model, config)
        print(a.reshape(-1))
        counter += 1
        if counter == 10:
            print(time.time() - x)
            x = time.time()
            counter = 0

        #print('Done', v, d, a)
