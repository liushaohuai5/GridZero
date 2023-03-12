import sys
# print(sys.path)
import time

sys.path.append('/workspace/AdversarialGridZero')

import datetime
import os
import numpy as np
import gym
import numpy
import torch
# from algo.grid_v3.main import RZero
from algo.grid_v5.main import RZero
from game.gridsim import make_gridsim
# from game.gridsim.Agent.RandomAgent import RandomAgent
from game.gridsim.Agent.RuleAgent import RuleAgent
from arg_utils import *
import pickle_utils

from game.gridsim.utils import get_state_from_obs, form_action
from utilize.settings import settings
from game.gridsim.utils import bus_to_line_matrix
import torch.nn as nn

# try:
#     import cv2
# except ModuleNotFoundError:
#     raise ModuleNotFoundError('\nPlease run "pip install gym[atari]"')


class GridSimExperientConfig:
    def __init__(self
                 , env_id, task_name
                 ):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization
        self.env_id = env_id
        self.task_name = task_name
        self.exp_name = 'Experiment'

        self.efficient_imitation = False
        self.imitation = False
        self.imitation_steps = 20 * 1e3 if self.imitation else 0
        self.only_imitate_policy = False
        self.load_pretrain_model = False if self.imitation else False
        self.imitation_log_std = -2.0
        self.model_load_path = './best_v3.pth'

        self.add_attacker = False
        self.attack_all = False
        self.N_k = 1
        self.attack_prob = 0.2

        self.is_plot = True
        self.log_pic = False

        self.use_amp = False
        self.use_bn = False
        self.activation = nn.LeakyReLU
        # self.activation = nn.ReLU
        self.multi_reward = False
        self.ground_truth_running_cost_reward = False
        self.norm_type = 'mean_std'  # mean_std or min_max

        self.reward_func = 'self_reward'
        # self.reward_func = 'epri_reward'
        self.coeff_line_over_flow = 3
        self.coeff_line_disconnect = 1
        self.coeff_renewable_consumption = 2
        self.coeff_thermal_backup = 1
        self.coeff_running_cost = 4
        self.coeff_balanced_gen = 1
        self.coeff_gen_reactive_power = 2
        self.coeff_sub_voltage = 1

        self.reward_coeffs = [self.coeff_line_over_flow, self.coeff_renewable_consumption, self.coeff_running_cost, self.coeff_balanced_gen, self.coeff_gen_reactive_power]

        self.enable_close_gen = True
        self.multi_on_off = False

        # safety layer
        self.delta_range = 30
        self.bal_safe_range = 100

        self.parameters = {
            "only_power": True,
            "only_thermal": False,
            "voltage_action_type": 'reactive',
            "enable_close_gen": self.enable_close_gen,
            "multi_on_off": self.multi_on_off,
            "encoder": "mlp",
        }

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = 4  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available
        self.total_cpus = 80
        self.num_gpus = 8

        ### Game
        self.observation_shape = (1, 1, 100)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space_size = 1            #  Fixed list of all possible actions. You should only edit the length
        self.players = list(range(1))  # List of players. You should only edit the length
        self.stacked_observations = 1  # How many observations are stacked together.

        ### Self-Play
        self.num_workers = 12  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.num_expert_workers = 4
        self.n_parallel = 16
        self.expert_n_parallel = 1
        self.selfplay_on_gpu = True
        self.max_moves = 125  # Maximum number of moves if game is not finished before
        self.num_simulations = 50  # Number of future moves self-simulated
        self.discount = 0.99    # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25  # before is 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.init_zero = 1
        self.mlp_obs_shape = 1
        self.mlp_action_shape = 1

        self.mlp_hidden_shape = 256
        self.mlp_proj_shape = 256
        self.mlp_dyn_shape = [256, 256]
        self.mlp_rep_shape = [512, ]
        self.mlp_rew_shape = [256, 256]
        self.mlp_val_shape = [256, ]
        self.mlp_pi_shape = [256, ]
        self.mlp_expert_pi_shape = [256, ]
        self.mlp_proj_net_shape = [512, ]
        self.mlp_proj_pred_net_shape = [512, ]

        # THIS IS VERY IMPORTANT!!!!!
        # MCTS Action Samples
        self.mcts_num_policy_samples = 8    # Number of actions that will be sampled from the current policy.
        self.mcts_num_random_samples = 8    # Number of actions that will be sampled randomly.
        self.mcts_num_expert_samples = 4

        ### Training
        self.time_prefix = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results",
                                         os.path.basename(__file__)[:-3])  # Path to store the model weights and TensorBoard logs

        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = int(1e6)  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 256  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 1000
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available
        self.save_interval = 10000
        self.test_interval = 2000
        self.log_interval = 1000
        self.frame_skip = 8

        self.target_update_interval = 200
        self.selfplay_update_interval = self.checkpoint_interval

        self.optimizer = "SGD"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 2e-5  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.001  # Initial learning rate,
        self.lr_decay_rate = 0.5  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 50e3
        self.lr_decay_type = 'cosine'   # cosine or exponential
        self.warmup_steps = 10e3

        ### Replay Buffer
        self.replay_buffer_size = 1000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps_reanalyze = 5  # Number of game moves to keep for every batch element
        self.num_unroll_steps = 5

        self.td_steps = 5  # Number of steps in the future to take into account for calculating the target value
        self.AER = False
        self.PER = False  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.6  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1
        self.PER_beta = 0.4

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
        self.policy_loss_coeff = 1.0  # 1.0
        self.attacker_policy_loss_coeff = 1.0
        self.reward_loss_coeff = 1.0
        self.value_loss_coeff = 1.0    # Switch to 1.0 can be better .?
        self.entropy_loss_coeff = 0.001  # prev 0.01
        self.consistency_loss_coeff = 0.5
        self.imitation_loss_coeff = 0.01
        self.max_grad_norm = 10

        self.selfplay_model_serve_num = 4
        self.model_worker_serve_num = 2
        self.batch_worker_num = 24 if not self.only_imitate_policy else 1
        self.cpu_workers_num = 4
        self.support_size = 200
        self.value_support_step = 0.5
        self.reward_support_size = 200
        self.reward_support_step = 0.5

        self.ssl_target = 1
        self.explore_type = 'manual'    # add, normal, or manual or reject
        self.explore_scale = 1.0

        self.reward_delta = 0.05
        self.reward_amp = 1.0

    def set_result_path(self):
        self.results_path = os.path.join(self.results_path, self.exp_name + '_' + self.time_prefix)
        print("Result path:", self.results_path)

    def get_network_config(self):
        config = {
            'mlp_obs_shape':        self.mlp_obs_shape,
            'mlp_action_shape':     self.mlp_action_shape,
            'mlp_hidden_shape':     self.mlp_hidden_shape,
            'mlp_proj_shape':       self.mlp_proj_shape
        }
        return config

    def new_game(self, seed=0, reward_func=None, is_test=False, two_player=False, attack_all=False):
        rule_agent = RuleAgent(settings, this_directory_path='/workspace/AdversarialGridZero/game/gridsim')
        return make_gridsim(config=self, rule_agent=rule_agent, reward_func=reward_func, is_test=is_test, two_player=two_player, attack_all=attack_all)

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """

        if trained_steps < 50e3:
            return 1.0
        elif trained_steps < 70e3:
            return 0.5
        else:
            return 0.25

    def initialize(self):
        env = make_gridsim(config=self)
        obs = env.reset(ori_obs=True)

        action_dim_p = obs.action_space['adjust_gen_p'].shape[0]
        action_dim_v = obs.action_space['adjust_gen_v'].shape[0]
        self.generator_num = action_dim_p
        self.one_hot_dim = self.generator_num + 1

        if not self.attack_all:
            self.attacker_action_dim = len(settings.white_list_random_disconnection) + 1
        else:
            self.attacker_action_dim = len(settings.lnname) + 1

        if self.parameters['only_power']:
            self.parameters['action_dim'] = action_dim_p
            self.rew_dyn_act_dim = self.generator_num + (self.generator_num + 1) + (self.generator_num + 1)
            # self.rew_dyn_act_dim = self.generator_num + 2
            self.mlp_action_shape = self.rew_dyn_act_dim
            self.policy_act_dim = self.generator_num + self.generator_num + (self.generator_num + 1) + (self.generator_num + 1)
        else:
            self.parameters['action_dim'] = action_dim_p + action_dim_v
            self.rew_dyn_act_dim = 2 * self.generator_num + (self.generator_num + 1) + (self.generator_num + 1)
            self.mlp_action_shape = self.rew_dyn_act_dim
            self.policy_act_dim = 2 * self.generator_num + 2 * self.generator_num + (self.generator_num + 1) + (
                        self.generator_num + 1)
        state, ready_thermal_mask, closable_thermal_mask = get_state_from_obs(obs, settings, self.parameters)
        self.mlp_obs_shape = len(state)
        self.env_action_space = obs.action_space
        self.action_space_size = self.mlp_action_shape

    def sample_random_actions(self, n, is_attacker):
        if not is_attacker:
            actions = []
            for _ in range(n):
                action = np.random.randn(self.generator_num)
                action = action.clip(-0.999, 0.999)
                open_one_hot = np.zeros(self.one_hot_dim)
                open_one_hot[-1] = 1
                close_one_hot = np.zeros(self.one_hot_dim)
                close_one_hot[-1] = 1
                action = np.concatenate((action, open_one_hot, close_one_hot), axis=0)
                actions.append(action)
            return np.array(actions)
        else:
            # actions = np.eye(self.attacker_action_dim)[:n, :]
            all_actions = np.eye(self.attacker_action_dim)
            random_idx = np.random.choice(self.attacker_action_dim, n).tolist()
            actions = all_actions[random_idx]
            return actions

    def sample_random_actions_fast(self, n):
        return np.random.randn(n, self.mlp_action_shape)


if __name__ == '__main__':
    config = GridSimExperientConfig(env_id='grid', task_name='balance')
    parser = create_parser_from_config(config)
    parser.add_argument('--test', type=int, default=0)
    parser.add_argument('--checkpoint', type=str, default='')
    args = parser.parse_args()
    config = override_config_from_arg(config, args)
    config.set_result_path()
    config.initialize()

    np.random.seed(seed=config.seed)

    os.makedirs(config.results_path, exist_ok=True)
    dump_config(os.path.join(config.results_path, 'config.json'), config)
    for k, v in config.__dict__.items():
        print(k, v)

    config.PER = False
    config.AER = False
    agent = RZero(config)
    # assert len(settings.white_list_random_disconnection) + 1 == (config.mcts_num_policy_samples +
    #                                                              config.mcts_num_random_samples +
    #                                                              config.mcts_num_expert_samples), \
    #     'attacker and operator action space mismatch!'
    print(f'imitation={config.efficient_imitation}, load_model={config.load_pretrain_model}, load_path={config.model_load_path}, '
          f'variance={config.imitation_log_std}, PER={config.PER}, AER={config.AER}, norm_method={config.norm_type}, '
          f'enable_close_gen={config.enable_close_gen}, action_dim={config.mlp_action_shape}, add_attacker={config.add_attacker}, '
          f'attack_all={config.attack_all}, '
          f'use_amp={config.use_amp}, use_bn={config.use_bn}, activation={config.activation}')

    if not args.test:
        agent.train_new(log_in_tensorboard=False)
    else:
        rollouts = agent.test_full(args.checkpoint, 8, 2)
        pickle_utils.gsave_data(rollouts, args.checkpoint.replace(".pth", "_rollout.pkl"))

