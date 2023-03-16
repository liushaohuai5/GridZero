import copy
import time

import numpy
import ray
import torch
from pickle_utils import save_data
# import math
from torch_utils import profile
import numpy as np
import time
from algo.grid_v5.self_play import GameHistory

@ray.remote
class LowDimFastReplayBuffer(object):
    """
    Class which run in a dedicated thread to store played games and generate batch.
    """

    def __init__(self, initial_checkpoint, initial_buffer, config, test_throughput=False, shared_storage=None):
        self.config = config

        self.buffer = copy.deepcopy(initial_buffer)
        self.expert_buffer = copy.deepcopy(initial_buffer)

        def extend(buffer):
            new_buffer = {}
            counter = 0
            for k, v in buffer.items():
                for i in range(1):
                    new_buffer[counter] = v
                    counter += 1
            return new_buffer

        if test_throughput:
            self.buffer = extend(self.buffer)

        self.num_played_games = initial_checkpoint["num_played_games"]
        self.num_played_steps = initial_checkpoint["num_played_steps"]
        self.num_expert_games = 0
        self.num_expert_steps = 0
        self.total_expert_samples = 0
        self.total_samples = sum(
            [len(game_history.root_values) for game_history in self.buffer.values()]
            # [len(game_history.observation_history) for game_history in self.buffer.values()]
        )

        if self.total_samples != 0:
            print(
                f"Replay buffer initialized with {self.total_samples} samples ({self.num_played_games} games).\n"
            )

        # Fix random generator seed
        numpy.random.seed(self.config.seed)

        self.observations = []
        # self.actions = []
        self.state_mean, self.state_std = np.zeros(self.config.mlp_obs_shape), np.ones(self.config.mlp_obs_shape)
        # self.action_mean, self.action_std = np.zeros(self.config.mlp_action_shape), np.ones(self.config.mlp_action_shape)

        self.state_min, self.state_max = 1e6 * np.ones(self.config.mlp_obs_shape), -1e6 * np.ones(self.config.mlp_obs_shape)
        # self.action_min, self.action_max = 1e6 * np.ones(self.config.mlp_action_shape), -1e6 * np.ones(self.config.mlp_action_shape)

        self.shared_storage = shared_storage

        self.shared_storage.set_info.remote(
            {
                "state_min": self.state_min,
                "state_max": self.state_max,
                # "action_min": self.action_min,
                # "action_max": self.action_max,
                "state_mean": self.state_mean,
                "state_std": self.state_std,
                # "action_mean": self.action_mean,
                # "action_std": self.action_std,
                "num_expert_games": self.num_expert_games,
                "num_expert_steps": self.num_expert_steps,
            }
        )

        self.game_priorities = {}
        self.expert_priorities = {}
        self.shared_storage = shared_storage

    def compute_target_value(self, game_history, index):
        # The value target is the discounted root value of the search tree td_steps into the
        # future, plus the discounted sum of all rewards until then.
        bootstrap_index = index + self.config.td_steps
        if bootstrap_index < len(game_history.root_values):
            root_values = (
                game_history.root_values
                if game_history.reanalysed_predicted_root_values is None
                else game_history.reanalysed_predicted_root_values
            )
            last_step_value = (
                root_values[bootstrap_index]
            )
            value = last_step_value * self.config.discount ** self.config.td_steps
        else:
            value = 0

        for i, reward in enumerate(
            game_history.reward_history[index + 1 : bootstrap_index + 1]
        ):
            # The value is oriented from the perspective of the current player
            value += (reward) * self.config.discount ** i

        return value

    def save_pool(self, game_histories):
        for game_history in game_histories:
            self.save_game(game_history)

    def save_expert_pool(self, expert_histories):
        for expert_history in expert_histories:
            self.save_expert_game(expert_history)

    def save_game(self, game_history):

        if self.config.norm_type == 'mean_std':
            ori_states = ray.get(game_history.origin_state_history)
            # ori_states = game_history.origin_state_history
            if len(self.observations) < 4 * 1e3 * self.config.replay_buffer_size:
                for i in range(ori_states.shape[0]):
                    self.observations.append(ori_states[i])
            else:
                for i in range(ori_states.shape[0]):
                    del self.observations[i]
                    self.observations.append(ori_states[i])

        if self.config.PER:
            if game_history.priorities is not None:
                # Avoid read only array when loading replay buffer from disk
                game_history.priorities = numpy.copy(game_history.priorities)
            else:
                # Initial priorities for the prioritized replay (See paper appendix Training)
                priorities = []
                for i, root_value in enumerate(game_history.root_values):
                    # print(f'T_VALUE={self.compute_target_value(game_history, i)}')
                    priority = (
                        numpy.abs(
                            root_value - self.compute_target_value(game_history, i)
                        )
                    )
                    priorities.append(priority)

                game_history.priorities = numpy.array(priorities, dtype="float32")
            game_history.game_priority = numpy.max(game_history.priorities)

        self.buffer[self.num_played_games] = game_history
        self.game_priorities[self.num_played_games] = game_history.game_priority
        self.num_played_games += 1
        self.num_played_steps += len(game_history.root_values)
        self.total_samples += len(game_history.root_values)

        if self.config.replay_buffer_size < len(self.buffer):
            del_id = self.num_played_games - len(self.buffer)
            self.total_samples -= len(self.buffer[del_id].root_values)
            del self.buffer[del_id]
            del self.game_priorities[del_id]


        self.shared_storage.set_info.remote({
            "num_played_games": self.num_played_games,
            "num_played_steps": self.num_played_steps
        })

    def save_expert_game(self, game_history):
        x = time.time()

        if self.config.norm_type == 'mean_std':
            ori_states = ray.get(game_history.origin_state_history)
            # ori_states = game_history.origin_state_history
            if len(self.observations) < 4 * 1e3 * self.config.replay_buffer_size:
                for i in range(ori_states.shape[0]):
                    self.observations.append(ori_states[i])
            else:
                for i in range(ori_states.shape[0]):
                    del self.observations[i]
                    self.observations.append(ori_states[i])

        if self.config.PER:
            if game_history.priorities is not None:
                # Avoid read only array when loading replay buffer from disk
                game_history.priorities = numpy.copy(game_history.priorities)
            else:
                # Initial priorities for the prioritized replay (See paper appendix Training)
                priorities = []
                for i, root_value in enumerate(game_history.root_values):
                    # print(f'T_VALUE={self.compute_target_value(game_history, i)}')
                    priority = (
                        numpy.abs(
                            root_value - self.compute_target_value(game_history, i)
                        )
                    )
                    priorities.append(priority)

                game_history.priorities = numpy.array(priorities, dtype="float32")
            game_history.game_priority = numpy.max(game_history.priorities)

        self.expert_buffer[self.num_expert_games] = game_history
        self.expert_priorities[self.num_expert_games] = game_history.game_priority
        self.num_expert_games += 1
        self.num_expert_steps += len(game_history.root_values)
        self.total_expert_samples += len(game_history.root_values)

        if self.config.replay_buffer_size < len(self.expert_buffer):
            del_id = self.num_expert_games - len(self.expert_buffer)
            self.total_expert_samples -= len(self.expert_buffer[del_id].root_values)
            del self.expert_buffer[del_id]
            del self.expert_priorities[del_id]

        self.shared_storage.set_info.remote({
            "num_expert_games": self.num_expert_games,
            "num_expert_steps": self.num_expert_steps,
        })


    def calc_state_mean_std(self):
        if len(self.observations) > 0:
            observations = np.array(self.observations)
            self.state_mean, self.state_std = observations.mean(axis=0), observations.std(axis=0)
            self.shared_storage.set_info.remote({
                "state_mean": self.state_mean,
                "state_std": self.state_std
            })
            print('state_std_error', np.where(np.abs(self.state_std)==0))
            # print(f'mean_shape={self.state_mean}')
        return self.state_mean, self.state_std

    def get_mean_std(self):
        return self.state_mean, self.state_std

    def calc_action_mean_std(self):
        if len(self.actions) > 0:
            actions = np.stack(self.actions, axis=0)
            self.action_mean, self.action_std = actions.mean(axis=0), actions.std(axis=0)
            self.shared_storage.set_info.remote({
                "action_mean": self.action_mean,
                "action_std": self.action_std
            })

        return self.action_mean, self.action_std

    def calc_state_min_max(self):
        if len(self.observations) > 0:
            observations = np.array(self.observations, axis=0)
            self.state_min, self.state_max = observations.min(axis=0), observations.max(axis=0)
            self.shared_storage.set_info.remote({
                "state_min": self.state_min,
                "state_max": self.state_max
            })
        return self.state_min, self.state_max

    def get_state_min_max(self):
        return self.state_min, self.state_max

    def calc_action_min_max(self):
        if len(self.actions) > 0:
            actions = np.array(self.actions)
            self.action_min, self.action_max = actions.min(axis=0), actions.max(axis=0)
            self.shared_storage.set_info.remote({
                "action_min": self.action_min,
                "action_max": self.action_max
            })
        return self.action_min, self.action_max

    def get_action_min_max(self):
        return self.action_min, self.action_max

    def sample_position(self, game_history, force_uniform=False, for_attacker=False):
        """
        Sample position from game either uniformly or according to some priority.
        See paper appendix Training.
        """
        position_prob = None
        if self.config.PER and not force_uniform:
            position_probs = game_history.priorities / sum(game_history.priorities)
            position_index = numpy.random.choice(len(position_probs), p=position_probs)
            position_prob = position_probs[position_index]
        elif self.config.AER and not force_uniform:
            position_probs = numpy.array(game_history.attack_priorities) / sum(game_history.attack_priorities)
            position_index = numpy.random.choice(len(position_probs), p=position_probs)
            position_prob = position_probs[position_index]
        else:
            position_index = numpy.random.choice(len(game_history.root_values))
            # position_index = numpy.random.choice(len(game_history.root_values)//2)
            # position_index = position_index * 2 if not for_attacker else position_index * 2 + 1
            position_prob = 1 / len(game_history.root_values)
        return position_index, position_prob

    def get_buffer(self):
        return self.buffer

    def get_n_total_samples(self):
        return self.total_samples

    def sample_game(self, force_uniform=False):
        """
        Sample game from buffer either uniformly or according to some priority.
        See paper appendix Training.
        """
        game_prob = None
        if self.config.PER and not force_uniform:
            game_probs = numpy.array(
                [game_history.game_priority for game_history in self.buffer.values()],
                dtype="float32",
            )
            game_probs /= numpy.sum(game_probs)
            game_index = numpy.random.choice(len(self.buffer), p=game_probs)
            game_prob = game_probs[game_index]
        else:
            game_index = numpy.random.choice(len(self.buffer))
        game_id = self.num_played_games - len(self.buffer) + game_index

        return game_id, self.buffer[game_id], game_prob

    def sample_n_games_wrapper(self, n_games, force_uniform=False, rank=0):
        # if rank == 0:
        #     from line_profiler import LineProfiler
        #     lp = LineProfiler()
        #     lp_wrapper = lp(self.sample_n_games)
        #     ret = lp_wrapper(n_games, force_uniform=force_uniform)
        #     lp.print_stats()
        # else:
        ret = self.sample_n_games(n_games, force_uniform=False)

        return ret


    # @profile
    def sample_n_games(self, n_games, force_uniform=False):
        # TODO: increase sample priority of actual attacks' transitions
        if self.config.PER and not force_uniform:
            game_id_list = []
            game_probs = []
            for game_id, game_history in self.buffer.items():
                game_id_list.append(game_id)
                game_probs.append(game_history.game_priority)

            game_probs = numpy.array(game_probs, dtype="float32")
            game_probs = game_probs ** self.config.PER_alpha
            game_probs /= numpy.sum(game_probs)

            game_prob_dict = dict([(game_id, prob) for game_id, prob in zip(game_id_list, game_probs)])
            selected_games = numpy.random.choice(game_id_list,
                                                 n_games//2 if self.config.efficient_imitation else n_games,
                                                 p=game_probs)
        elif self.config.AER and not force_uniform:
            game_id_list = []
            game_probs = []
            for game_id, game_history in self.buffer.items():
                game_id_list.append(game_id)
                game_probs.append(game_history.game_attack_priority)

            game_probs = numpy.array(game_probs, dtype="float32")
            game_probs = game_probs ** self.config.PER_alpha
            game_probs /= numpy.sum(game_probs)

            game_prob_dict = dict([(game_id, prob) for game_id, prob in zip(game_id_list, game_probs)])
            selected_games = numpy.random.choice(game_id_list,
                                                 n_games // 2 if self.config.efficient_imitation else n_games,
                                                 p=game_probs)
        else:
            selected_games = numpy.random.choice(list(self.buffer.keys()),
                                                 n_games//2 if self.config.efficient_imitation else n_games)
            game_prob_dict = {}

        ret = []
        for i, game_id in enumerate(selected_games):
            game = self.buffer[game_id]
            game_pos, pos_prob = self.sample_position(game, force_uniform=False,
                                                      for_attacker=False if i < n_games // 2 else True)
            ret.append((game_id, game, game_prob_dict.get(game_id), game_pos, pos_prob))

        if self.config.efficient_imitation:
            selected_games = np.random.choice(list(self.expert_buffer.keys()), n_games//2)
            game_prob_dict = {}
            for game_id in selected_games:
                game = self.expert_buffer[game_id]
                game_pos, pos_prob = self.sample_position(game, force_uniform=True)
                ret.append((game_id, game, game_prob_dict.get(game_id), game_pos, pos_prob))

        return ret

    def update_game_history(self, game_id, game_history):
        # The element could have been removed since its selection and update
        if next(iter(self.buffer)) <= game_id:
            if self.config.PER:
                # Avoid read only array when loading replay buffer from disk
                game_history.priorities = numpy.copy(game_history.priorities)
            self.buffer[game_id] = game_history

    def update_game_history_reanalyze(self, game_ids, game_history_reanalyze_values):
        # The element could have been removed since its selection and update

        cmp_id = next(iter(self.buffer))
        for i, game_id in enumerate(game_ids):
            if cmp_id <= game_id:
                self.buffer[game_id].reanalysed_predicted_root_values = game_history_reanalyze_values[i]

    def update_priorities(self, priorities, index_info):
        """
        Update game and position priorities with priorities calculated during the training.
        See Distributed Prioritized Experience Replay https://arxiv.org/abs/1803.00933
        """

        for i in range(len(index_info)):
            game_id, game_pos = index_info[i]

            # The element could have been removed since its selection and training
            if next(iter(self.buffer)) <= game_id:
                # Update position priorities
                priority = priorities[i, :]
                start_index = game_pos
                end_index = min(
                    game_pos + len(priority), len(self.buffer[game_id].priorities)
                )
                self.buffer[game_id].priorities[start_index:end_index] = priority[
                    : end_index - start_index
                ]

                # Update game priorities
                self.buffer[game_id].game_priority = numpy.max(
                    self.buffer[game_id].priorities
                )

