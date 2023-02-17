import ray
import numpy as np
import time
import torch
import os

import csv

import torch_utils
from torch_utils import profile
from algo.grid_v5.mcts import MCTS
from mcts_tree_sample.dummy import run_multi_support

from game.gridsim.utils import *
from utilize.settings import settings
from algo.grid_v5.model import MLPModel
# from game.gridsim.Agent.RandomAgent import RandomAgent
import copy
from torch.cuda.amp import autocast
import collections
from Reward.rewards import *
from ori_opf_SG126 import traditional_solver



class GameHistory:
    """
    Store only useful information of a self-play game.
    """
    def __init__(self):
        self.observation_history = []
        self.origin_state_history = []
        self.render_history = []
        self.action_history = []
        self.reward_history = []
        self.reward_true_history = []

        self.line_overflow_rewards = []
        self.renewable_consumption_rewards = []
        self.running_cost_rewards = []
        self.balanced_gen_rewards = []
        self.reactive_power_rewards = []

        self.frames = []
        self.trees = []
        self.phys_states = []

        self.ready_mask_history = []
        self.closable_mask_history = []
        self.action_high_history = []
        self.action_low_history = []

        # self.to_play_history = []

        self.child_visits = []
        self.child_values = []
        self.child_qinits = []
        self.root_values = []
        self.root_actions = []

        self.reanalysed_predicted_root_values = None

        # For PER
        self.priorities = None
        self.game_priority = None

    def to_list(self):

        temp = [ray.put(np.array(self.observation_history).astype(np.float32)),
                ray.put(np.array(self.origin_state_history).astype(np.float32)),
                ray.put(np.array(self.action_history).astype(np.float32)),
                self.reward_history, self.reward_true_history,
                ray.put(np.array(self.ready_mask_history).astype(np.float32)),
                ray.put(np.array(self.closable_mask_history).astype(np.float32)),
                ray.put(np.array(self.action_high_history).astype(np.float32)),
                ray.put(np.array(self.action_low_history).astype(np.float32)),
                self.child_visits, self.child_values,
                self.child_qinits, self.root_values, self.root_actions]

        # temp_ref = ray.put(temp)
        # import ipdb
        # ipdb.set_trace()
        return temp

    def from_list(self, list_ref):
        # import ipdb
        # ipdb.set_trace()
        # list = ray.get(list_ref)
        self.observation_history, self.origin_state_history, self.action_history, self.reward_history, \
        self.reward_true_history, self.ready_mask_history, self.closable_mask_history, self.action_high_history, \
        self.action_low_history, self.child_visits, self.child_values, self.child_qinits, self.root_values, \
        self.root_actions = list_ref

        self.observation_history = ray.get(self.observation_history)
        self.origin_state_history = ray.get(self.origin_state_history)
        self.action_history = ray.get(self.action_history)
        self.ready_mask_history = ray.get(self.ready_mask_history)
        self.closable_mask_history = ray.get(self.closable_mask_history)
        self.action_high_history = ray.get(self.action_high_history)
        self.action_low_history = ray.get(self.action_low_history)

    def game_over(self):
        self.observation_history = ray.put(np.array(self.observation_history).astype(np.float32))
        self.origin_state_history = ray.put(np.array(self.origin_state_history).astype(np.float32))
        self.action_history = ray.put(np.array(self.action_history).astype(np.float32))
        self.ready_mask_history = ray.put(np.array(self.ready_mask_history).astype(np.float32))
        self.closable_mask_history = ray.put(np.array(self.closable_mask_history).astype(np.float32))
        self.action_high_history = ray.put(np.array(self.action_high_history).astype(np.float32))
        self.action_low_history = ray.put(np.array(self.action_low_history).astype(np.float32))
        self.root_actions = ray.put(np.array(self.root_actions).astype(np.float32))
        self.child_visits = ray.put(np.array(self.child_visits).astype(np.float32))
        self.child_qinits = ray.put(np.array(self.child_qinits).astype(np.float32))
        self.child_values = ray.put(np.array(self.child_values).astype(np.float32))
        self.reward_true_history = ray.put(np.array(self.reward_true_history).astype(np.float32))
        return self

    def subset(self, pos, duration):

        if pos < 0:
            pos = 0

        res = GameHistory()
        res.observation_history = ray.get(self.observation_history)[pos:pos + duration]
        res.origin_state_history = ray.get(self.origin_state_history)[pos:pos + duration]
        res.action_history = ray.get(self.action_history)[pos:pos + duration]
        res.reward_history = self.reward_history[pos:pos + duration]
        res.reward_true_history = ray.get(self.reward_true_history)[pos:pos + duration]

        res.line_overflow_rewards = self.line_overflow_rewards[pos:pos + duration]
        res.renewable_consumption_rewards = self.renewable_consumption_rewards[pos:pos + duration]
        res.running_cost_rewards = self.running_cost_rewards[pos:pos + duration]
        res.balanced_gen_rewards = self.balanced_gen_rewards[pos:pos + duration]
        res.reactive_power_rewards = self.reactive_power_rewards[pos:pos + duration]

        res.child_visits = ray.get(self.child_visits)[pos:pos + duration]
        res.root_values = self.root_values[pos:pos + duration]
        res.root_actions = ray.get(self.root_actions)[pos:pos + duration]
        res.ready_mask_history = ray.get(self.ready_mask_history)[pos:pos + duration]
        res.closable_mask_history = ray.get(self.closable_mask_history)[pos:pos + duration]
        res.action_high_history = ray.get(self.action_high_history)[pos:pos + duration]
        res.action_low_history = ray.get(self.action_low_history)[pos:pos + duration]

        if self.reanalysed_predicted_root_values is not None:
            res.reanalysed_predicted_root_values = self.reanalysed_predicted_root_values[pos:pos + duration]

        if self.priorities is not None:
            res.priorities = self.priorities[pos:pos + duration]

        return res

    def store_search_statistics(self, root):
        # Turn visit count from root into a policy
        if root is not None:
            sum_visits = sum(child.visit_count for child in root.children.values())
            root_action = []
            root_child_qinit = []
            root_child_value = []
            root_child_visit = []
            for action_id in root.action_ids:
                root_child_visit.append(root.children[action_id].visit_count / sum_visits)
                root_action.append(root.actions[action_id])
                root_child_qinit.append(root.children[action_id].q_init)
                root_child_value.append(root.children[action_id].value())

            self.child_qinits.append(root_child_qinit)
            self.child_values.append(root_child_value)
            self.child_visits.append(root_child_visit)
            self.root_actions.append(root_action)
            self.root_values.append(root.value())
        else:
            self.root_values.append(None)

    def get_stacked_info(self, index, num_stacked):
        index = index % len(self.observation_history)
        observations = []
        actions = []
        ready_masks = []
        closable_masks = []
        action_highs = []
        action_lows = []

        for past_observation_index in reversed(
                range(index + 1 - num_stacked, index + 1)
        ):
            if 0 <= past_observation_index:
                observations.append(self.observation_history[past_observation_index])
                if len(self.observation_history) > 1:
                    actions.append(self.action_history[past_observation_index + 1])
                    ready_masks.append(self.ready_mask_history[past_observation_index + 1])
                    closable_masks.append(self.closable_mask_history[past_observation_index + 1])
                    action_highs.append(self.action_high_history[past_observation_index + 1])
                    action_lows.append(self.action_low_history[past_observation_index + 1])
                else:
                    actions.append(self.action_history[past_observation_index])
                    ready_masks.append(self.ready_mask_history[past_observation_index])
                    closable_masks.append(self.closable_mask_history[past_observation_index])
                    action_highs.append(self.action_high_history[past_observation_index])
                    action_lows.append(self.action_low_history[past_observation_index])
            else:
                observations.append(self.observation_history[0])
                actions.append(self.action_history[0])
                ready_masks.append(self.ready_mask_history[0])
                closable_masks.append(self.closable_mask_history[0])
                action_highs.append(self.action_high_history[0])
                action_lows.append(self.action_low_history[0])

        observations = np.concatenate(observations, axis=0)
        actions = np.concatenate(actions, axis=0)
        ready_masks = np.concatenate(ready_masks, axis=0)
        closable_masks = np.concatenate(closable_masks, axis=0)
        action_highs = np.concatenate(action_highs, axis=0)
        action_lows = np.concatenate(action_lows, axis=0)

        return observations, actions, ready_masks, closable_masks, action_highs, action_lows


    def get_stacked_observations(self, index, num_stacked_observations):
        """
        Generate a new observation with the observation at the index position
        and num_stacked_observations past observations and actions stacked.
        """

        # Convert to positive index
        index = index % len(self.observation_history)

        # [t, t-1, t-2, ...]
        # stacked_observations = self.observation_history[index].copy()
        observations = []

        for past_observation_index in reversed(
            range(index + 1 - num_stacked_observations, index + 1)
        ):

            if 0 <= past_observation_index:
                observations.append(self.observation_history[past_observation_index])

            else:
                observations.append(self.observation_history[0])

        stacked_observations = np.concatenate(observations, axis=0)
        return stacked_observations

    def get_stacked_origin_states(self, index, num_stacked_observations):
        """
        Generate a new observation with the observation at the index position
        and num_stacked_observations past observations and actions stacked.
        """

        # Convert to positive index
        index = index % len(self.observation_history)

        # [t, t-1, t-2, ...]
        # stacked_observations = self.observation_history[index].copy()
        origin_states = []

        for past_observation_index in reversed(
            range(index + 1 - num_stacked_observations, index + 1)
        ):

            if 0 <= past_observation_index:
                origin_states.append(self.origin_state_history[past_observation_index])

            else:
                origin_states.append(self.origin_state_history[0])

        stacked_origin_states = np.concatenate(origin_states, axis=0)
        return stacked_origin_states

    def get_stacked_actions(self, index, num_stacked_actions):
        index = index % len(self.observation_history)
        actions = []
        for past_observation_index in reversed(
                range(index + 1 - num_stacked_actions, index + 1)
        ):
            if 0 <= past_observation_index:
                actions.append(self.action_history[past_observation_index])

            else:
                actions.append(self.action_history[0])

        stacked_actions = np.concatenate(actions, axis=0)
        return stacked_actions

    def get_stacked_masks(self, index, num_stacked_observations):
        """
        Generate a new observation with the observation at the index position
        and num_stacked_observations past observations and actions stacked.
        """

        # Convert to positive index
        index = index % len(self.observation_history)

        # [t, t-1, t-2, ...]
        ready_masks, closable_masks = [], []

        for past_observation_index in reversed(
            range(index + 1 - num_stacked_observations, index + 1)
        ):

            if 0 <= past_observation_index:
                ready_masks.append(self.ready_mask_history[past_observation_index])
                closable_masks.append(self.closable_mask_history[past_observation_index])

            else:
                ready_masks.append(self.ready_mask_history[0])
                closable_masks.append(self.closable_mask_history[0])

        stacked_ready_masks = np.concatenate(ready_masks, axis=0)
        stacked_closable_masks = np.concatenate(closable_masks, axis=0)
        return stacked_ready_masks, stacked_closable_masks


    def get_stacked_action_space(self, index, num_stacked_observations):
        """
        Generate a new observation with the observation at the index position
        and num_stacked_observations past observations and actions stacked.
        """

        # Convert to positive index
        index = index % len(self.observation_history)

        # [t, t-1, t-2, ...]
        action_highs, action_lows = [], []

        for past_observation_index in reversed(
            range(index + 1 - num_stacked_observations, index + 1)
        ):

            if 0 <= past_observation_index:
                action_highs.append(self.action_high_history[past_observation_index])
                action_lows.append(self.action_low_history[past_observation_index])

            else:
                action_highs.append(self.action_high_history[0])
                action_lows.append(self.action_low_history[0])

        stacked_action_highs = np.concatenate(action_highs, axis=0)
        stacked_action_lows = np.concatenate(action_lows, axis=0)
        return stacked_action_highs, stacked_action_lows



def reward_shaping_func(r, delta=1, amp=1):
    if delta == 0:
        return r
    return (r // delta) * delta * amp

@ray.remote(num_gpus=0.12)
class ExpertPlay:
    def __init__(self, id, config, seed, shared_storage, replay_buffer):
        self.model = MLPModel(config)
        self.model.to(torch.device("cuda"))
        self.model.eval()

        self.shared_storage = shared_storage
        self.replay_buffer = replay_buffer

        self.n_parallel = config.n_parallel
        self.expert_n_parallel = config.expert_n_parallel
        self.envs = [config.new_game(seed+i,
                                     reward_func=config.reward_func
                                     )
                     for i in range(config.expert_n_parallel)]
        self.config = config

        self.last_model_index = -1
        self.env_tmp = config.new_game(0, reward_func=config.reward_func)
        self.traditional_solver = traditional_solver(self.config, self.env_tmp.reset(ori_obs=True), dc_opf_flag=True,
                                                     unit_comb_flag=True, for_training_flag=True)
        self.id = id

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def spin(self):
        # time.sleep(5*self.id)
        training_steps = ray.get(self.shared_storage.get_info.remote("training_step"))
        # time.sleep(1000)
        while training_steps < self.config.training_steps:
            training_steps = ray.get(self.shared_storage.get_info.remote("training_step"))
            new_model_index = training_steps // self.config.checkpoint_interval
            if new_model_index > self.last_model_index:
                self.last_model_index = new_model_index
                self.model.set_weights(ray.get(self.shared_storage.get_info.remote("weights")))
                self.model.to('cuda')
                self.model.eval()
                # print("selfplay update!!!!!!!!")

            try:
                game_histories = self.play_expert_games(training_steps)
            except:
                continue

            for game_history in game_histories:
                game_history.game_over()

            self.replay_buffer.save_expert_pool.remote(game_histories)

    def play_expert_games(self, trained_steps=0):
        # import ipdb
        # ipdb.set_trace()
        if self.config.norm_type == 'min_max':
            state_min = ray.get(self.shared_storage.get_info.remote("state_min"))
            state_max = ray.get(self.shared_storage.get_info.remote("state_max"))
            action_min = ray.get(self.shared_storage.get_info.remote("action_min"))
            action_max = ray.get(self.shared_storage.get_info.remote("action_max"))
        else:
            state_mean = ray.get(self.shared_storage.get_info.remote("state_mean"))
            state_std = ray.get(self.shared_storage.get_info.remote("state_std"))

        observations = [env.reset(ori_obs=True) for env in self.envs]

        game_histories = [GameHistory() for i in range(self.expert_n_parallel)]
        ori_states, ready_masks, closable_masks, action_highs, action_lows = [], [], [], [], []
        for i, game_history in enumerate(game_histories):
            state, ready_mask, closable_mask = get_state_from_obs(observations[i], settings, self.config.parameters)
            if self.config.norm_type == 'min_max':
                state_norm = (state - state_min) / (state_max - state_min + 1e-4)
            else:
                state_norm = (state - state_mean) / (state_std + 1e-4)
            action_high, action_low = get_action_space(observations[i], self.config.parameters, settings)
            random_action = self.config.sample_random_actions(1).reshape(-1)
            game_history.action_history.append(random_action)
            game_history.observation_history.append(state_norm)
            game_history.origin_state_history.append(state)
            game_history.reward_history.append(0)
            game_history.reward_true_history.append(0)
            game_history.line_overflow_rewards.append(0)
            game_history.renewable_consumption_rewards.append(0)
            game_history.running_cost_rewards.append(0)
            game_history.balanced_gen_rewards.append(0)
            game_history.reactive_power_rewards.append(0)
            game_history.ready_mask_history.append(ready_mask)
            game_history.closable_mask_history.append(closable_mask)
            game_history.action_high_history.append(action_high)
            game_history.action_low_history.append(action_low)

            ori_states.append(state)
            ready_masks.append(ready_mask)
            closable_masks.append(closable_mask)
            action_highs.append(action_high)
            action_lows.append(action_low)

        ori_states = np.array(ori_states)
        dones = [0 for i in range(self.expert_n_parallel)]
        infos = [{} for i in range(self.expert_n_parallel)]
        if self.config.norm_type == 'min_max':
            states_norm = (ori_states - state_min) / (state_max - state_min + 1e-4)
        else:
            states_norm = (ori_states - state_mean) / (state_std + 1e-4)
        ready_masks = np.array(ready_masks)
        closable_masks = np.array(closable_masks)
        action_highs = np.array(action_highs)
        action_lows = np.array(action_lows)

        steps = 0
        while sum(dones) < self.expert_n_parallel and steps < self.config.max_moves:
            steps += 1
            root_values, root_distributions, root_actions = run_multi_support(
                observations=states_norm,
                model=self.model,
                config=self.config,
                ready_masks=ready_masks,
                closable_masks=closable_masks,
                action_highs=action_highs,
                action_lows=action_lows,
                origin_states=ori_states,
                train_steps=trained_steps
            )
            root_values = np.array(root_values)  # [EsNV_ID, 1]
            root_actions = np.array(root_actions)  # [ENV_ID, N_ACTIONS, ACTION_DIM]
            root_visit_counts = np.array(root_distributions).astype(np.float32)  # [ENV_ID, N_ACTIONS, 1]

            next_ori_states, next_ready_masks, next_closable_masks, next_action_highs, next_action_lows = [], [], [], [], []
            next_observations = []

            for i, game_history in enumerate(game_histories):
                if dones[i]:
                    next_observations.append(observations[i])
                    next_ori_states.append(ori_states[i])
                    next_ready_masks.append(ready_masks[i])
                    next_closable_masks.append(closable_masks[i])
                    next_action_highs.append(action_highs[i])
                    next_action_lows.append(action_lows[i])
                    continue

                mcts_value = root_values[i]
                mcts_action = root_actions[i]
                mcts_visit_count = root_visit_counts[i]

                action_1 = self.select_action(mcts_action, mcts_visit_count,
                                            self.config.visit_softmax_temperature_fn(trained_steps))
                if not self.config.multi_on_off:
                    action_true_1 = combine_one_hot_action(ori_states[i], action_1, ready_masks[i], closable_masks[i], self.config, action_highs[i], action_lows[i])
                else:
                    action_true_1 = modify_action_v3(action_1, ready_masks[i], closable_masks[i], self.config, action_highs[i], action_lows[i])

                action_true, _, open_hot, close_hot = self.traditional_solver.run_opf(observations[i])
                action_true = action_true.clip(action_lows[i], action_highs[i])
                action = (action_true - action_lows[i]) / (action_highs[i] - action_lows[i] + 1e-3) * 2 - 1
                action *= (np.ones_like(ready_masks[i]) - ready_masks[i])[:-1]
                action *= (np.ones_like(closable_masks[i]) - closable_masks[i])[:-1]
                action[settings.balanced_id] = 0.0
                # if abs(action).max() > 1.1:
                #     import ipdb
                #     ipdb.set_trace()

                action = np.concatenate((action, open_hot, close_hot))
                next_observation, reward, done, info = self.envs[i].step(action_true, ori_obs=True)
                # print(f'step={len(game_history.action_history)}, reward={reward:.3f}')
                state, ready_mask, closable_mask = get_state_from_obs(next_observation, settings, self.config.parameters)
                if self.config.norm_type == 'min_max':
                    state_norm = (state - state_min) / (state_max - state_min + 1e-4)
                else:
                    state_norm = (state - state_mean) / (state_std + 1e-4)
                action_high, action_low = get_action_space(next_observation, self.config.parameters, settings)
                reward_clipped = reward_shaping_func(reward, self.config.reward_delta, self.config.reward_amp)
                next_observations.append(next_observation)
                next_ori_states.append(state)
                next_ready_masks.append(ready_mask)
                next_closable_masks.append(closable_mask)
                next_action_highs.append(action_high)
                next_action_lows.append(action_low)

                if done:
                    dones[i] = 1
                    infos[i] = info

                game_history.action_history.append(action)
                game_history.observation_history.append(state_norm)
                game_history.origin_state_history.append(state)
                game_history.reward_history.append(reward_clipped)
                game_history.reward_true_history.append(reward)

                game_history.line_overflow_rewards.append(line_over_flow_reward(next_observation, settings))
                game_history.renewable_consumption_rewards.append(
                    renewable_consumption_reward(next_observation, settings))
                game_history.running_cost_rewards.append(
                    running_cost_reward_v2(next_observation, self.envs[i].last_obs, settings))
                game_history.balanced_gen_rewards.append(balanced_gen_reward(next_observation, settings))
                game_history.reactive_power_rewards.append(gen_reactive_power_reward(next_observation, settings))

                game_history.child_visits.append(mcts_visit_count)
                game_history.root_actions.append(mcts_action)
                game_history.root_values.append(mcts_value)
                game_history.ready_mask_history.append(ready_mask)
                game_history.closable_mask_history.append(closable_mask)
                game_history.action_high_history.append(action_high)
                game_history.action_low_history.append(action_low)

            observations = copy.deepcopy(next_observations)
            ori_states = np.array(next_ori_states)
            # states_norm = (ori_states - state_min) / (state_max - state_min + 1e-3)
            if self.config.norm_type == 'min_max':
                states_norm = (ori_states - state_min) / (state_max - state_min + 1e-4)
            else:
                states_norm = (ori_states - state_mean) / (state_std + 1e-4)
            ready_masks = np.array(next_ready_masks)
            closable_masks = np.array(next_closable_masks)
            action_highs = np.array(next_action_highs)
            action_lows = np.array(next_action_lows)

        # state_mins, state_maxs, action_mins, action_maxs = [], [], [], []
        for i, game_history in enumerate(game_histories):
            print("Expert, Len={}, Reward={}, info={}".format(len(game_history.reward_true_history),
                                             sum(game_history.reward_true_history), infos[i]))
            if self.config.norm_type == 'min_max':
                ori_states = np.array(game_history.origin_state_history)
                state_min = np.concatenate((ori_states, state_min.reshape(1, *state_min.shape)), axis=0).min(axis=0)
                state_max = np.concatenate((ori_states, state_max.reshape(1, *state_max.shape)), axis=0).max(axis=0)
                ori_actions = np.array(game_history.action_history)
                action_max = np.concatenate((ori_actions, action_max.reshape(1, *action_max.shape)), axis=0).max(axis=0)
                action_min = np.concatenate((ori_actions, action_min.reshape(1, *action_min.shape)), axis=0).min(axis=0)

        if self.config.norm_type == 'min_max':
            self.shared_storage.set_info.remote({
                "state_min": state_min,
                "state_max": state_max,
                "action_min": action_min,
                "action_max": action_max
            })
        return game_histories

    def select_action(self, actions, visit_counts, temperature):
        """
      Select action according to the visit count distribution and the temperature.
      The temperature is changed dynamically with the visit_softmax_temperature function
      in the config.
      """
        # n_child_action = len(node.children)
        #
        # visit_counts = np.array(
        #     [node.children[action_id].visit_count for action_id in node.action_ids], dtype="int32"
        # )

        if temperature == 0:
            action = actions[np.argmax(visit_counts)]

        elif temperature == float("inf"):
            action_id = np.random.choice(visit_counts.shape[0])
            action = actions[action_id]

        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action_id = np.random.choice(visit_counts.shape[0], p=visit_count_distribution)
            action = actions[action_id]

        return action

# @ray.remote(num_gpus=0.08)
@ray.remote(num_gpus=0.12)
class SelfPlay:
    def __init__(self, id, config, seed, shared_storage, replay_buffer):
        self.model = MLPModel(config)
        self.model.to(torch.device("cuda"))
        self.model.eval()

        self.shared_storage = shared_storage
        self.replay_buffer = replay_buffer

        self.n_parallel = config.n_parallel
        self.expert_n_parallel = config.expert_n_parallel
        self.envs = [config.new_game(seed+i,
                                     reward_func=config.reward_func
                                     )
                     for i in range(config.n_parallel)]
        self.config = config

        self.last_model_index = -1
        # time.sleep(10000)

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def spin(self):
        training_steps = ray.get(self.shared_storage.get_info.remote("training_step"))
        # time.sleep(1000)
        while training_steps < self.config.training_steps:
            training_steps = ray.get(self.shared_storage.get_info.remote("training_step"))
            new_model_index = training_steps // self.config.checkpoint_interval
            if new_model_index > self.last_model_index:
                self.last_model_index = new_model_index
                self.model.set_weights(ray.get(self.shared_storage.get_info.remote("weights")))
                self.model.to('cuda')
                self.model.eval()
                # print("selfplay update!!!!!!!!")

            # try:
            game_histories = self.play_multi_games(training_steps)
            # except:
            #     continue
            self.shared_storage.set_info.remote(
                {
                    "episode_length": len(game_histories[0].reward_history) - 1,
                    "total_reward": sum(game_histories[0].reward_history),
                    "total_true_reward": sum(game_histories[0].reward_true_history),
                    "mean_value": np.mean(
                        [value for value in game_histories[0].root_values if value]
                    ),
                }
            )

            for game_history in game_histories:
                game_history.game_over()

            self.replay_buffer.save_pool.remote(game_histories)

    # @profile
    def play_multi_games(self, trained_steps=0):
        if self.config.norm_type == 'min_max':
            state_min = ray.get(self.shared_storage.get_info.remote("state_min"))
            state_max = ray.get(self.shared_storage.get_info.remote("state_max"))
            action_min = ray.get(self.shared_storage.get_info.remote("action_min"))
            action_max = ray.get(self.shared_storage.get_info.remote("action_max"))
        else:
            state_mean = ray.get(self.shared_storage.get_info.remote("state_mean"))
            state_std = ray.get(self.shared_storage.get_info.remote("state_std"))

        observations = [env.reset(ori_obs=True) for env in self.envs]

        game_histories = [GameHistory() for i in range(self.n_parallel)]
        ori_states, ready_masks, closable_masks, action_highs, action_lows = [], [], [], [], []
        for i, game_history in enumerate(game_histories):
            state, ready_mask, closable_mask = get_state_from_obs(observations[i], settings, self.config.parameters)
            if self.config.norm_type == 'min_max':
                state_norm = (state - state_min) / (state_max - state_min + 1e-4)
            else:
                state_norm = (state - state_mean) / (state_std + 1e-4)
            action_high, action_low = get_action_space(observations[i], self.config.parameters, settings)
            random_action = self.config.sample_random_actions(1).reshape(-1)
            game_history.action_history.append(random_action)
            # if self.config.parameters['only_power']:
            #     game_history.action_history.append(np.concatenate((random_action, random_action, random_action), axis=-1))
            # else:
            #     game_history.action_history.append(
            #         np.concatenate((random_action, random_action, random_action[:2]), axis=-1))
            game_history.observation_history.append(state_norm)
            game_history.origin_state_history.append(state)
            game_history.reward_history.append(0)
            game_history.reward_true_history.append(0)
            game_history.line_overflow_rewards.append(0)
            game_history.renewable_consumption_rewards.append(0)
            game_history.running_cost_rewards.append(0)
            game_history.balanced_gen_rewards.append(0)
            game_history.reactive_power_rewards.append(0)
            game_history.ready_mask_history.append(ready_mask)
            game_history.closable_mask_history.append(closable_mask)
            game_history.action_high_history.append(action_high)
            game_history.action_low_history.append(action_low)

            ori_states.append(state)
            ready_masks.append(ready_mask)
            closable_masks.append(closable_mask)
            action_highs.append(action_high)
            action_lows.append(action_low)

        ori_states = np.array(ori_states)
        dones = [0 for i in range(self.n_parallel)]
        infos = [{} for i in range(self.n_parallel)]
        if self.config.norm_type == 'min_max':
            states_norm = (ori_states - state_min) / (state_max - state_min + 1e-4)
        else:
            states_norm = (ori_states - state_mean) / (state_std + 1e-4)
        ready_masks = np.array(ready_masks)
        closable_masks = np.array(closable_masks)
        action_highs = np.array(action_highs)
        action_lows = np.array(action_lows)

        steps = 0
        while sum(dones) < self.n_parallel and steps < self.config.max_moves:
            steps += 1
            root_values, root_distributions, root_actions = run_multi_support(
                observations=states_norm,
                model=self.model,
                config=self.config,
                ready_masks=ready_masks,
                closable_masks=closable_masks,
                action_highs=action_highs,
                action_lows=action_lows,
                origin_states=ori_states,
                train_steps=trained_steps
            )
            root_values = np.array(root_values)  # [EsNV_ID, 1]
            root_actions = np.array(root_actions)  # [ENV_ID, N_ACTIONS, ACTION_DIM]
            root_visit_counts = np.array(root_distributions).astype(np.float32)  # [ENV_ID, N_ACTIONS, 1]

            next_ori_states, next_ready_masks, next_closable_masks, next_action_highs, next_action_lows = [], [], [], [], []

            for i, game_history in enumerate(game_histories):
                if dones[i]:
                    # next_observations.append(observations[i])
                    next_ori_states.append(ori_states[i])
                    next_ready_masks.append(ready_masks[i])
                    next_closable_masks.append(closable_masks[i])
                    next_action_highs.append(action_highs[i])
                    next_action_lows.append(action_lows[i])
                    continue

                mcts_value = root_values[i]
                mcts_action = root_actions[i]
                mcts_visit_count = root_visit_counts[i]

                action = self.select_action(mcts_action, mcts_visit_count,
                                            self.config.visit_softmax_temperature_fn(trained_steps))
                if not self.config.multi_on_off:
                    # action_true = modify_action_v2(action, ready_masks[i], closable_masks[i], self.config, action_highs[i], action_lows[i])
                    action_true = combine_one_hot_action(ori_states[i], action, ready_masks[i], closable_masks[i], self.config, action_highs[i], action_lows[i])
                    # if self.config.enable_close_gen:
                    #     action_true = action_true[:-2]
                    # else:
                    #     action_true = action_true[:-1]
                else:
                    action_true = modify_action_v3(action, ready_masks[i], closable_masks[i], self.config, action_highs[i], action_lows[i])
                next_observation, reward, done, info = self.envs[i].step(action_true, ori_obs=True)
                state, ready_mask, closable_mask = get_state_from_obs(next_observation, settings, self.config.parameters)
                if self.config.norm_type == 'min_max':
                    state_norm = (state - state_min) / (state_max - state_min + 1e-4)
                else:
                    state_norm = (state - state_mean) / (state_std + 1e-4)
                action_high, action_low = get_action_space(next_observation, self.config.parameters, settings)
                reward_clipped = reward_shaping_func(reward, self.config.reward_delta, self.config.reward_amp)
                # next_observations.append(next_observation)
                next_ori_states.append(state)
                next_ready_masks.append(ready_mask)
                next_closable_masks.append(closable_mask)
                next_action_highs.append(action_high)
                next_action_lows.append(action_low)

                if done:
                    dones[i] = 1
                    infos[i] = info

                game_history.action_history.append(action)
                game_history.observation_history.append(state_norm)
                game_history.origin_state_history.append(state)
                # game_history.reward_history.append(reward_clipped)
                game_history.reward_history.append(reward)
                game_history.reward_true_history.append(reward)

                game_history.line_overflow_rewards.append(line_over_flow_reward(next_observation, settings))
                game_history.renewable_consumption_rewards.append(renewable_consumption_reward(next_observation, settings))
                game_history.running_cost_rewards.append(running_cost_reward_v2(next_observation, self.envs[i].last_obs, settings))
                game_history.balanced_gen_rewards.append(balanced_gen_reward(next_observation, settings))
                game_history.reactive_power_rewards.append(gen_reactive_power_reward(next_observation, settings))

                game_history.child_visits.append(mcts_visit_count)
                game_history.root_actions.append(mcts_action)
                game_history.root_values.append(mcts_value)
                game_history.ready_mask_history.append(ready_mask)
                game_history.closable_mask_history.append(closable_mask)
                game_history.action_high_history.append(action_high)
                game_history.action_low_history.append(action_low)

            # observations = np.array(next_observations)
            ori_states = np.array(next_ori_states)
            # states_norm = (ori_states - state_min) / (state_max - state_min + 1e-3)
            if self.config.norm_type == 'min_max':
                states_norm = (ori_states - state_min) / (state_max - state_min + 1e-4)
            else:
                states_norm = (ori_states - state_mean) / (state_std + 1e-4)
            ready_masks = np.array(next_ready_masks)
            closable_masks = np.array(next_closable_masks)
            action_highs = np.array(next_action_highs)
            action_lows = np.array(next_action_lows)

        # state_mins, state_maxs, action_mins, action_maxs = [], [], [], []
        line_overflow_mean, renewable_consumption_mean, running_cost_mean, bal_gen_mean, reac_power_mean = 0, 0, 0, 0, 0

        for i, game_history in enumerate(game_histories):
            # print("Len={}, Reward={}, info={}".format(len(game_history.reward_true_history),
            #                                  sum(game_history.reward_true_history), infos[i]))

            line_overflow_mean += np.mean(game_history.line_overflow_rewards)
            renewable_consumption_mean += np.mean(game_history.renewable_consumption_rewards)
            running_cost_mean += np.mean(game_history.running_cost_rewards)
            bal_gen_mean += np.mean(game_history.balanced_gen_rewards)
            reac_power_mean += np.mean(game_history.reactive_power_rewards)

            if self.config.norm_type == 'min_max':
                ori_states = np.array(game_history.origin_state_history)
                state_min = np.concatenate((ori_states, state_min.reshape(1, *state_min.shape)), axis=0).min(axis=0)
                state_max = np.concatenate((ori_states, state_max.reshape(1, *state_max.shape)), axis=0).max(axis=0)
                ori_actions = np.array(game_history.action_history)
                action_max = np.concatenate((ori_actions, action_max.reshape(1, *action_max.shape)), axis=0).max(axis=0)
                action_min = np.concatenate((ori_actions, action_min.reshape(1, *action_min.shape)), axis=0).min(axis=0)

        self.shared_storage.set_info.remote({
            "line_overflow_mean": line_overflow_mean / self.config.n_parallel,
            "renewable_consumption_mean": renewable_consumption_mean / self.config.n_parallel,
            "running_cost_mean": running_cost_mean / self.config.n_parallel,
            "bal_gen_mean": bal_gen_mean / self.config.n_parallel,
            "reac_power_mean": reac_power_mean / self.config.n_parallel
        })
        if self.config.norm_type == 'min_max':
            self.shared_storage.set_info.remote({
                "state_min": state_min,
                "state_max": state_max,
                "action_min": action_min,
                "action_max": action_max
            })
        return game_histories



    def select_action(self, actions, visit_counts, temperature):
        """
      Select action according to the visit count distribution and the temperature.
      The temperature is changed dynamically with the visit_softmax_temperature function
      in the config.
      """
        # n_child_action = len(node.children)
        #
        # visit_counts = np.array(
        #     [node.children[action_id].visit_count for action_id in node.action_ids], dtype="int32"
        # )

        if temperature == 0:
            action = actions[np.argmax(visit_counts)]

        elif temperature == float("inf"):
            action_id = np.random.choice(visit_counts.shape[0])
            action = actions[action_id]

        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action_id = np.random.choice(visit_counts.shape[0], p=visit_count_distribution)
            action = actions[action_id]

        return action

# @ray.remote(num_gpus=0.08)
@ray.remote(num_gpus=0.12)
class LowDimTestWorker:
    """
    Self-play workers running in parallel.

    """
    def __init__(self, checkpoint, config, seed, shared_storage, replay_buffer):
        self.config = config
        self.checkpoint = checkpoint
        # print(config.parameters['action_dim'])
        self.game = config.new_game(seed)
        self.model = MLPModel(config)
        self.model.set_weights(copy.deepcopy(self.checkpoint["weights"]))
        self.model.to('cuda')
        self.model.eval()
        self.seed = seed
        self.shared_storage = shared_storage
        self.replay_buffer = replay_buffer
        self.Q = collections.deque(maxlen=10)

        # self.env_tmp = config.new_game(0, reward_func=config.reward_func)
        # self.traditional_solver = traditional_solver(self.config, self.env_tmp.reset(ori_obs=True), dc_opf_flag=True,
        #                                              unit_comb_flag=True, for_training_flag=True)


    def spin(self):
        """

        :param shared_storage:
        :param replay_buffer:
        :param test_mode:
        :return:
        """

        from game.gridsim.Agent.RuleAgent import RuleAgent
        rule_agent = RuleAgent(settings, this_directory_path='/workspace/GridZero/game/gridsim')
        training_step = 0
        last_test_index = 0
        start_sample_idx = [
            22753, 16129, 74593, 45793, 32257, 53569, 13826,
            26785,
            1729, 17281,
            34273, 36289, 44353, 52417, 67105, 75169, 289, 4897, 15841, 31969,
            16980-144
            # 67600,
            #                 15250, 6320, 3980, 6620, 94800,
                            # 16324, 24691,
                            # 450, 500, 550, 600
                            ]
        while training_step < self.config.training_steps:
            training_step = ray.get(self.shared_storage.get_info.remote("training_step"))
            if training_step // self.config.test_interval >= last_test_index:
                voltage_violations, reactive_violations, bal_p_violations, soft_overflows, hard_overflows = 0, 0, 0, 0, 0
                running_costs = []
                renewable_consumption_rate = []
                self.model.set_weights(ray.get(self.shared_storage.get_info.remote("weights")))
                self.model.to('cuda')
                self.model.eval()
                print(f'test_model updated!!!')
                last_test_index += 1
                test_episode_lengths, test_total_rewards, test_mean_values = [], [], []
                for i in range(len(start_sample_idx)):
                    game_history, epi_vol_voilations, epi_reac_violations, epi_bal_p_violations, epi_soft_overflows, epi_hard_overflows, epi_running_cost, epi_renewable_consumption \
                        = self.play_game(
                        temperature=0,
                        temperature_threshold=self.config.temperature_threshold,
                        render=False,
                        rule_agent=rule_agent,
                        start_sample_idx=start_sample_idx[i],
                        train_steps=training_step
                    )
                    test_episode_lengths.append(len(game_history.action_history)-1)
                    test_total_rewards.append(sum(game_history.reward_true_history))
                    test_mean_values.append(np.mean([value for value in game_history.root_values]))
                    voltage_violations += epi_vol_voilations
                    reactive_violations += epi_reac_violations
                    bal_p_violations += epi_bal_p_violations
                    soft_overflows += epi_soft_overflows
                    hard_overflows += epi_hard_overflows
                    running_costs.append(epi_running_cost)
                    renewable_consumption_rate.append(epi_renewable_consumption)

                print(f'test_epi_reward={sum(test_total_rewards) / len(test_total_rewards)}')

                self.shared_storage.set_info.remote(
                    {
                        "test_episode_length": sum(test_episode_lengths) / len(test_episode_lengths),
                        "test_total_reward": sum(test_total_rewards) / len(test_total_rewards),
                        "test_mean_value": sum(test_mean_values) / len(test_mean_values),
                        "test_vol_vio_rate": voltage_violations / sum(test_episode_lengths),
                        "test_reac_vio_rate": reactive_violations / sum(test_episode_lengths),
                        "test_bal_p_vio_rate": bal_p_violations / sum(test_episode_lengths),
                        "test_soft_overflows": soft_overflows / sum(test_episode_lengths),
                        "test_hard_overflows": hard_overflows / sum(test_episode_lengths),
                        "test_running_cost": sum(running_costs) / sum(test_episode_lengths),
                        "test_renewable_consumption": sum(renewable_consumption_rate) / sum(test_episode_lengths),
                    }
                )
                for i, ep in enumerate(start_sample_idx):
                    self.shared_storage.set_info.remote(
                        {
                            f'running_cost_{ep}': running_costs[i]
                        }
                    )
            else:
                time.sleep(5)

        self.close_game()

    @staticmethod
    def select_action(node, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function
        in the config.
        """
        n_child_action = len(node.children)

        visit_counts = np.array(
            [node.children[action_id].visit_count for action_id in node.action_ids], dtype="int32"
        )

        if temperature == 0:
            action = node.actions[node.action_ids[np.argmax(visit_counts)]]

        elif temperature == float("inf"):
            action_id = np.random.choice(n_child_action)
            action = node.actions[action_id]

        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action_id = np.random.choice(n_child_action, p=visit_count_distribution)
            action = node.actions[action_id]

        return action

    def close_game(self):
        self.game.close()

    def play_game(
            self, temperature, temperature_threshold, render, add_exploration_noise=False, record=False,
            rule_agent=None, start_sample_idx=None, train_steps=0
    ):
        """
        Play one game with actions based on the Monte Carlo tree search at each moves.
        """
        voltage_violations, reactive_violations, bal_p_violations, soft_overflows, hard_overflows = 0, 0, 0, 0, 0
        running_costs, renewable_cosumption_rates = [], []
        observation = self.game.reset(ori_obs=True, start_sample_idx=start_sample_idx, seed=self.seed)
        state, ready_mask, closable_mask = get_state_from_obs(observation, settings, self.config.parameters)
        action_high, action_low = get_action_space(observation, self.config.parameters, settings)

        # if self.config.is_plot:
        #     state_mean = np.load('./state_mean.npy')
        #     state_std = np.load('./state_std.npy')
        # else:
        if self.config.norm_type == 'min_max':
            state_min = ray.get(self.shared_storage.get_info.remote("state_min"))
            state_max = ray.get(self.shared_storage.get_info.remote("state_max"))
        else:
            state_mean = ray.get(self.shared_storage.get_info.remote("state_mean"))
            state_std = ray.get(self.shared_storage.get_info.remote("state_std"))

        if self.config.norm_type == 'mean_std':
            state_norm = (state - state_mean) / (state_std + 1e-4)
        else:
            state_norm = (state - state_min) / (state_max - state_min + 1e-4)

        game_history = GameHistory()
        game_history.action_history.append(self.config.sample_random_actions(1).reshape(-1))
        game_history.observation_history.append(state_norm)
        game_history.origin_state_history.append(state)
        game_history.reward_history.append(0)
        game_history.reward_true_history.append(0)
        game_history.ready_mask_history.append(ready_mask)
        game_history.closable_mask_history.append(closable_mask)
        game_history.action_high_history.append(action_high)
        game_history.action_low_history.append(action_low)

        """
            Obs_history:    [Obs0,  Obs1,  Obs2, ...]
            Act_history:    [0,     Act0,  Act1, ...]
            Rew_history:    [0,     Rew0,  Act1, ...]
            Root_history:   [Root0, Root1, Root2, ...]
        """

        done = False

        data = []

        with torch.no_grad():
            while (
                    not done and len(game_history.action_history) <= self.config.max_moves
            ):
                last_observation = copy.deepcopy(observation)
                # Choose the action
                # x = time.time()
                root = MCTS(self.config).run(
                    model=self.model,
                    observation=state_norm.reshape(1, *state.shape),
                    action_high=action_high.reshape(1, *action_high.shape),
                    action_low=action_low.reshape(1, *action_low.shape),
                    ready_mask=ready_mask.reshape(1, *ready_mask.shape),
                    closable_mask=closable_mask.reshape(1, *closable_mask.shape),
                    origin_state=state.reshape(1, *state.shape),
                    add_exploration_noise=add_exploration_noise,
                    is_test=True,
                    train_steps=train_steps
                )
                action = self.select_action(
                    root,
                    temperature
                    # if not temperature_threshold
                    #    or len(game_history.action_history) < temperature_threshold
                    # else 0,
                )
                # print(f'decision time={time.time()-x:.3f}')
                if not self.config.multi_on_off:
                    # action_real = modify_action_v2(action, ready_mask, closable_mask, self.config, action_high, action_low)    # for using MCTS planning
                    action_real = combine_one_hot_action(state, action, ready_mask, closable_mask, self.config, action_high, action_low, is_test=True)    # for using MCTS planning
                else:
                    action_real = modify_action_v3(action, ready_mask, closable_mask, self.config, action_high,
                                                   action_low)  # for using MCTS planning

                # action_true, restart_flag = rule_agent.act(observation)
                # action_n = action_true['adjust_gen_p']

                if not self.config.multi_on_off:
                    renewable_consumptions = []
                    for action_tmp in root.actions.values():
                        action_tmp_real = combine_one_hot_action(state, action_tmp, ready_mask, closable_mask, self.config, action_high, action_low)
                        if self.config.parameters['only_power']:
                            renewable_consumption = sum((np.asarray(observation.gen_p)+action_tmp_real)[settings.renewable_ids])/sum(np.asarray(observation.nextstep_renewable_gen_p_max))
                        else:
                            renewable_consumption = sum((np.asarray(observation.gen_p)+action_tmp_real[:54])[settings.renewable_ids])/sum(np.asarray(observation.nextstep_renewable_gen_p_max))
                        renewable_consumption = ((renewable_consumption*1000)//1)/1000
                        renewable_consumptions.append(renewable_consumption)

                if record:
                    print('Now step', len(game_history.frames))
                    frame = self.game.env.render(mode='rgb_array',
                                                 height=100,
                                                 width=100,
                                                 camera_id=0)
                    game_history.frames.append(frame)
                    game_history.phys_states.append(self.game.env.get_phy_state())
                    game_history.trees.append(root.dump_tree())

                # print_log(observation, settings)
                # action_true, _, open_hot, close_hot = self.traditional_solver.run_opf(observation)
                # action_true = action_true.clip(action_low, action_high)

                observation, reward, done, info = self.game.step(action_real, ori_obs=True)
                self.Q.append([state[0], last_observation.gen_p[17], action_real[:54].sum(),
                               np.where(action[settings.num_gen:2*settings.num_gen]>0)[0],
                               np.where(action[2*settings.num_gen+1:-1])[0]])
                if ((np.asarray(observation.rho) > settings.soft_overflow_bound) * (np.asarray(observation.rho) <= settings.hard_overflow_bound)).any():
                    soft_overflows += 1
                elif (np.asarray(observation.rho) > settings.hard_overflow_bound).any():
                    hard_overflows += 1

                if gen_reactive_power_reward(observation, settings) < 0.0:
                    reactive_violations += 1
                if sub_voltage_reward(observation, settings) < -0.006:
                    voltage_violations += 1
                if balanced_gen_reward(observation, settings) < 0.0:
                    bal_p_violations += 1

                running_costs.append(calc_running_cost_rew(np.asarray(observation.gen_p),
                                                           np.asarray(observation.gen_status),
                                                           np.asarray(last_observation.gen_status), is_real=True))
                renewable_cosumption_rates.append(sum(np.asarray(observation.gen_p)[settings.renewable_ids]) / sum(np.asarray(observation.curstep_renewable_gen_p_max)))


                if len(game_history.action_history) % 2 == 0:
                    print('----------------------------------------------------------')
                    print(f'test_start_idx={start_sample_idx}, step={len(game_history.action_history)}, reward={reward:.3f}, value={root.value():.3f}, '
                          f'line_outage={len(observation.line_status) - sum(np.array(observation.line_status).astype(np.float32))}')
                    # print(f'test_mse_loss = {((action-action_n)**2).mean()}')
                    # print(f'test_mae_loss = {np.abs(action-action_n).mean()}')
                    print(
                        f'penetration_rate={sum(np.asarray(observation.gen_p)[settings.renewable_ids]) / sum(np.asarray(observation.curstep_renewable_gen_p_max)):.3f}')
                    print(f'root_sampled_actions_renewable_rates={renewable_consumptions}')
                    print(f'root_q_inits={[child.q_init for child in root.children.values()]}')
                    print(f'root_policy_sigma={root.policy_sigma}')
                    print(f'closed_gen_num={54 - sum(observation.gen_status)}')
                    print(f'closable_gen_num={sum(closable_mask)}')
                    print(f'root visits={[child.visit_count for child in root.children.values()]}')
                    print(f'root_values={[child.value() for child in root.children.values()]}')
                    print(
                        f'root_last_ucb_value_score={[child.last_ucb_value_score for child in root.children.values()]}')
                    print(
                        f'root_last_ucb_prior_score={[child.last_ucb_prior_score for child in root.children.values()]}')
                    print(f'root_rewards={[child.reward for child in root.children.values()]}')
                    print(
                        f'root_action_std={np.array([action for action in root.actions.values()]).std(axis=0).mean()}')
                    print(
                        f'line_over_flow_rew={line_over_flow_reward(observation, settings):.3f}, renewable_consumption_rew={renewable_consumption_reward(observation, settings):.3f}, '
                        f'bal_gen_rew={balanced_gen_reward(observation, settings):.3f}, '
                        f'gen_reactive_rew={gen_reactive_power_reward(observation, settings):.3f}, sub_voltage_rew={sub_voltage_reward(observation, settings):.3f}, '
                        f'running_cost_rew={running_cost_reward(observation, last_observation, settings):.3f},'
                        f'gen_reac_rew_v2={gen_reactive_power_reward_v2(observation, settings):.3f},'
                        f'bal_gen_rew_v2={balanced_gen_reward_v2(observation, settings):.3f}')
                    print_log(observation, settings)
                    print('**********************************************************')

                if done:
                    print(info)
                    steps = len(game_history.action_history)
                    while len(self.Q) > 0:
                        data1 = self.Q.pop()
                        print(f'step={steps}, delta_load={data1[0]:.3f}, bal_gen_p={data1[1]:.3f}, action_p={data1[2]:.3f}, open_id={data1[3]}, close_id={data1[4]}')
                        steps -= 1
                    # import ipdb
                    # ipdb.set_trace()

                state, ready_mask, closable_mask = get_state_from_obs(observation, settings, self.config.parameters)
                # print(f'closable_gen_num={sum(closable_mask)}')
                open_ids = np.where(observation.gen_status > 0)[0].tolist()
                close_ids = np.where(observation.gen_status == 0)[0].tolist()
                open_action_high, open_action_low = 0, 0
                for i in range(settings.num_gen):
                    if i in open_ids and i != settings.balanced_id:
                        open_action_high += (observation.gen_p[i] + observation.action_space['adjust_gen_p'].high[i])
                        if i in settings.thermal_ids:
                            open_action_low += max(settings.min_gen_p[i], (
                                        observation.gen_p[i] + observation.action_space['adjust_gen_p'].low[i]))
                        else:
                            open_action_low += observation.gen_p[i] + observation.action_space['adjust_gen_p'].high[i]
                    if i == settings.balanced_id:
                        open_action_high += settings.max_gen_p[i]
                        open_action_low += settings.min_gen_p[i]

                data.append((sum(observation.load_p), sum(np.asarray(observation.gen_p)[settings.renewable_ids]),
                             sum(observation.curstep_renewable_gen_p_max), 54 - sum(observation.gen_status), sum(closable_mask),
                             open_action_high, open_action_low, list(observation.gen_status)))
                action_high, action_low = get_action_space(observation, self.config.parameters, settings)
                reward_clipped = reward_shaping_func(reward, self.config.reward_delta, self.config.reward_amp)

                if self.config.norm_type == 'mean_std':
                    state_norm = (state - state_mean) / (state_std + 1e-4)
                else:
                    state_norm = (state - state_min) / (state_max - state_min + 1e-4)

                game_history.store_search_statistics(root)
                # Next batch
                game_history.action_history.append(action)
                game_history.observation_history.append(state_norm)
                game_history.origin_state_history.append(state)
                game_history.reward_true_history.append(reward)
                # game_history.reward_history.append(reward)
                game_history.reward_history.append(reward_clipped)
                game_history.ready_mask_history.append(ready_mask)
                game_history.closable_mask_history.append(closable_mask)
                game_history.action_high_history.append(action_high)
                game_history.action_low_history.append(action_low)

        if self.config.is_plot:
            headers = ('load', 'actual', 'max', 'closed_num', 'closable_num', 'open_action_high', 'open_action_low', 'gen_status')
            path = os.path.join(self.config.results_path, f'data_{start_sample_idx}.csv')
            with open(path, 'w', encoding='utf-8', newline='') as f:
                write = csv.writer(f)
                write.writerow(headers)
                write.writerows(data)

        return game_history, voltage_violations, reactive_violations, bal_p_violations, soft_overflows, hard_overflows, \
                sum(running_costs), sum(renewable_cosumption_rates)
               # sum(running_costs)/len(running_costs), sum(renewable_cosumption_rates)/len(renewable_cosumption_rates)