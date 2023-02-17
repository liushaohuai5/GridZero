from game.gridsim.utils import voltage_action, get_state_from_obs, form_action, get_action_space
from utilize.settings import settings
import numpy as np
import copy
from Reward.rewards import self_reward

class GridSimWrapper:
    def __init__(self, env, rule_agent=None, reward_func=None, config=None):

        self.env = env
        self.reward_func = reward_func
        self.config = config
        self.parameters = self.config.parameters
        self.rule_agent = rule_agent

        self.last_obs = None
        self.step_cnt = 0

    def step(self, action, ori_obs=False):
        if self.parameters['only_power']:
            action_v = voltage_action(self.last_obs, settings, type=self.parameters['voltage_action_type'])
            action = form_action(action, action_v)
        else:
            action = form_action(action[:len(action)//2], action[len(action)//2:])

        observation, reward, done, info = self.env.step(action)
        self.step_cnt += 1
        if self.reward_func == 'self_reward':
            reward = self_reward(observation, self.last_obs, self.config, settings)
        self.last_obs = copy.deepcopy(observation)

        if not ori_obs:
            state, ready_thermal_mask, closable_thermal_mask = get_state_from_obs(observation, settings, self.parameters)
            state = np.asarray(state, dtype="float32")
            return state, reward, done, info
        else:
            return observation, reward, done, info

    def step_only_attack(self, action_one_hot, ori_obs=False):
        action = np.where(action_one_hot > 0)[0][0]
        if action == len(action_one_hot) - 1:
            action = None
        # assert action < settings.num_line, "attack on a nonexistent line, attack should range from 0 to line_num-1"
        observation, reward, done, info = self.env.step_only_attack(action)
        if self.reward_func == 'self_reward':
            reward = self_reward(observation, self.last_obs, self.config, settings)
        self.last_obs = copy.deepcopy(observation)

        if not ori_obs:
            state, ready_thermal_mask, closable_thermal_mask = get_state_from_obs(observation, settings,
                                                                                  self.parameters)
            state = np.asarray(state, dtype="float32")
            return state, reward, done, info
        else:
            return observation, reward, done, info


    def get_results(self, snapshot, action):
        if len(action) > 2:
            if self.parameters['only_power']:
                action_v = np.zeros_like(action)
                action = form_action(action, action_v)
            else:
                action = form_action(action[:len(action)//2], action[len(action)//2:])

        new_snapshot, reward, done, info = self.env.get_results(snapshot, action)
        new_obs, sample_idx = new_snapshot

        state, ready_thermal_mask, closable_thermal_mask = get_state_from_obs(new_obs, settings, self.parameters)
        state = np.asarray(state, dtype="float32")
        return new_snapshot, state, reward, done, info

    def get_snapshot(self):
        return self.env.get_snapshot()

    def reset_snapshot(self):
        observation = self.env.reset()
        sample_idx = self.env.sample_idx
        state, ready_thermal_mask, closable_thermal_mask = get_state_from_obs(observation, settings, self.parameters)
        return (observation, sample_idx), state

    def reset(self, ori_obs=False, start_sample_idx=None, seed=0):
        self.step_cnt = 0
        observation = self.env.reset(start_sample_idx=start_sample_idx)
        self.last_obs = copy.deepcopy(observation)

        if not ori_obs:
            state, ready_thermal_mask, closable_thermal_mask = get_state_from_obs(observation, settings, self.parameters)
            state = np.asarray(state, dtype="float32")
            return state
        else:
            return observation

    def close(self):
        self.env.close()
