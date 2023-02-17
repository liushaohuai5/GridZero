from game.gridsim.utils import voltage_action, get_state_from_obs, form_action, get_action_space
from utilize.settings import settings
import numpy as np
import copy
from Reward.rewards import self_reward

class GridSimWrapper:
    def __init__(self, env, rule_agent=None, reward_func=None, config=None):

        self.env = env
        self.reward_func = reward_func
        # print(reward_func)
        self.config = config
        self.parameters = self.config.parameters
        self.rule_agent = rule_agent

        # obs = self.env.reset()
        # action_dim_p = obs.action_space['adjust_gen_p'].shape[0]
        # action_dim_v = obs.action_space['adjust_gen_v'].shape[0]
        # self.parameters['action_dim'] = action_dim_p+1 if self.parameters['only_power'] else (action_dim_p + action_dim_v+1)

        self.last_obs = None
        self.step_cnt = 0

    def step(self, action, ori_obs=False):
        illegal_type = 0    # 0 - legal, 1 - balance out of range, 2 - grid is not converged
        # state, ready_thermal_mask, closable_thermal_mask = get_state_from_obs(self.last_obs, settings, self.parameters)
        if self.parameters['only_power']:
            action_high, action_low = get_action_space(self.last_obs, self.parameters, settings)
            # import ipdb
            # ipdb.set_trace()
            # delta_load_p = np.asarray(self.last_obs.nextstep_load_p) - np.asarray(self.last_obs.load_p)
            # action = self.check_balance(action, delta_load_p, action_high, action_low, ready_thermal_mask, closable_thermal_mask)
            action_v = voltage_action(self.last_obs, settings, type=self.parameters['voltage_action_type'])
            # print(action_v)
            action = form_action(action, action_v)
        else:
            action = form_action(action[:len(action)//2], action[len(action)//2:])

        # snapshot = self.get_snapshot()
        # _, _, _, done, info = self.get_results(snapshot, action)
        # if done:
        #     action, _ = self.rule_agent.act(self.last_obs)
        #     if info is 'grid is not converged':
        #         illegal_type = 2
        #     elif info is 'balance gen out of bound':
        #         illegal_type = 1

        observation, reward, done, info = self.env.step(action)
        # snapshot, state, reward, done, info = self.get_results(snapshot, action)
        # observation, _ = snapshot
        # if illegal_type == 1:
        #     reward -= 1
        # elif illegal_type == 2:
        #     reward -= 2
        self.step_cnt += 1
        # if done:
        #     import ipdb
        #     ipdb.set_trace()
        # print(self.step_cnt, done)
        if self.reward_func == 'self_reward':
            # print('****************** self_reward ******************')
            reward = self_reward(observation, self.last_obs, self.config, settings)
        self.last_obs = copy.deepcopy(observation)

        if not ori_obs:
            state, ready_thermal_mask, closable_thermal_mask = get_state_from_obs(observation, settings, self.parameters)
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

        # sub_snap, _ = snapshot

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
        # if self.reward_func is not None:
        #     self.reward_func.reset(0)

        if not ori_obs:
            state, ready_thermal_mask, closable_thermal_mask = get_state_from_obs(observation, settings, self.parameters)
            state = np.asarray(state, dtype="float32")
            return state
        else:
            return observation

    def close(self):
        self.env.close()
