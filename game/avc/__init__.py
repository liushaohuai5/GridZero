import copy

import pandapower as pp
import numpy as np
import pandapower.networks as pn
import pandapower.converter as pc
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
from pypower.api import runpf

class Environment:

    def __init__(self, is_test=False):
        self.net = pn.case33bw()
        pp.runpp(self.net)
        self.ori_voltages = copy.deepcopy(self.net.res_bus.vm_pu.values)
        self.gen_to_bus = {0: 9, 1: 14, 2: 21, 3: 23, 4: 26, 5: 29}
        self.load_to_bus = {0: 6, 1: 7, 2: 11, 3: 16, 4: 18, 5: 24, 6: 27, 7: 29}
        # self.agent_buses = {0: [8, 9, 10, 11, 12], 1: [13, 14, 15, 16, 17], 2: [0, 1, 18, 19, 20, 21],
        #                     3: [2, 3, 4, 22, 23, 24], 4: [5, 6, 7, 25, 26, 27], 5: [28, 29, 30, 31, 32]}
        # self.agent_loads = {0: 2, 1: 3, 2: 4, 3: 5, 4: [0, 1, 6], 5: 7}
        # self.agent_gens = self.gen_to_bus
        self.segment = 288 * 7
        self.load_p_data = self.normalize(pd.read_csv('/workspace/AdversarialGridZero/data/load_p.csv').values[:self.segment, list(self.load_to_bus.values())]) #/ 5
        self.load_q_data = self.normalize(pd.read_csv('/workspace/AdversarialGridZero/data/load_q.csv').values[:self.segment, list(self.load_to_bus.values())]) #/ 5
        self.max_renewable_p_data = self.normalize(pd.read_csv('/workspace/AdversarialGridZero/data/max_renewable_gen_p.csv').values[:self.segment, :]) #/ 5

        for i in range(self.load_p_data.shape[1]):
            self.load_p_data[:, i] = scipy.signal.savgol_filter(self.load_p_data[:, i], 53, 3)
        for i in range(self.load_q_data.shape[1]):
            self.load_q_data[:, i] = scipy.signal.savgol_filter(self.load_q_data[:, i], 53, 3)
        for i in range(self.max_renewable_p_data.shape[1]):
            self.max_renewable_p_data[:, i] = scipy.signal.savgol_filter(self.max_renewable_p_data[:, i], 53, 3)

        # plt.subplot(3, 1, 1)
        # plt.plot(self.load_p_data[:, 1], label='load_p')
        # plt.subplot(3, 1, 2)
        # plt.plot(self.load_q_data[:, 1], label='load_q')
        # plt.subplot(3, 1, 3)
        # plt.plot(self.max_renewable_p_data[:, 1], label='renewable_p')
        # plt.show()
        # import ipdb
        # ipdb.set_trace()
        self.init_net()
        self.len = self.load_p_data.shape[0]
        self.idx = 0

        self.q_max = 1.0
        self.q_min = -1.0
        self.is_test = is_test
        self.agent_num = len(self.agent_gens)

    def normalize(self, data):
        return (data - data.min(0)) / (data.max(0) - data.min(0))
        # return (data - data.mean(0)) / data.std(0)
        # return data

    def init_net(self):
        self.net.clear()
        self.net = pn.case33bw()
        self.net.ext_grid.max_p_mw[0] = 200
        self.net.ext_grid.max_q_mvar[0] = 180
        self.net.ext_grid.min_q_mvar[0] = -180
        self.init_gen()
        self.init_load()
        # self.init_line()
        self.init_bus()

    def init_bus(self):
        for bus in range(len(self.net.bus)):
            self.net.bus.max_vm_pu[bus] = 1.5
            self.net.bus.min_vm_pu[bus] = 0.5

    def init_line(self):
        for i in self.net.line.in_service.keys():
            if not self.net.line.in_service[i]:
                self.net.line.in_service[i] = True

    def init_gen(self):
        for i in self.gen_to_bus.keys():
            bus = self.gen_to_bus[i]
            pp.create.create_sgen(self.net, bus, self.max_renewable_p_data[0, i])

    def init_load(self):
        ori_ids = self.net.load.p_mw.keys().tolist()
        for i in ori_ids:
            self.net.load.p_mw[i] = 0.0
            self.net.load.q_mvar[i] = 0.0
        # for i in self.load_to_bus.keys():
        #     bus = self.load_to_bus[i]
        #     pp.create.create_load(self.net, bus, self.load_p_data[0, i], self.load_q_data[0, i])

    def reset(self, idx=None):
        if idx is None:
            idx = np.random.randint(0, self.len - 288)
            self.idx = idx
            self.start_idx = idx
        self.init_net()
        self.net.load.p_mw[list(self.load_to_bus.values())] = np.maximum(self.load_p_data[idx, list(self.load_to_bus.keys())], 0)
        self.net.load.q_mvar[list(self.load_to_bus.values())] = self.load_q_data[idx, list(self.load_to_bus.keys())]
        self.net.sgen.p_mw[list(self.gen_to_bus.keys())] = np.maximum(self.max_renewable_p_data[idx, list(self.gen_to_bus.keys())], 0)
        # self.net.sgen.p_mw[list(self.gen_to_bus.keys())] = self.load_p_data[idx, :].sum()
        # self.net.sgen.q_mvar[list(self.gen_to_bus.keys())] = sum(self.load_q_data[idx, list(self.load_to_bus.keys())])
        pp.runpp(self.net)
        obs, _ = self.form_obs_reward(False)

        self.ori_voltages = copy.deepcopy(self.net.res_bus.vm_pu.values)
        self.first_obs = copy.deepcopy(obs)
        return obs

    def step(self, action):
        self.adjust_net(action)

        try:
            pp.runpp(self.net,
                     # init='results'
                     init_vm_pu=self.net.res_bus.vm_pu,
                     init_va_degree=self.net.res_bus.va_degree
                     )
            done = False
            info = {}
        except:
            done = True
            info = 'power flow not converged'

        try:
            self.read_next_snapshot()
        except:
            info = 'sample idx out of bound'
            done = True

        next_obs, reward = self.form_obs_reward(done)

        # print(f'gen_p={self.net.res_sgen.p_mw.values.sum():.3f}, load_p={self.net.res_load.p_mw.values.sum():.3f}, ext_p={self.net.res_ext_grid.p_mw.values.sum():.3f}')
        # print(f'gen_q={self.net.res_sgen.q_mvar.values.sum():.3f}, load_q={self.net.res_load.q_mvar.values.sum():.3f}, ext_q={self.net.res_ext_grid.q_mvar.values.sum():.3f}')
        return next_obs, reward, done, info

    def adjust_net(self, action):
        # self.net.sgen.q_mvar[list(self.gen_to_bus.keys())] += (action + 1) / 2 * (self.q_max - self.q_min) + self.q_min
        self.net.sgen.q_mvar[list(self.gen_to_bus.keys())] = (action + 1) / 2 * (self.q_max - self.q_min) + self.q_min

    def read_next_snapshot(self):
        self.idx += 1
        self.net.sgen.p_mw[list(self.gen_to_bus.keys())] = self.max_renewable_p_data[self.idx, list(self.gen_to_bus.keys())]
        # self.net.sgen.p_mw[list(self.gen_to_bus.keys())] = self.load_p_data[self.idx, :].sum() / 6
        self.net.load.p_mw[list(self.load_to_bus.keys())] = self.load_p_data[self.idx, list(self.load_to_bus.keys())]
        self.net.load.q_mvar[list(self.load_to_bus.keys())] = self.load_q_data[self.idx, list(self.load_to_bus.keys())]

    def calc_reward(self, voltages):
        rewards = []
        for buses in self.agent_buses:
            reward = -np.abs(voltages[buses] - self.ori_voltages[buses]).sum()
            # reward = -np.abs(voltages[buses] - 1).sum()
            rewards.append(reward)
            # reward = -np.abs(voltages - self.ori_voltages).sum()
        return np.asarray(rewards)

    def form_obs_reward(self, done):
        v = self.net.res_bus.vm_pu.values
        gen_q = self.net.sgen.q_mvar.values
        load_q = self.net.load.q_mvar[list(self.load_to_bus.keys())].values
        gen_p = self.net.sgen.p_mw.values
        load_p = self.net.load.p_mw[list(self.load_to_bus.keys())].values

        # obs = []
        # for gens, loads, buses in zip(self.agent_gens.keys(), self.agent_loads.values(), self.agent_buses.values()):
        #     q_gen = gen_q[gens] if len(np.array(gen_q[gens]).shape) == 1 else np.array([gen_q[gens]])
        #     q_load = load_q[loads] if len(np.array(load_q[loads]).shape) == 1 else np.array([load_q[loads]])
        #     p_gen = gen_p[gens] if len(np.array(gen_p[gens]).shape) == 1 else np.array([gen_p[gens]])
        #     p_load = load_p[loads] if len(np.array(load_p[loads]).shape) == 1 else np.array([load_p[loads]])
        #     m_obs = np.concatenate((q_gen, q_load, p_gen, p_load, v[buses]))
        #     obs.append(m_obs)
        obs = np.asarray([gen_p, load_p, gen_q, load_q, v])

        if not done:
            reward = self.calc_reward(v)
        else:
            obs = copy.deepcopy(self.first_obs)
            reward = np.asarray([0.0 for _ in range(self.agent_num)])

        return obs, reward


def make_gridsim():
    env = Environment()
    return env