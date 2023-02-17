import sys
sys.path.append('/workspace/RobotEZero')
from Observation.observation import Observation
from Reward.rewards import *
from utilize.read_forecast_value import ForecastReader
from utilize.line_cutting import Disconnect
from utilize.action_space import ActionSpace
from utilize.legal_action import *
import example
import copy
import numpy as np
import warnings
from ori_opf_SG126 import rerun_opf
warnings.filterwarnings('ignore')

class Environment:
    def __init__(self, settings, reward_type="EPRIReward", is_test=False):
        self.settings = copy.deepcopy(settings)
        self.forecast_reader = ForecastReader(self.settings, is_test=is_test)
        self.reward_type = reward_type
        self.done = True
        self.action_space_cls = ActionSpace(settings)
        self.is_test = is_test

    def reset_attr(self):
        # Reset attr in the base env
        self.grid = example.Print()
        self.done = False
        self.timestep = 0
        self.gen_status = np.ones(settings.num_gen)
        self.steps_to_recover_gen = np.zeros(settings.num_gen, dtype=int)
        self.steps_to_close_gen = np.zeros(settings.num_gen, dtype=int)
        self.steps_to_reconnect_line = np.zeros(settings.num_line, dtype=int)
        self.count_soft_overflow_steps = np.zeros(settings.num_line, dtype=int)

    def reset(self, seed=None, start_sample_idx=None):
        self.reset_attr()
        settings = self.settings
        grid = self.grid

        # Instead of using `np.random` use `self.np_random`.
        # It won't be affected when user using `np.random`.
        self.np_random = np.random.RandomState()
        if seed is not None:
            self.np_random.seed(seed=seed)
            np.random.seed(seed=seed)

        self.disconnect = Disconnect(self.np_random, self.settings)

        if start_sample_idx is not None:
            self.sample_idx = start_sample_idx
        else:
            self.sample_idx = self.np_random.randint(0, settings.num_sample)
            # print(f'episode start idx={self.sample_idx}')
        assert settings.num_sample > self.sample_idx >= 0

        # Read self.sample_idx timestep data
        """
        NOTE:
            1. C++ read the data by the row number of the csv file;
            2. The first row of the csv file is the header.
        """
        row_idx = self.sample_idx + 1 
        grid.readdata(row_idx,
                      settings.load_p_filepath if not self.is_test else settings.test_load_p_filepath,
                      settings.load_q_filepath if not self.is_test else settings.test_load_q_filepath,
                      settings.gen_p_filepath if not self.is_test else settings.test_gen_p_filepath,
                      settings.gen_q_filepath if not self.is_test else settings.test_gen_q_filepath
                      )

        injection_gen_p = self._round_p(grid.itime_unp[0])
        grid.env_feedback(settings.name_index, injection_gen_p, [], row_idx, [])
        rounded_gen_p = self._round_p(grid.prod_p[0])

        self._update_gen_status(injection_gen_p)
        try:
            self._check_gen_status(injection_gen_p, rounded_gen_p)
        except:
            print('-------------------- Recalculate Gen P -------------------------')
            rho = self._calc_rho(grid, settings)
            # Update forecast value
            curstep_renewable_gen_p_max, nextstep_renewable_gen_p_max = \
                self.forecast_reader.read_step_renewable_gen_p_max(self.sample_idx)
            nextstep_load_p = self.forecast_reader.read_step_load_p(self.sample_idx)

            # action_space = self.action_space_cls.update(grid, self.steps_to_recover_gen, self.steps_to_close_gen,
            #                                             rounded_gen_p, nextstep_renewable_gen_p_max)
            injection_gen_p, recover_ids, close_ids = rerun_opf(
                Observation(
                    grid=grid, timestep=self.timestep, action_space=None,
                    steps_to_reconnect_line=self.steps_to_reconnect_line,
                    count_soft_overflow_steps=self.count_soft_overflow_steps, rho=rho,
                    gen_status=self.gen_status, steps_to_recover_gen=self.steps_to_recover_gen,
                    steps_to_close_gen=self.steps_to_close_gen,
                    curstep_renewable_gen_p_max=curstep_renewable_gen_p_max,
                    nextstep_renewable_gen_p_max=nextstep_renewable_gen_p_max,
                    rounded_gen_p=rounded_gen_p,
                    nextstep_load_p=nextstep_load_p
                )
            )
            injection_gen_p = self._round_p(injection_gen_p)
            grid.env_feedback(settings.name_index, injection_gen_p, [], row_idx, [])
            rounded_gen_p = self._round_p(grid.prod_p[0])
            for i in recover_ids:
                self.gen_status[i] = 1
            for i in close_ids:
                self.gen_status[i] = 0
                # self.steps_to_recover_gen[i] = 39
                self.steps_to_recover_gen[i] = settings.max_steps_to_recover_gen[i]


        self.last_injection_gen_p = copy.deepcopy(injection_gen_p)

        rho = self._calc_rho(grid, settings)

        # Update forecast value
        curstep_renewable_gen_p_max, nextstep_renewable_gen_p_max = \
            self.forecast_reader.read_step_renewable_gen_p_max(self.sample_idx)
        nextstep_load_p = self.forecast_reader.read_step_load_p(self.sample_idx)
        action_space = self.action_space_cls.update(grid, self.steps_to_recover_gen, self.steps_to_close_gen,
                                                     rounded_gen_p, nextstep_renewable_gen_p_max)

        self.obs = Observation(
            grid=grid, timestep=self.timestep, action_space=action_space,
            steps_to_reconnect_line=self.steps_to_reconnect_line,
            count_soft_overflow_steps=self.count_soft_overflow_steps, rho=rho,
            gen_status=self.gen_status, steps_to_recover_gen=self.steps_to_recover_gen,
            steps_to_close_gen=self.steps_to_close_gen,
            curstep_renewable_gen_p_max=curstep_renewable_gen_p_max,
            nextstep_renewable_gen_p_max=nextstep_renewable_gen_p_max,
            rounded_gen_p=rounded_gen_p,
            nextstep_load_p=nextstep_load_p
        )
        return copy.deepcopy(self.obs)

    def get_results(self, snapshot, act):
        if self.done:
            raise Exception("The env is game over, please reset.")
        settings = self.settings
        last_obs, sample_idx = snapshot
        timestep = last_obs.timestep
        grid = self.grid

        self._check_action(act)
        act['adjust_gen_p'] = self._round_p(act['adjust_gen_p'])

        # Compute the injection value
        adjust_gen_p = act['adjust_gen_p']
        injection_gen_p = [adjust_gen_p[i] + last_obs.gen_p[i] for i in range(len(adjust_gen_p))]
        injection_gen_p = self._round_p(injection_gen_p)

        adjust_gen_v = act['adjust_gen_v']
        injection_gen_v = [adjust_gen_v[i] + last_obs.gen_v[i] for i in range(len(adjust_gen_v))]

        # Judge the legality of the action
        legal_flag, fail_info = is_legal(act, last_obs, settings)
        if not legal_flag:
            done = True
            new_obs, reward, done, info = self.return_res(fail_info, done)
            return (new_obs, sample_idx), reward, done, info

        disc_name, steps_to_reconnect_line, count_soft_overflow_steps = self.disconnect.get_disc_name(
            last_obs)

        sample_idx += 1
        timestep += 1

        """
        NOTE:
            1. C++ read the data by the row number of the csv file;
            2. The first row of the csv file is the header.
        """
        row_idx = sample_idx + 1
        # Read the power data of the next step from .csv file
        grid.readdata(row_idx,
                      settings.load_p_filepath if not self.is_test else settings.test_load_p_filepath,
                      settings.load_q_filepath if not self.is_test else settings.test_load_q_filepath,
                      settings.gen_p_filepath if not self.is_test else settings.test_gen_p_filepath,
                      settings.gen_q_filepath if not self.is_test else settings.test_gen_q_filepath
                      )

        injection_gen_p = self._injection_auto_mapping(injection_gen_p)

        # Update generator running status
        gen_status, steps_to_recover_gen, steps_to_close_gen = self.__update_gen_status(injection_gen_p,
                                                                                        last_obs.gen_status,
                                                                                        last_obs.steps_to_recover_gen,
                                                                                        last_obs.steps_to_close_gen)

        # Power flow calculation
        grid.env_feedback(grid.un_nameindex, injection_gen_p, injection_gen_v, row_idx, disc_name)

        flag, info = self.check_done(grid, settings)
        if flag:
            done = True
            new_obs, reward, done, info = self.return_res(fail_info, done)
            return (new_obs, sample_idx), reward, done, info

        rounded_gen_p = self._round_p(grid.prod_p[0])
        # print(np.asarray(rounded_gen_p)-np.asarray(injection_gen_p))
        # import ipdb
        # ipdb.set_trace()
        self.__check_gen_status(injection_gen_p, rounded_gen_p, gen_status)
        self.last_injection_gen_p = copy.deepcopy(injection_gen_p)

        # Update forecast value
        curstep_renewable_gen_p_max, nextstep_renewable_gen_p_max = \
            self.forecast_reader.read_step_renewable_gen_p_max(sample_idx)
        nextstep_load_p = self.forecast_reader.read_step_load_p(sample_idx)

        action_space = self.action_space_cls.update(grid, steps_to_recover_gen, steps_to_close_gen,
                                                    rounded_gen_p, nextstep_renewable_gen_p_max)

        rho = self._calc_rho(grid, settings)

        # pack obs
        self.obs = Observation(
            grid=grid, timestep=timestep, action_space=action_space,
            steps_to_reconnect_line=steps_to_reconnect_line,
            count_soft_overflow_steps=count_soft_overflow_steps, rho=rho,
            gen_status=gen_status, steps_to_recover_gen=steps_to_recover_gen,
            steps_to_close_gen=steps_to_close_gen,
            curstep_renewable_gen_p_max=curstep_renewable_gen_p_max,
            nextstep_renewable_gen_p_max=nextstep_renewable_gen_p_max,
            rounded_gen_p=rounded_gen_p,
            nextstep_load_p=nextstep_load_p
        )

        self.reward = self.get_reward(self.obs, last_obs)
        new_obs, reward, done, info = self.return_res()
        new_snapshot = (new_obs, sample_idx,
                        # timestep, steps_to_reconnect_line, count_soft_overflow_steps, gen_status,
                        # steps_to_recover_gen, steps_to_close_gen
                        )
        return new_snapshot, reward, done, info

    def step_only_attack(self, act):
        # print(f'sample_idx={self.sample_idx}')
        if self.done:
            raise Exception("The env is game over, please reset.")
        settings = self.settings
        last_obs = self.obs
        grid = self.grid

        # self._check_action(act)
        # act['adjust_gen_p'] = self._round_p(act['adjust_gen_p'])

        # Compute the injection value
        # adjust_gen_p = act['adjust_gen_p']
        # injection_gen_p = [adjust_gen_p[i] + last_obs.gen_p[i] for i in range(len(adjust_gen_p))]
        # injection_gen_p = self._round_p(injection_gen_p)
        injection_gen_p = [gen_p for gen_p in last_obs.gen_p]

        # adjust_gen_v = act['adjust_gen_v']
        # injection_gen_v = [adjust_gen_v[i] + last_obs.gen_v[i] for i in range(len(adjust_gen_v))]
        injection_gen_v = [gen_v for gen_v in last_obs.gen_v]

        # Judge the legality of the action
        # legal_flag, fail_info = is_legal(act, last_obs, settings)
        # if not legal_flag:
        #     self.done = True
        #     return self.return_res(fail_info, self.done)

        disc_name, self.steps_to_reconnect_line, self.count_soft_overflow_steps = self.disconnect.get_disc_name(
            last_obs)
        attack_lnname = settings.white_list_random_disconnection[act]
        disc_name.append(attack_lnname)
        attack_idx = settings.lnname.index(attack_lnname)
        self.steps_to_reconnect_line[attack_idx] = settings.max_steps_to_reconnect_line

        # self.sample_idx += 1
        # self.timestep += 1

        """
        NOTE:
            1. C++ read the data by the row number of the csv file;
            2. The first row of the csv file is the header.
        """
        row_idx = self.sample_idx + 1
        # Read the power data of the next step from .csv file
        # if self.is_test:
        #     print(settings.test_gen_q_filepath)
        grid.readdata(row_idx,
                      settings.load_p_filepath if not self.is_test else settings.test_load_p_filepath,
                      settings.load_q_filepath if not self.is_test else settings.test_load_q_filepath,
                      settings.gen_p_filepath if not self.is_test else settings.test_gen_p_filepath,
                      settings.gen_q_filepath if not self.is_test else settings.test_gen_q_filepath
                      )

        injection_gen_p = self._injection_auto_mapping(injection_gen_p)
        # injection_gen_p = self._injection_auto_mapping_v2(injection_gen_p)

        # Update generator running status
        # self._update_gen_status(injection_gen_p)

        # Power flow calculation
        grid.env_feedback(grid.un_nameindex, injection_gen_p, injection_gen_v, row_idx, disc_name)

        flag, info = self.check_done(grid, settings)
        if flag:
            self.done = True
            return self.return_res(info, self.done)

        rounded_gen_p = self._round_p(grid.prod_p[0])

        self._check_gen_status(injection_gen_p, rounded_gen_p)
        self.last_injection_gen_p = copy.deepcopy(injection_gen_p)

        # Update forecast value
        curstep_renewable_gen_p_max, nextstep_renewable_gen_p_max = \
            self.forecast_reader.read_step_renewable_gen_p_max(self.sample_idx)
        nextstep_load_p = self.forecast_reader.read_step_load_p(self.sample_idx)

        action_space = self.action_space_cls.update(grid, self.steps_to_recover_gen, self.steps_to_close_gen,
                                                     rounded_gen_p, nextstep_renewable_gen_p_max)

        rho = self._calc_rho(grid, settings)

        # pack obs
        self.obs = Observation(
            grid=grid, timestep=self.timestep, action_space=action_space,
            steps_to_reconnect_line=self.steps_to_reconnect_line,
            count_soft_overflow_steps=self.count_soft_overflow_steps, rho=rho,
            gen_status=self.gen_status, steps_to_recover_gen=self.steps_to_recover_gen,
            steps_to_close_gen=self.steps_to_close_gen,
            curstep_renewable_gen_p_max=curstep_renewable_gen_p_max,
            nextstep_renewable_gen_p_max=nextstep_renewable_gen_p_max,
            rounded_gen_p=rounded_gen_p,
            nextstep_load_p=nextstep_load_p
        )

        self.reward = -self.get_reward(self.obs, last_obs)
        return self.return_res()

    def step(self, act):
        # print(f'sample_idx={self.sample_idx}')
        if self.done:
            raise Exception("The env is game over, please reset.")
        settings = self.settings
        last_obs = self.obs
        grid = self.grid

        self._check_action(act)
        act['adjust_gen_p'] = self._round_p(act['adjust_gen_p'])

        # Compute the injection value
        adjust_gen_p = act['adjust_gen_p']
        injection_gen_p = [adjust_gen_p[i] + last_obs.gen_p[i] for i in range(len(adjust_gen_p))]
        injection_gen_p = self._round_p(injection_gen_p)

        adjust_gen_v = act['adjust_gen_v']
        injection_gen_v = [adjust_gen_v[i] + last_obs.gen_v[i] for i in range(len(adjust_gen_v))]

        # Judge the legality of the action
        legal_flag, fail_info = is_legal(act, last_obs, settings)
        if not legal_flag:
            self.done = True
            return self.return_res(fail_info, self.done)

        disc_name, self.steps_to_reconnect_line, self.count_soft_overflow_steps = self.disconnect.get_disc_name(
            last_obs)
        # disc_name = []

        self.sample_idx += 1
        self.timestep += 1

        """
        NOTE:
            1. C++ read the data by the row number of the csv file;
            2. The first row of the csv file is the header.
        """
        row_idx = self.sample_idx + 1
        # Read the power data of the next step from .csv file
        # if self.is_test:
        #     print(settings.test_gen_q_filepath)
        grid.readdata(row_idx,
                      settings.load_p_filepath if not self.is_test else settings.test_load_p_filepath,
                      settings.load_q_filepath if not self.is_test else settings.test_load_q_filepath,
                      settings.gen_p_filepath if not self.is_test else settings.test_gen_p_filepath,
                      settings.gen_q_filepath if not self.is_test else settings.test_gen_q_filepath
                      )

        injection_gen_p = self._injection_auto_mapping(injection_gen_p)
        # injection_gen_p = self._injection_auto_mapping_v2(injection_gen_p)

        # Update generator running status
        self._update_gen_status(injection_gen_p)

        # Power flow calculation
        grid.env_feedback(grid.un_nameindex, injection_gen_p, injection_gen_v, row_idx, disc_name)

        flag, info = self.check_done(grid, settings)
        if flag:
            self.done = True
            return self.return_res(info, self.done)

        rounded_gen_p = self._round_p(grid.prod_p[0])

        self._check_gen_status(injection_gen_p, rounded_gen_p)
        self.last_injection_gen_p = copy.deepcopy(injection_gen_p)

        # Update forecast value
        curstep_renewable_gen_p_max, nextstep_renewable_gen_p_max = \
            self.forecast_reader.read_step_renewable_gen_p_max(self.sample_idx)
        nextstep_load_p = self.forecast_reader.read_step_load_p(self.sample_idx)

        action_space = self.action_space_cls.update(grid, self.steps_to_recover_gen, self.steps_to_close_gen,
                                                     rounded_gen_p, nextstep_renewable_gen_p_max)

        rho = self._calc_rho(grid, settings)

        # pack obs
        self.obs = Observation(
            grid=grid, timestep=self.timestep, action_space=action_space,
            steps_to_reconnect_line=self.steps_to_reconnect_line,
            count_soft_overflow_steps=self.count_soft_overflow_steps, rho=rho,
            gen_status=self.gen_status, steps_to_recover_gen=self.steps_to_recover_gen,
            steps_to_close_gen=self.steps_to_close_gen,
            curstep_renewable_gen_p_max=curstep_renewable_gen_p_max,
            nextstep_renewable_gen_p_max=nextstep_renewable_gen_p_max,
            rounded_gen_p=rounded_gen_p,
            nextstep_load_p=nextstep_load_p
        )

        self.reward = self.get_reward(self.obs, last_obs)
        return self.return_res()

    def get_snapshot(self):
        return (self.obs, self.sample_idx)

    def _check_balance_bound(self, grid, settings):
        balanced_id = settings.balanced_id
        min_balanced_bound = settings.min_balanced_gen_bound
        max_balanced_bound = settings.max_balanced_gen_bound
        max_gen_p = settings.max_gen_p
        min_gen_p = settings.min_gen_p
        prod_p = grid.prod_p[0]
        return prod_p[balanced_id] < min_balanced_bound * min_gen_p[balanced_id] or \
            prod_p[balanced_id] > max_balanced_bound * max_gen_p[balanced_id]

    def _calc_rho(self, grid, settings):
        limit = settings.line_thermal_limit
        num_line = settings.num_line
        a_or = grid.a_or
        a_ex = grid.a_ex
        _rho = [None] * num_line
        for i in range(num_line):
            _rho[i] = max(a_or[0][i], a_ex[0][i]) / (limit[i] + 1e-3)
        return _rho

    def _injection_auto_mapping(self, injection_gen_p):
        """
        based on the last injection q, map the value of injection_gen_p
        from (0, min_gen_p) to 0/min_gen_p
        """
        for i in self.settings.thermal_ids:
            if injection_gen_p[i] > 0 and injection_gen_p[i] < settings.min_gen_p[i]:
                if self.last_injection_gen_p[i] == settings.min_gen_p[i]:
                    injection_gen_p[i] = 0.0  # close the generator
                elif self.last_injection_gen_p[i] > settings.min_gen_p[i]:
                    injection_gen_p[i] = settings.min_gen_p[i]  # mapped to the min_gen_p
                elif self.last_injection_gen_p[i] == 0.0:
                    injection_gen_p[i] = settings.min_gen_p[i]  # open the generator
                else:
                    assert False  # should never in (0, min_gen_p)

        return injection_gen_p

    def _injection_auto_mapping_v2(self, inject_gen_p):
        """
        based on the last injection q, map the value of injection_gen_p
        from (0, min_gen_p) to 0/min_gen_p
        """
        injection_gen_p = inject_gen_p
        for i in self.settings.thermal_ids:
            if injection_gen_p[i] > 0 and injection_gen_p[i] < settings.min_gen_p[i]:
                if self.last_injection_gen_p[i] > settings.min_gen_p[i]:
                    # if injection_gen_p[i] < settings.min_gen_p[i]:
                    injection_gen_p[i] = settings.min_gen_p[i]
                elif self.last_injection_gen_p[i] == settings.min_gen_p[i]:
                    if injection_gen_p[i] < settings.min_gen_p[i]/2:
                        injection_gen_p[i] = 0.0
                    else:
                        injection_gen_p[i] = settings.min_gen_p[i]
                elif self.last_injection_gen_p[i] == 0.0:
                    if injection_gen_p[i] > settings.min_gen_p[i]/2:
                        injection_gen_p[i] = settings.min_gen_p[i]
                    else:
                        injection_gen_p[i] = 0.0
        return injection_gen_p

    def _update_gen_status(self, injection_gen_p):
        settings = self.settings
        for i in settings.thermal_ids:
            if injection_gen_p[i] == 0.0:
                if self.gen_status[i] == 1:  # the generator is open
                    assert self.steps_to_close_gen[i] == 0
                    self.gen_status[i] = 0  # close the generator
                    # self.steps_to_recover_gen[i] = settings.max_steps_to_recover_gen
                    self.steps_to_recover_gen[i] = settings.max_steps_to_recover_gen[i]
            elif injection_gen_p[i] == settings.min_gen_p[i]:
                if self.gen_status[i] == 0:  # the generator is shutdown
                    assert self.steps_to_recover_gen[i] == 0  # action isLegal function should have checked
                    self.gen_status[i] = 1  # open the generator
                    # self.steps_to_close_gen[i] = settings.max_steps_to_close_gen
                    self.steps_to_close_gen[i] = settings.max_steps_to_close_gen[i]

            if self.steps_to_recover_gen[i] > 0:
                self.steps_to_recover_gen[i] -= 1  # update recover timesteps counter
            if self.steps_to_close_gen[i] > 0:
                self.steps_to_close_gen[i] -= 1  # update close timesteps counter


    def __update_gen_status(self, injection_gen_p, gen_status, steps_to_recover_gen, steps_to_close_gen):
        new_gen_status = copy.deepcopy(gen_status)
        new_steps_to_recover_gen = copy.deepcopy(steps_to_recover_gen)
        new_steps_to_close_gen = copy.deepcopy(steps_to_close_gen)
        settings = self.settings
        for i in settings.thermal_ids:
            if injection_gen_p[i] == 0.0:
                if gen_status[i] == 1:  # the generator is open
                    assert steps_to_close_gen[i] == 0
                    new_gen_status[i] = 0  # close the generator
                    # new_steps_to_recover_gen[i] = settings.max_steps_to_recover_gen
                    new_steps_to_recover_gen[i] = settings.max_steps_to_recover_gen[i]
            elif injection_gen_p[i] == settings.min_gen_p[i]:
                if gen_status[i] == 0:  # the generator is shutdown
                    assert steps_to_recover_gen[i] == 0  # action isLegal function should have checked
                    new_gen_status[i] = 1  # open the generator
                    # new_steps_to_close_gen[i] = settings.max_steps_to_close_gen
                    new_steps_to_close_gen[i] = settings.max_steps_to_close_gen[i]

            if steps_to_recover_gen[i] > 0:
                new_steps_to_recover_gen[i] -= 1  # update recover timesteps counter
            if steps_to_close_gen[i] > 0:
                new_steps_to_close_gen[i] -= 1  # update close timesteps counter
        return new_gen_status, new_steps_to_recover_gen, new_steps_to_close_gen

    def _check_gen_status(self, injection_gen_p, rounded_gen_p):
        # check gen_p value of thermal generators after calling grid.env_feedback
        for i in settings.thermal_ids:
            if self.gen_status[i] == 0:
                assert rounded_gen_p[i] == 0.0
            else:
                assert rounded_gen_p[i] >= self.settings.min_gen_p[i], (i, rounded_gen_p[i], self.settings.min_gen_p[i])

            assert abs(injection_gen_p[i] - rounded_gen_p[i]) <= self.settings.env_allow_precision, (i, injection_gen_p[i], rounded_gen_p[i])

        for i in settings.renewable_ids:
            assert abs(injection_gen_p[i] - rounded_gen_p[i]) <= self.settings.env_allow_precision, (i, injection_gen_p[i], rounded_gen_p[i])

    def __check_gen_status(self, injection_gen_p, rounded_gen_p, gen_status):
        # check gen_p value of thermal generators after calling grid.env_feedback
        for i in settings.thermal_ids:
            if gen_status[i] == 0:
                assert rounded_gen_p[i] == 0.0
            else:
                # print(gen_status[i], rounded_gen_p[i], self.settings.min_gen_p[i])
                assert rounded_gen_p[i] >= self.settings.min_gen_p[i], (i, rounded_gen_p[i], self.settings.min_gen_p[i])

            assert abs(injection_gen_p[i] - rounded_gen_p[i]) <= self.settings.env_allow_precision, (i, injection_gen_p[i], rounded_gen_p[i])

        for i in settings.renewable_ids:
            assert abs(injection_gen_p[i] - rounded_gen_p[i]) <= self.settings.env_allow_precision, (i, injection_gen_p[i], rounded_gen_p[i])

    def _check_action(self, act):
        assert 'adjust_gen_p' in act
        assert 'adjust_gen_v' in act

        adjust_gen_p = act['adjust_gen_p']
        adjust_gen_v = act['adjust_gen_v']

        assert isinstance(adjust_gen_p, (list, tuple, np.ndarray))
        assert len(adjust_gen_p) == self.settings.num_gen

        assert isinstance(adjust_gen_v, (list, tuple, np.ndarray))
        assert len(adjust_gen_v) == self.settings.num_gen

    def _round_p(self, p):
        return [round(x, self.settings.keep_decimal_digits) for x in p]

    def get_reward(self, obs, last_obs):
        reward_dict = {
            "EPRIReward": EPRIReward,
            "line_over_flow_reward": line_over_flow_reward,
            "renewable_consumption_reward": renewable_consumption_reward,
            "running_cost_reward": running_cost_reward,
            "balanced_gen_reward": balanced_gen_reward,
            "gen_reactive_power_reward": gen_reactive_power_reward,
            "sub_voltage_reward": sub_voltage_reward,
        }
        reward_func = reward_dict[self.reward_type]
        return reward_func(obs, last_obs, settings)

    def check_done(self, grid, settings):
        if grid.flag == 1:
            return True, 'grid is not converged'
        if self.sample_idx >= settings.num_sample:
            return True, 'sample idx reach the limit'
        if self._check_balance_bound(grid, settings):
           return True, 'balance gen out of bound'
        return False, None

    def return_res(self, info=None, done=False):
        ret_obs = copy.deepcopy(self.obs)
        if done:
            if not info:
                return ret_obs, 0, True, {}
            else:
                return ret_obs, 0, True, {'fail_info':info}
        else:
            assert self.reward, "the reward are not calculated yet"
            return ret_obs, self.reward, False, {}

