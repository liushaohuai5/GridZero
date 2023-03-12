import os
import sys
sys.path.append('/workspace/AdversarialGridZero')
import csv

from pypower.api import case126, case39, runpf, runopf, ppoption, printpf, rundcopf, runuopf, runduopf
import numpy as np
from utilize.settings import settings
from Reward.rewards import *
from game.gridsim.utils import *
# from experiments.gridsim_v2 import GridSimExperientConfig
import copy
import time


from pypower.opf_args import opf_args2
from pypower.ppoption import ppoption
from pypower.isload import isload
from pypower.totcost import totcost
from pypower.fairmax import fairmax
from pypower.opf import opf

from pypower.idx_bus import PD
from pypower.idx_gen import GEN_STATUS, PG, QG, PMIN, PMAX, MU_PMIN

from numpy import flatnonzero as find
from model_jm.plot_sg126_net import Visualizer


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def run_uopf(observation, ppopt, ppc):
    open_hot = np.zeros(settings.num_gen + 1)
    close_hot = np.zeros(settings.num_gen + 1)
    t0 = time.time()  ## start timer

    ##-----  do combined unit commitment/optimal power flow  -----

    ## check for sum(Pmin) > total load, decommit as necessary
    on = find((ppc["gen"][:, GEN_STATUS] > 0) & ~isload(ppc["gen"]))  ## gens in service
    onld = find((ppc["gen"][:, GEN_STATUS] > 0) & isload(ppc["gen"]))  ## disp loads in serv
    load_capacity = sum(ppc["bus"][:, PD]) - sum(ppc["gen"][onld, PMIN])  ## total load capacity
    Pmin = ppc["gen"][on, PMIN]
    Pmax = ppc["gen"][on, PMAX]
    while sum(Pmin) > load_capacity:
        thermal_on = list(set(on) & set(settings.thermal_ids))
        ## shut down most expensive unit
        Pmin_thermal_on = ppc["gen"][thermal_on, PMIN]
        avgPmincost = totcost(ppc["gencost"][thermal_on, :], Pmin_thermal_on) / Pmin_thermal_on
        # _, i_to_close = fairmax(avgPmincost)  ## pick one with max avg cost at Pmin
        avgPmincost = list(avgPmincost)
        i_to_close = avgPmincost.index(max(avgPmincost))  ## pick one with max avg cost at Pmin
        i = thermal_on[i_to_close]  ## convert to generator index

        ## set generation to zero
        ppc["gen"][i, [PG, QG, GEN_STATUS, PMIN, PMAX]] = 0

        ## update minimum gen capacity
        on = find((ppc["gen"][:, GEN_STATUS] > 0) & ~isload(ppc["gen"]))  ## gens in service
        Pmin = ppc["gen"][on, PMIN]
        close_hot[i] = 1
        # print('Shutting down generator %d.\n' % i)

    off = find((ppc["gen"][:, GEN_STATUS] == 0))
    while sum(Pmax) < load_capacity:
        thermal_off = list(set(off) & set(settings.thermal_ids))
        ## restart cheapest unit
        # Pmin_thermal_off = ppc["gen"][thermal_off, PMIN]
        # avgPmincost = totcost(ppc['gen'][thermal_off, :], Pmin_thermal_off) / Pmin_thermal_off
        avgPmincost = np.array(settings.startup_cost)[thermal_off]
        # _, i_to_restart = fairmax(-avgPmincost)
        avgPmincost = list(avgPmincost)
        i_to_restart = avgPmincost.index(min(avgPmincost))
        i = thermal_off[i_to_restart]

        # restart
        ppc['gen'][i, [PG, PMIN, PMAX]] = settings.min_gen_p[i]
        ppc['gen'][i, GEN_STATUS] = 1

        on = find((ppc["gen"][:, GEN_STATUS] > 0) & ~isload(ppc['gen']))
        off = find((ppc["gen"][:, GEN_STATUS] == 0))
        Pmax = ppc["gen"][on, PMAX]
        open_hot[i] = 1
        print('restarting generator %d.\n' % i)

    ## run initial opf
    results = opf(ppc, ppopt)

    ## best case so far
    # results1 = copy.deepcopy(results)
    #
    # ## best case for this stage (ie. with n gens shut down, n=0,1,2 ...)
    # results0 = copy.deepcopy(results1)
    # ppc["bus"] = results0["bus"].copy()  ## use these V as starting point for OPF
    #
    # while True:
    #     ## get candidates for shutdown
    #     candidates = find((results0["gen"][:, MU_PMIN] > 0) & (results0["gen"][:, PMIN] > 0))
    #     candidates = list(set(candidates) & set(settings.thermal_ids))
    #     if len(candidates) == 0:
    #         break
    #
    #     ## do not check for further decommitment unless we
    #     ##  see something better during this stage
    #     done = True
    #
    #     for k in candidates:
    #         ## start with best for this stage
    #         ppc["gen"] = results0["gen"].copy()
    #
    #         ## shut down gen k
    #         ppc["gen"][k, [PG, QG, GEN_STATUS, PMIN, PMAX]] = 0
    #
    #         ## run opf
    #         results = opf(ppc, ppopt)
    #
    #         ## something better?
    #         if results['success'] and (results["f"] < results1["f"]):
    #             results1 = copy.deepcopy(results)
    #             k1 = k
    #             done = False  ## make sure we check for further decommitment
    #
    #     if done:
    #         ## decommits at this stage did not help, so let's quit
    #         break
    #     else:
    #         ## shutting something else down helps, so let's keep going
    #         print('Shutting down generator %d.\n' % k1)
    #         close_hot[k1] = 1
    #         results0 = copy.deepcopy(results1)
    #         ppc["bus"] = results0["bus"].copy()  ## use these V as starting point for OPF
    #         break
    #
    #
    # while True:
    #     ## get candidates for restart
    #     candidates = find((results0["gen"][:, MU_PMIN] > 0) & (results0["gen"][:, PMIN] > 0))
    #     candidates = list(set(candidates) & set(settings.thermal_ids))
    #     if len(candidates) == 0:
    #         break
    #
    #     ## do not check for further decommitment unless we
    #     ##  see something better during this stage
    #     done = True
    #
    #     for k in candidates:
    #         ## start with best for this stage
    #         ppc["gen"] = results0["gen"].copy()
    #
    #         ## restart gen k
    #         ppc['gen'][k, [PG, PMIN, PMAX]] = settings.min_gen_p[k]
    #         ppc['gen'][k, GEN_STATUS] = 1
    #
    #         ## run opf
    #         results = opf(ppc, ppopt)
    #
    #         ## something better?
    #         if results['success'] and (results["f"] < results1["f"]):
    #             results1 = copy.deepcopy(results)
    #             k1 = k
    #             done = False  ## make sure we check for further decommitment
    #
    #     if done:
    #         ## decommits at this stage did not help, so let's quit
    #         break
    #     else:
    #         ## shutting something else down helps, so let's keep going
    #         print('restarting generator %d.\n' % k1)
    #         open_hot[k1] = 1
    #         results0 = copy.deepcopy(results1)
    #         ppc["bus"] = results0["bus"].copy()  ## use these V as starting point for OPF
    #         break


    ## compute elapsed time
    et = time.time() - t0

    ## finish preparing output
    # results0['et'] = et
    # return results0, open_hot, close_hot
    return results, open_hot, close_hot

def rerun_opf(observation):
    ppc = case126()
    gen2busM = gen_to_bus_matrix(observation, settings)
    ld2busM = load_to_bus_matrix(observation, settings)

    open_ids = np.where(observation.gen_status > 0)[0].tolist()
    close_ids = np.where(observation.gen_status == 0)[0].tolist()

    # BUS DATA
    load_p = np.asarray(observation.nextstep_load_p)
    ppc['bus'][:, 2] = np.matmul(load_p, ld2busM)  # Pd
    load_q = np.asarray(observation.load_q)
    ppc['bus'][:, 3] = np.matmul(load_q, ld2busM)  # Qd
    Vm = np.asarray(observation.bus_v)
    ppc['bus'][:, 7] = Vm  # Vm
    # TODO: add bus angle information
    Va = np.asarray(observation.bus_ang)  # Va
    ppc['bus'][:, 8] = Va

    # GEN_DATA
    gen_p = np.asarray(observation.gen_p)
    ppc['gen'][:, 1] = gen_p  # Pg
    gen_q = np.asarray(observation.gen_q)
    ppc['gen'][:, 2] = gen_q  # Qg
    gen_v = np.asarray(observation.gen_v)
    ppc['gen'][:, 5] = gen_v  # Vg
    gen_status = np.asarray(observation.gen_status)
    ppc['gen'][:, 7] = gen_status  # status

    bal_gen_p_mid = (settings.min_gen_p[settings.balanced_id] + settings.max_gen_p[settings.balanced_id]) / 2

    ppc['gen'][:, 8] = np.array(settings.max_gen_p)
    ppc['gen'][settings.balanced_id, 8] = bal_gen_p_mid + 100
    ppc['gen'][close_ids, 8] = 0
    ppc['gen'][settings.renewable_ids, 8] = np.array(observation.nextstep_renewable_gen_p_max) * 0.6
    ppc['gen'][:, 9] = np.array(settings.min_gen_p)
    ppc['gen'][settings.balanced_id, 9] = bal_gen_p_mid - 100
    ppc['gen'][close_ids, 9] = 0

    ppopt = ppoption(PF_DC=True, VERBOSE=0)
    # result = opf(ppc, ppopt)
    result, open_hot, close_hot = run_uopf(observation, ppopt, ppc)
    result['gen'][:, 1] = result['gen'][:, 1].clip(ppc['gen'][:, 9], ppc['gen'][:, 8])
    new_gen_p = copy.deepcopy(result['gen'][:, 1])
    recover_ids = np.where(open_hot[:settings.num_gen] > 0)[0].tolist()
    close_ids = np.where(close_hot[:settings.num_gen] > 0)[0].tolist()
    return new_gen_p, recover_ids, close_ids


class traditional_solver:

    def __init__(self, config, observation, dc_opf_flag, unit_comb_flag, for_training_flag):
        self.config = config
        self.game = config.new_game(
            is_test=False, two_player=True, attack_all=False
        )
        self.ppc = case126()
        # observation = self.game.reset(ori_obs=True)
        self.gen2busM = gen_to_bus_matrix(observation, settings)
        self.ld2busM = load_to_bus_matrix(observation, settings)
        # self.ppc['gencost'][settings.renewable_ids, 4:] = 0
        self.ppc['gencost'][settings.renewable_ids, 4] = min(self.ppc['gencost'][settings.renewable_ids, 4]) / 2
        self.ppc['gencost'][settings.renewable_ids, 5] = min(self.ppc['gencost'][settings.renewable_ids, 5]) / 2
        self.ppc['gencost'][settings.renewable_ids, 6] = min(self.ppc['gencost'][settings.renewable_ids, 6]) / 2
        self.dc_opf_flag = dc_opf_flag
        self.unit_comb_flag = unit_comb_flag
        self.for_training_flag = for_training_flag
        self.visualizer = Visualizer()

    def spin(self):
        voltage_violations, reactive_violations, bal_p_violations, soft_overflows, hard_overflows = 0, 0, 0, 0, 0
        running_costs, renewable_consumption = [], []
        start_sample_idx = [
            22753, 16129, 74593, 45793, 32257, 53569, 13826, 26785, 1729, 17281,
            34273, 36289, 44353, 52417, 67105, 75169, 289, 4897, 15841, 31969]

        scores = []
        steps = []
        for i, start_idx in enumerate(start_sample_idx):
            st = time.time()
            step, score, epi_vol_voilations, epi_reac_violations, epi_bal_p_violations, epi_soft_overflows, epi_hard_overflows, epi_running_cost, epi_renewable_consumption = self.play_game(start_idx, i)
            print(f'episode time={time.time()-st:.3f}')
            scores.append(score)
            steps.append(step)
            voltage_violations += epi_vol_voilations
            reactive_violations += epi_reac_violations
            bal_p_violations += epi_bal_p_violations
            soft_overflows += epi_soft_overflows
            hard_overflows += epi_hard_overflows
            running_costs.append(epi_running_cost)
            renewable_consumption.append(epi_renewable_consumption)

        test_vol_vio_rate = voltage_violations / sum(steps)
        test_reac_vio_rate = reactive_violations / sum(steps)
        test_bal_p_vio_rate = bal_p_violations / sum(steps)
        test_soft_overflows = soft_overflows / sum(steps)
        test_hard_overflows = hard_overflows / sum(steps)
        # test_running_cost = sum(running_costs) / len(running_costs)
        test_running_cost = sum(running_costs) / sum(steps)
        # test_renewable_consumption = sum(renewable_consumption) / len(renewable_consumption)
        test_renewable_consumption = sum(renewable_consumption) / sum(steps)
        print(f'test_vol_vio_rate={test_vol_vio_rate:.3f}, test_reac_vio_rate={test_reac_vio_rate:.3f}, '
              f'test_bal_p_vio_rate={test_bal_p_vio_rate:.3f}, test_soft_overflows={test_soft_overflows:.3f}, '
              f'test_hard_overflows={test_hard_overflows:.3f}')
        print(f'test_running_cost={test_running_cost:.2f}, test_renewable_consumption={test_renewable_consumption:.3f}')

        return scores, steps

    def play_game(self, start_idx, epi_idx, expert_uc=None, opt_name=None, ori_renewable_max=None):

        voltage_violations, reactive_violations, bal_p_violations, soft_overflows, hard_overflows = 0, 0, 0, 0, 0
        running_costs, renewable_cosumption_rates = [], []
        data = []
        observation = self.game.reset(ori_obs=True, start_sample_idx=start_idx)
        done = False
        score = 0
        step = 0

        # if not os.path.exists(
        #         os.path.join('./results/gridsim_v2/trad_expert', f'figs/epi_{start_idx}_{opt_name}')):
        #     os.makedirs(os.path.join('./results/gridsim_v2/trad_expert', f'figs/epi_{start_idx}_{opt_name}'))
        # self.visualizer.plot(observation.gen_p, observation.gen_q, observation.load_p,
        #                      observation.load_q, observation.gen_status,
        #                      save_path=os.path.join('./results/gridsim_v2/trad_expert',
        #                                             f'figs/epi_{start_idx}_{opt_name}'),
        #                      step=step)

        if expert_uc is not None:
            tmp = np.zeros_like(expert_uc[:, :, 0:1])
            expert_uc = np.concatenate((expert_uc, tmp), axis=2)

        while not done and step < 288:
            last_observation = copy.deepcopy(observation)
            adjust_gen_p, result, _, _ = self.run_opf(observation, expert_uc[:, :, step:step+3] if expert_uc is not None else None)
            observation, reward, done, info = self.game.step(adjust_gen_p, ori_obs=True)
            state, ready_mask, closable_mask = get_state_from_obs(observation, settings, self.config.parameters)

            open_action_high = observation.action_space['adjust_gen_p'].high * observation.gen_status
            adjust_gen_p_max = sum(open_action_high[:17]) + sum(open_action_high[18:]) + settings.max_gen_p[17] - observation.gen_p[17]
            open_action_low = observation.action_space['adjust_gen_p'].low * (1 - closable_mask[:-1]) * observation.gen_status
            adjust_gen_p_min = sum(open_action_low[:17]) + sum(open_action_low[18:]) + settings.min_gen_p[17] - observation.gen_p[17]
            delta_load = sum(observation.nextstep_load_p) - sum(observation.load_p)
            print(f'epi_idx={epi_idx}, start_idx={start_idx}, step={step}, delta_load={delta_load:.2f}, adjust_gen_p_max={adjust_gen_p_max}, '
                  f'balanced_gen={observation.gen_p[17]}')
            print(f'expert_on={expert_uc[0, :, step].sum()}, now_on={observation.gen_status[settings.thermal_ids].sum()}')
            print(f'renewable_loss={sum(observation.curstep_renewable_gen_p_max) - sum(np.array(observation.gen_p)[settings.renewable_ids]):.3f}')
            if delta_load > 0:
                if adjust_gen_p_max < delta_load:
                    print('no solution...need restart')
            if delta_load < 0:
                if adjust_gen_p_min > delta_load:
                    print('no solution...need closing')
            if info != {}:
                import ipdb
                ipdb.set_trace()
                print(info)

            if ((np.asarray(observation.rho) > 1) * (np.asarray(observation.rho) <= 1.35)).any():
                soft_overflows += 1
            elif (np.asarray(observation.rho) > 1.35).any():
                hard_overflows += 1

            if gen_reactive_power_reward(observation, settings) < 0.0:
                reactive_violations += 1
            if sub_voltage_reward(observation, settings) < -0.006:
                voltage_violations += 1
            if balanced_gen_reward(observation, settings) < 0.0:
                bal_p_violations += 1

            r_c = 0.0
            for i, name in enumerate(settings.gen_name_list):
                idx = observation.unnameindex[name]
                if idx not in settings.renewable_ids:
                    r_c -= settings.second_order_cost[i] * (observation.gen_p[idx]) ** 2 + \
                           settings.first_order_cost[i] * \
                           observation.gen_p[idx] + settings.constant_cost[i]
                if observation.gen_status[idx] != last_observation.gen_status[idx] and idx in settings.thermal_ids:
                    r_c -= settings.startup_cost[i]
            running_costs.append(r_c)
            renewable_cosumption_rates.append(sum(np.asarray(observation.gen_p)[settings.renewable_ids]) / sum(
                np.asarray(observation.curstep_renewable_gen_p_max)))

            open_ids = np.where(observation.gen_status > 0)[0].tolist()
            close_ids = np.where(observation.gen_status == 0)[0].tolist()
            open_action_high, open_action_low = 0, 0
            for i in range(settings.num_gen):
                if i in open_ids and i != settings.balanced_id:
                    open_action_high += (observation.gen_p[i] + observation.action_space['adjust_gen_p'].high[i])
                    if i in settings.thermal_ids:
                        open_action_low += max(settings.min_gen_p[i], (observation.gen_p[i]+observation.action_space['adjust_gen_p'].low[i]))
                    else:
                        open_action_low += observation.gen_p[i] + observation.action_space['adjust_gen_p'].high[i]
                if i == settings.balanced_id:
                    open_action_high += settings.max_gen_p[i]
                    open_action_low += settings.min_gen_p[i]

            data.append((sum(observation.load_p), sum(np.asarray(observation.gen_p)[settings.renewable_ids]),
                         sum(observation.curstep_renewable_gen_p_max), 54 - sum(observation.gen_status),
                         sum(closable_mask), ori_renewable_max[settings.renewable_ids, step].sum(),
                         open_action_high, open_action_low, observation.gen_status))

            score += reward
            step += 1

            # if not os.path.exists(
            #         os.path.join('./results/gridsim_v2/trad_expert', f'figs/epi_{start_idx}_{opt_name}')):
            #     os.makedirs(os.path.join('./results/gridsim_v2/trad_expert', f'figs/epi_{start_idx}_{opt_name}'))
            # self.visualizer.plot(observation.gen_p, observation.gen_q, observation.load_p,
            #                      observation.load_q, observation.gen_status,
            #                      save_path=os.path.join('./results/gridsim_v2/trad_expert',
            #                                             f'figs/epi_{start_idx}_{opt_name}'),
            #                      step=step)

        headers = ('load', 'actual', 'max', 'closed_num', 'closable_num', 'renewable_prediction', 'open_action_high', 'open_action_low', 'gen_status')
        if opt_name is not None:
            path = os.path.join('./results/gridsim_v2/trad_expert/', f'new_data_{start_idx}_{opt_name}.csv')
        else:
            path = os.path.join('./results/gridsim_v2/trad_expert/', f'new_data_{start_idx}.csv')

        with open(path, 'w+', encoding='utf-8', newline='') as f:
            write = csv.writer(f)
            write.writerow(headers)
            write.writerows(data)


        print(f'Epi_{start_idx}_{opt_name}_total_running_cost={sum(running_costs)}')
        return step, score, voltage_violations, reactive_violations, bal_p_violations, soft_overflows, hard_overflows, \
               sum(running_costs), sum(renewable_cosumption_rates)
               # sum(running_costs)/len(running_costs), sum(renewable_cosumption_rates)/len(renewable_cosumption_rates)


    def run_opf(self, observation, expert_uc=None):
        self.modify_ppc(observation)
        parameters = {}
        parameters['encoder'] = 'mlp'
        parameters['action_dim'] = settings.num_gen
        parameters['only_power'] = True
        parameters['only_thermal'] = False
        _, ready_mask, closable_mask = get_state_from_obs(observation, settings, parameters)
        if self.dc_opf_flag:
            ppopt = ppoption(PF_DC=True, VERBOSE=0)
            if self.unit_comb_flag:
                result, open_hot, close_hot = self.run_uopf(observation, ppopt)
            else:
                open_hot = np.zeros(settings.num_gen + 1)
                close_hot = np.zeros(settings.num_gen + 1)
                if expert_uc is not None:
                    cnt = 0
                    for i, u in enumerate(settings.thermal_ids):
                        # # if close at next step or start at this step
                        if cnt == 1:
                            break
                        if expert_uc[2, i, 0] == 1:
                            if closable_mask[u] == 1:
                                self.ppc['gen'][u, 9] = 0
                                self.ppc['gen'][u, 8] = 0
                                cnt += 1
                            # else:
                            #     self.ppc['gen'][u, 8] = self.ppc['gen'][u, 9]

                        if expert_uc[0, i, 0] == 0 and observation.gen_status[u] == 1:
                            if closable_mask[u] == 1:
                                self.ppc['gen'][u, 9] = 0
                                self.ppc['gen'][u, 8] = 0
                                cnt += 1
                            # else:
                            #     self.ppc['gen'][u, 8] = self.ppc['gen'][u, 9]

                    for i, u in enumerate(settings.thermal_ids):
                        # if turn-on and steps==0
                        if expert_uc[1, i, 0] == 1:
                            if ready_mask[u] == 1:
                                self.ppc['gen'][u, 8] = settings.min_gen_p[u]
                                self.ppc['gen'][u, 9] = settings.min_gen_p[u]
                            break

                        if expert_uc[0, i, 0] == 1 and observation.gen_status[u] == 0:
                            if ready_mask[u] == 1:
                                self.ppc['gen'][u, 8] = settings.min_gen_p[u]
                                self.ppc['gen'][u, 9] = settings.min_gen_p[u]
                            break

                    result = opf(self.ppc, ppopt)
                    result['gen'][:, 1] = result['gen'][:, 1].clip(self.ppc['gen'][:, 9], self.ppc['gen'][:, 8])

        else:
            ppopt = ppoption(VERBOSE=0)
            if self.unit_comb_flag:
                result, open_hot, close_hot = self.run_uopf(observation, ppopt)
            else:
                result = opf(self.ppc, ppopt)
                open_hot = np.zeros(settings.num_gen + 1)
                close_hot = np.zeros(settings.num_gen + 1)
                if expert_uc is not None:
                    for i, u in enumerate(settings.thermal_ids):
                        # if closing in the coming steps, reduce power
                        if sum(expert_uc[2, i]) > 0:
                            result['gen'][u, 1] = self.ppc['gen'][u, 9]
                        # if plan is closed, and not closed now, reduce power
                        # if expert_uc[0, i, 0] == 0 and observaion.gen_p[u] >= settings.min_gen_p[u]:
                        if expert_uc[0, i, 0] == 0 and observation.gen_status[u] == 1:
                            result['gen'][u, 1] = self.ppc['gen'][u, 9]
                        # if turn-on and steps==0a
                        # if expert_uc[1, i, 0] == 1 and observaion.steps_to_recover_gen[u] == 0:
                        if expert_uc[1, i, 0] == 1:
                            result['gen'][u, 1] = self.ppc['gen'][u, 8]
                        if expert_uc[0, i, 0] == 1 and observation.gen_status[u] == 0:
                            result['gen'][u, 1] = self.ppc['gen'][u, 8]

                # result = runopf(self.ppc)
        new_gen_p = copy.deepcopy(result['gen'][:, 1])
        adjust_gen_p = new_gen_p - observation.gen_p

        open_hot[-1] = 1 - sum(open_hot[:-1])
        close_hot[-1] = 1 - sum(close_hot[:-1])
        return adjust_gen_p, result, open_hot, close_hot

    def run_uopf(self, observation, ppopt):
        open_hot = np.zeros(settings.num_gen + 1)
        close_hot = np.zeros(settings.num_gen + 1)
        _, ready_mask, closable_mask = get_state_from_obs(observation, settings, self.config.parameters)
        ready_mask = ready_mask[:-1]
        closable_mask = closable_mask[:-1]
        t0 = time.time()  ## start timer

        ##-----  do combined unit commitment/optimal power flow  -----

        ## check for sum(Pmin) > total load, decommit as necessary
        on_closable = find(closable_mask > 0 & ~isload(self.ppc["gen"]))
        close_ready = find(ready_mask > 0)
        on = find((self.ppc["gen"][:, GEN_STATUS] > 0) & ~isload(self.ppc["gen"]))  ## gens in service
        onld = find((self.ppc["gen"][:, GEN_STATUS] > 0) & isload(self.ppc["gen"]))  ## disp loads in serv
        load_capacity = sum(self.ppc["bus"][:, PD]) - sum(self.ppc["gen"][onld, PMIN])  ## total load capacity
        Pmin = self.ppc["gen"][on, PMIN]
        Pmax = self.ppc["gen"][on, PMAX]
        while sum(Pmin) > load_capacity:
            if len(on_closable) == 0:
                break
            ## shut down most expensive unit
            Pclosable = self.ppc["gen"][on_closable, PMIN]
            avgPmincost = totcost(self.ppc["gencost"][on_closable, :], Pclosable) / Pclosable
            _, i_to_close = fairmax(avgPmincost)  ## pick one with max avg cost at Pmin
            i = on_closable[i_to_close]  ## convert to generator index

            ## set generation to zero
            self.ppc["gen"][i, [PG, QG, GEN_STATUS, PMIN, PMAX]] = 0

            ## update minimum gen capacity
            on = find((self.ppc["gen"][:, GEN_STATUS] > 0) & ~isload(self.ppc["gen"]))  ## gens in service
            on_closable = np.delete(on_closable, i_to_close)
            closable_mask[i_to_close] = 0
            Pmin = self.ppc["gen"][on, PMIN]

        while sum(Pmax) < load_capacity:
            if len(close_ready) == 0:
                break
            ## restart cheapest unit
            Pready = self.ppc['gen'][close_ready, PMIN]
            avgPmincost = totcost(self.ppc['gen'][close_ready, PMIN], Pready) / Pready
            _, i_to_restart = fairmax(-avgPmincost)
            i = close_ready[i_to_restart]

            # restart
            self.ppc['gen'][i, [PG, PMIN, PMAX]] = settings.min_gen_p[i]
            self.ppc['gen'][i, GEN_STATUS] = 1

            on = find((self.ppc["gen"][:, GEN_STATUS] > 0) & ~isload(self.ppc['gen']))
            close_ready = np.delete(close_ready, i_to_restart)
            ready_mask[i_to_restart] = 0
            Pmax = self.ppc["gen"][on, PMAX]

        ## run initial opf
        results = opf(self.ppc, ppopt)

        ## best case so far
        results1 = copy.deepcopy(results)

        ## best case for this stage (ie. with n gens shut down, n=0,1,2 ...)
        results0 = copy.deepcopy(results1)
        self.ppc["bus"] = results0["bus"].copy()  ## use these V as starting point for OPF

        while True:
            ## get candidates for shutdown
            # candidates = find((results0["gen"][:, MU_PMIN] > 0) & (results0["gen"][:, PMIN] > 0) & (closable_mask > 0))
            candidates = find(closable_mask > 0)
            if len(candidates) == 0:
                break

            ## do not check for further decommitment unless we
            ##  see something better during this stage
            done = True

            for k in candidates:
                ## start with best for this stage
                self.ppc["gen"] = results0["gen"].copy()

                ## shut down gen k
                self.ppc["gen"][k, [PG, QG, GEN_STATUS, PMIN, PMAX]] = 0

                ## run opf
                results = opf(self.ppc, ppopt)

                ## something better?
                if results['success'] and (results["f"] < results1["f"]):
                    results1 = copy.deepcopy(results)
                    k1 = k
                    done = False  ## make sure we check for further decommitment

            if done:
                ## decommits at this stage did not help, so let's quit
                break
            else:
                ## shutting something else down helps, so let's keep going
                # print('Shutting down generator %d.\n' % k1)
                close_hot[k1] = 1
                results0 = copy.deepcopy(results1)
                self.ppc["bus"] = results0["bus"].copy()  ## use these V as starting point for OPF
                if self.for_training_flag:
                    break


        while True:
            ## get candidates for shutdown
            # candidates = find((results0["gen"][:, MU_PMIN] > 0) & (results0["gen"][:, PMIN] > 0) & (closable_mask > 0))
            candidates = find(ready_mask > 0)
            if len(candidates) == 0:
                break

            ## do not check for further decommitment unless we
            ##  see something better during this stage
            done = True

            for k in candidates:
                ## start with best for this stage
                self.ppc["gen"] = results0["gen"].copy()

                ## restart gen k
                self.ppc['gen'][k, [PG, PMIN, PMAX]] = settings.min_gen_p[k]
                self.ppc['gen'][k, GEN_STATUS] = 1

                ## run opf
                results = opf(self.ppc, ppopt)

                ## something better?
                if results['success'] and (results["f"] < results1["f"]):
                    results1 = copy.deepcopy(results)
                    k1 = k
                    done = False  ## make sure we check for further decommitment

            if done:
                ## decommits at this stage did not help, so let's quit
                break
            else:
                ## shutting something else down helps, so let's keep going
                # print('restarting generator %d.\n' % k1)
                open_hot[k1] = 1
                results0 = copy.deepcopy(results1)
                self.ppc["bus"] = results0["bus"].copy()  ## use these V as starting point for OPF
                if self.for_training_flag:
                    break


        ## compute elapsed time
        et = time.time() - t0

        ## finish preparing output
        results0['et'] = et

        return results0, open_hot, close_hot


    def modify_ppc(self, observation):

        open_ids = np.where(observation.gen_status > 0)[0].tolist()
        close_ids = np.where(observation.gen_status == 0)[0].tolist()

        # BUS DATA
        load_p = np.asarray(observation.nextstep_load_p)
        self.ppc['bus'][:, 2] = np.matmul(load_p, self.ld2busM)     # Pd
        load_q = np.asarray(observation.load_q)
        self.ppc['bus'][:, 3] = np.matmul(load_q, self.ld2busM)     # Qd
        Vm = np.asarray(observation.bus_v)
        self.ppc['bus'][:, 7] = Vm      # Vm
        # TODO: add bus angle information
        Va = np.asarray(observation.bus_ang)    # Va
        self.ppc['bus'][:, 8] = Va

        # GEN_DATA
        gen_p = np.asarray(observation.gen_p)
        self.ppc['gen'][:, 1] = gen_p   # Pg
        gen_q = np.asarray(observation.gen_q)
        self.ppc['gen'][:, 2] = gen_q   # Qg
        gen_v = np.asarray(observation.gen_v)
        self.ppc['gen'][:, 5] = gen_v   # Vg
        gen_status = np.asarray(observation.gen_status)
        self.ppc['gen'][:, 7] = gen_status  # status

        bal_gen_p_mid = (settings.min_gen_p[settings.balanced_id] + settings.max_gen_p[settings.balanced_id]) / 2

        action_high = observation.action_space['adjust_gen_p'].high
        self.ppc['gen'][:, 8] = observation.gen_p + action_high     # Pmax
        # self.ppc['gen'][settings.balanced_id, 8] = settings.max_gen_p[settings.balanced_id] - 70
        self.ppc['gen'][settings.balanced_id, 8] = bal_gen_p_mid + 100
        # self.ppc['gen'][:, 8] *= gen_status
        self.ppc['gen'][close_ids, 8] = 0
        action_low = observation.action_space['adjust_gen_p'].low
        self.ppc['gen'][:, 9] = observation.gen_p + action_low      # Pmin
        # self.ppc['gen'][settings.balanced_id, 9] = settings.min_gen_p[settings.balanced_id] + 70
        self.ppc['gen'][settings.balanced_id, 9] = bal_gen_p_mid - 100
        # self.ppc['gen'][:, 9] *= gen_status
        tmp = np.concatenate([np.expand_dims(np.array(settings.min_gen_p)[open_ids], axis=1),
                              np.expand_dims(self.ppc['gen'][open_ids, 9], axis=1)], axis=1)
        self.ppc['gen'][open_ids, 9] = np.max(tmp, axis=1)
        self.ppc['gen'][close_ids, 9] = 0


        # BRANCH DATA
        # line_status = np.array(observation.line_status, dtype=np.float32)
        # self.ppc['branch'][:185, 10] = line_status


# config = GridSimExperientConfig(env_id='grid', task_name='balance')
# config.initialize()
# trad_solver = traditional_solver(config)
# st = time.time()
# scores, steps = trad_solver.spin()
# print(f'all time={time.time()-st:.3f}')
# import ipdb
# ipdb.set_trace()
# mean_score = sum(scores) / len(scores)
# mean_steps = sum(steps) / len(steps)
# print('trad_mean_score:', mean_score)

