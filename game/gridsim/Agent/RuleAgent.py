
import sys
sys.path.append("/workspace/RobotEZero/game/gridsim")
from Agent.BaseAgent import BaseAgent
from utilize.form_action import *
import numpy as np
from utilize.settings import settings
# print(sys.path)
import warnings
warnings.filterwarnings('ignore')
import os
import copy

def get_adjacent_matrix(obs, settings):
    bus_num = len(settings.busname)
    adjacent_matrix = np.zeros((bus_num, bus_num))
    idx_or = 0
    for bus in settings.busname:
        # adjacent_matrix[idx_or][idx_or] = 1
        line_connectors = obs.bus_branch[bus]
        for connector in line_connectors:
            if 'or' in connector:
                line_ex = connector[:-2] + 'ex'
                for k in settings.busname:
                    lines = obs.bus_branch[k]
                    if line_ex in lines:
                        idx_ex = settings.busname.index(k)
                        adjacent_matrix[idx_or, idx_ex] = 1
                        adjacent_matrix[idx_ex, idx_or] = 1
                        break
        idx_or += 1
    return adjacent_matrix

def get_degree_matrix(obs, settings):
    adjacent_matrix = get_adjacent_matrix(obs, settings)
    degree_matrix = np.diag(np.sum(adjacent_matrix, axis=0))
    return degree_matrix

def gen_to_bus_matrix(obs, settings):
    gen_num = settings.num_gen
    bus_num = len(settings.busname)
    gen2busM = np.zeros((gen_num, bus_num))
    for bus in settings.busname:
        bus_idx = settings.busname.index(bus)
        for gen in obs.bus_gen[bus]:
            if gen is not '':
                gen_idx = settings.gen_name_list.index(gen)
                gen2busM[gen_idx][bus_idx] = 1
    return gen2busM

def bus_to_line_matrix(obs, settings):
    bus_num = len(settings.busname)
    ln_num = settings.num_line
    bus2lineM = np.zeros((bus_num, ln_num))
    miss_lines = []
    for bus in settings.busname:
        bus_idx = settings.busname.index(bus)
        for branch in obs.bus_branch[bus]:
            lnname = branch[:-3]
            # if lnname == 'branch7':
            #     import ipdb
            #     ipdb.set_trace()
            #     continue
            if lnname in settings.lnname:
                ln_idx = settings.lnname.index(lnname)
            else:
                miss_lines.append(lnname)
                continue
            if branch[-2:] == 'or':
                if obs.p_or[ln_idx] > 0:
                    bus2lineM[bus_idx][ln_idx] = 1
                elif obs.p_or[ln_idx] < 0:
                    bus2lineM[bus_idx][ln_idx] = -1
            elif branch[-2:] == 'ex':
                if obs.p_ex[ln_idx] < 0:
                    bus2lineM[bus_idx][ln_idx] = -1
                elif obs.p_ex[ln_idx] > 0:
                    bus2lineM[bus_idx][ln_idx] = 1
    # print(miss_lines)
    return bus2lineM

def gen_to_line_influence_matrix(obs, settings):

    adjacent_matrix = get_adjacent_matrix(obs, settings)
    gen2busM = gen_to_bus_matrix(obs, settings)
    bus2lineM = bus_to_line_matrix(obs, settings)
    SecondOrderA = np.matmul(adjacent_matrix, adjacent_matrix)
    ThirdOrderA = np.matmul(adjacent_matrix, np.matmul(adjacent_matrix, adjacent_matrix))
    temp = gen2busM + 0.3*(np.matmul(gen2busM, adjacent_matrix)) + 0.1*(np.matmul(gen2busM, SecondOrderA)) \
           + 0.03*(np.matmul(gen2busM, ThirdOrderA))
    gen2lineInfM = np.matmul(temp, bus2lineM)
    return gen2lineInfM

def adjust_renewable_generator(obs, settings, type='middle'):  # type = 'max' or 'middle'
    delta_load = sum(obs.nextstep_load_p) - sum(obs.load_p)
    # delta_load /= 100
    if type == 'max':
        adjust_gen_renewable = obs.action_space['adjust_gen_p'].high[settings.renewable_ids]
    else:

        renewable_action_high = obs.action_space['adjust_gen_p'].high[settings.renewable_ids]
        renewable_action_low = obs.action_space['adjust_gen_p'].low[settings.renewable_ids]
        if delta_load >= sum(renewable_action_low) and delta_load <= sum(renewable_action_high):
            # import ipdb
            # ipdb.set_trace()
            ratio = (delta_load - sum(renewable_action_low)) / (sum(renewable_action_high) - sum(renewable_action_low))
            adjust_gen_renewable = ratio * (renewable_action_high - renewable_action_low) + renewable_action_low
        elif delta_load < sum(renewable_action_low):
            adjust_gen_renewable = obs.action_space['adjust_gen_p'].low[settings.renewable_ids]
        elif delta_load > sum(renewable_action_high):
            adjust_gen_renewable = obs.action_space['adjust_gen_p'].high[settings.renewable_ids]
    return adjust_gen_renewable, delta_load - sum(adjust_gen_renewable)

def get_gen_rho_mean(obs, settings):
    gen_rho_mean = np.zeros(settings.num_gen)
    gen2busM = gen_to_bus_matrix(obs, settings)
    for gen_idx in range(settings.num_gen):
        bus_idx = np.where(gen2busM[gen_idx, :] == 1)[0].tolist()[0]
        bus_name = settings.busname[bus_idx]
        connectors = obs.bus_branch[bus_name]
        temp = []
        for connector in connectors:
            line_name = connector[:-3]
            if line_name in settings.lnname:
                line_idx = settings.lnname.index(line_name)
                temp.append(obs.rho[line_idx])
        gen_rho_mean[gen_idx] = sum(temp) / (len(temp)+1e-6)

    return gen_rho_mean


def adjust_generator_p(obs, settings, original_bus_branch, last_p_or, last_p_ex, last_rho):
    restart_flag = False
    adjust_gen_p_max = sum(obs.action_space['adjust_gen_p'].high[:17]) + sum(obs.action_space['adjust_gen_p'].high[18:]) \
                       + settings.max_gen_p[17] - obs.gen_p[17]
    delta_load = sum(obs.nextstep_load_p) - sum(obs.load_p)
    if delta_load > 0:
        if adjust_gen_p_max < delta_load:
            print('no solution')

    gen_rho_mean = get_gen_rho_mean(obs, settings)
    gen_rho_inv = 1 / gen_rho_mean
    gen_degree = np.diagonal(get_degree_matrix(obs, settings))
    balance_upper_redundancy = 150
    balance_lower_redundancy = 100

    if obs.gen_p[settings.balanced_id] > settings.max_gen_p[settings.balanced_id] - balance_upper_redundancy:  #??????????threshold???????????
        delta_balance = obs.gen_p[settings.balanced_id] - (settings.max_gen_p[settings.balanced_id] - balance_upper_redundancy)
    elif obs.gen_p[settings.balanced_id] < settings.min_gen_p[settings.balanced_id] + balance_lower_redundancy:  #??????????threshold?? ?????????
        delta_balance = obs.gen_p[settings.balanced_id] - (settings.min_gen_p[settings.balanced_id] + balance_lower_redundancy)
    else:
        delta_balance = 0
    # print(f'delta_load={delta_load:.2f}')
    # print(f'delta_balance={delta_balance:.2f}')
    delta_load += delta_balance
    gen_ids = [i for i in range(settings.num_gen)]
    closed_gen_ids = np.where(obs.gen_status == 0)
    closed_gen_ids = closed_gen_ids[0].tolist()  # ??????id
    renewable_adjustable_ids = copy.copy(settings.renewable_ids)
    open_gen_ids = []   # ??????
    for i in settings.thermal_ids:
        if i not in closed_gen_ids:
            open_gen_ids.append(i)
    closable_gen_ids = []
    for i in open_gen_ids:
        if obs.steps_to_close_gen[i] == 0 and obs.gen_p[i] == settings.min_gen_p[i]:
            closable_gen_ids.append(i)
    adjustable_gen_ids = []
    for i in open_gen_ids:
        if i not in closable_gen_ids:
            adjustable_gen_ids.append(i)
    ready_thermal_ids = []  # ???????
    for i in closed_gen_ids:
        if obs.steps_to_recover_gen[i] == 0:
            ready_thermal_ids.append(i)
    # print(f'closed_gen={closed_gen_ids}')
    # print(f'step_restart={obs.steps_to_recover_gen[closed_gen_ids]}')
    # print(obs.action_space['adjust_gen_p'].high[closed_gen_ids])
    # print(f'closable_gen={closable_gen_ids}')
    # print(obs.action_space['adjust_gen_p'].low[closable_gen_ids])

    adjust_gen_p = np.zeros(settings.num_gen)
    overflow_line_ids = []
    lowrho_line_ids = []
    out_line_ids = []
    restart_gen_num = 0
    close_gen_num = 0
    overflow_diff = 0
    rho_diff = 0

    # if obs.timestep > 39:
    for line_idx in range(settings.num_line):
        if obs.timestep < 40:
            if obs.rho[line_idx] > settings.soft_overflow_bound:  # ???????
                overflow_line_ids.append(line_idx)
            elif obs.rho[line_idx] < 0.1:
                lowrho_line_ids.append(line_idx)
        else:
            if obs.rho[line_idx] > settings.soft_overflow_bound - 0.05:  # ???????
                overflow_line_ids.append(line_idx)
            elif obs.rho[line_idx] < 0.1:
                lowrho_line_ids.append(line_idx)
        if not obs.line_status[line_idx]:
            # overflow_line_ids.append(line_idx)
            out_line_ids.append(line_idx)
            # print('there is down line')

    if len(out_line_ids) > 0:
        out_adjust = 25
        adjacentM = get_adjacent_matrix(obs, settings)
        out_upgrade_gen_ids = []
        out_downgrade_gen_ids = []
        adjacentM = adjacentM + np.matmul(adjacentM, adjacentM)
        for out_line_id in out_line_ids:
            # print(f'overflow_lines={overflow_line_ids}')
            lnname = settings.lnname[out_line_id]
            find_or = False
            find_ex = False
            for bus in settings.busname:
                # for branch in obs.bus_branch[bus]:
                for branch in original_bus_branch[bus]:
                    if branch[:-3] == lnname and branch[-2:] == 'or':
                        find_or = True
                        idx = settings.busname.index(bus)
                        bus_or_idx = np.where(adjacentM[:, idx] > 0)[0].tolist()
                        bus_or, gen_or = [], []
                        cnt = 0
                        for idx in bus_or_idx:
                            bus_or.append(settings.busname[idx])
                            if obs.bus_gen[bus_or[cnt]][0] is not '':
                                gen_or.append(obs.bus_gen[bus_or[cnt]][0])
                            cnt += 1
                        # bus_or = settings.busname[bus_or_idx]
                        # gen_or = obs.bus_gen[bus_or]

                        # import ipdb
                        # ipdb.set_trace()
                        for gen in gen_or:
                            gen_idx = settings.gen_name_list.index(gen)
                            if last_p_or[out_line_id] > 0 and obs.steps_to_reconnect_line[out_line_id] == 16:
                                out_downgrade_gen_ids.append(gen_idx)
                                if gen_idx in adjustable_gen_ids:
                                    adjust_gen_p[gen_idx] -= out_adjust
                                    adjustable_gen_ids.remove(gen_idx)
                                elif gen_idx in renewable_adjustable_ids:
                                    adjust_gen_p[gen_idx] -= out_adjust
                                    renewable_adjustable_ids.remove(gen_idx)
                                adjust_gen_p[gen_idx] = np.clip(adjust_gen_p[gen_idx],
                                                                obs.action_space['adjust_gen_p'].low[gen_idx],
                                                                obs.action_space['adjust_gen_p'].high[gen_idx])
                            elif last_p_or[out_line_id] < 0 and obs.steps_to_reconnect_line[out_line_id] == 16:
                                out_upgrade_gen_ids.append(gen_idx)
                                if gen_idx in adjustable_gen_ids:
                                    adjust_gen_p[gen_idx] += out_adjust
                                    adjustable_gen_ids.remove(gen_idx)
                                elif gen_idx in renewable_adjustable_ids:
                                    adjust_gen_p[gen_idx] += out_adjust
                                    renewable_adjustable_ids.remove(gen_idx)
                                elif gen_idx in closable_gen_ids:
                                    adjust_gen_p[gen_idx] += out_adjust
                                    closable_gen_ids.remove(gen_idx)
                                elif gen_idx in ready_thermal_ids:
                                    adjust_gen_p[gen_idx] += out_adjust
                                    ready_thermal_ids.remove(gen_idx)
                                adjust_gen_p[gen_idx] = np.clip(adjust_gen_p[gen_idx],
                                                                obs.action_space['adjust_gen_p'].low[gen_idx],
                                                                obs.action_space['adjust_gen_p'].high[gen_idx])

                    elif branch[:-3] == lnname and branch[-2:] == 'ex':
                        find_ex = True
                        idx = settings.busname.index(bus)
                        bus_ex_idx = np.where(adjacentM[:, idx] > 0)[0].tolist()
                        bus_ex, gen_ex = [], []
                        cnt = 0
                        for idx in bus_ex_idx:
                            bus_ex.append(settings.busname[idx])
                            if obs.bus_gen[bus_ex[cnt]][0] is not '':
                                gen_ex.append(obs.bus_gen[bus_ex[cnt]][0])
                            cnt += 1
                        # bus_ex = settings.busname[bus_ex_idx]
                        # gen_ex = obs.bus_gen[bus_ex]
                        # import ipdb
                        # ipdb.set_trace()
                        for gen in gen_ex:
                            gen_idx = settings.gen_name_list.index(gen)
                            if last_p_ex[out_line_id] > 0 and obs.steps_to_reconnect_line[out_line_id] == 16:
                                out_downgrade_gen_ids.append(gen_idx)
                                if gen_idx in adjustable_gen_ids:
                                    adjust_gen_p[gen_idx] -= out_adjust
                                    adjustable_gen_ids.remove(gen_idx)
                                elif gen_idx in renewable_adjustable_ids:
                                    adjust_gen_p[gen_idx] -= out_adjust
                                    renewable_adjustable_ids.remove(gen_idx)
                                adjust_gen_p[gen_idx] = np.clip(adjust_gen_p[gen_idx], obs.action_space['adjust_gen_p'].low[gen_idx], obs.action_space['adjust_gen_p'].high[gen_idx])
                            elif last_p_ex[out_line_id] < 0 and obs.steps_to_reconnect_line[out_line_id] == 16:
                                out_upgrade_gen_ids.append(gen_idx)
                                if gen_idx in adjustable_gen_ids:
                                    adjust_gen_p[gen_idx] += out_adjust
                                    adjustable_gen_ids.remove(gen_idx)
                                elif gen_idx in renewable_adjustable_ids:
                                    adjust_gen_p[gen_idx] += out_adjust
                                    renewable_adjustable_ids.remove(gen_idx)
                                elif gen_idx in closable_gen_ids:
                                    adjust_gen_p[gen_idx] += out_adjust
                                    closable_gen_ids.remove(gen_idx)
                                elif gen_idx in ready_thermal_ids:
                                    adjust_gen_p[gen_idx] += out_adjust
                                    ready_thermal_ids.remove(gen_idx)
                                adjust_gen_p[gen_idx] = np.clip(adjust_gen_p[gen_idx],
                                                                obs.action_space['adjust_gen_p'].low[gen_idx],
                                                                obs.action_space['adjust_gen_p'].high[gen_idx])
                if find_ex and find_or:
                    # print(
                    #     f'overflow: {lnname}, status={obs.line_status[out_line_id]}, rho={obs.rho[out_line_id]:.2f} id:{out_line_id}, p_or={obs.p_or[out_line_id]:.2f}, bus_or={bus_or}, gen_or={gen_or}, p_ex={obs.p_ex[out_line_id]:.2f}, bus_ex={bus_ex}, gen_ex={gen_ex}')
                    # print(
                    #     f'outline: {lnname}, status={obs.line_status[out_line_id]}, rho={last_rho[out_line_id]:.2f} id:{out_line_id}, p_or={last_p_or[out_line_id]:.2f}, bus_or={bus_or}, gen_or={gen_or}, p_ex={last_p_ex[out_line_id]:.2f}, bus_ex={bus_ex}, gen_ex={gen_ex}, up_action={adjust_gen_p[out_upgrade_gen_ids]}, up_id={out_upgrade_gen_ids}, down_action={adjust_gen_p[out_downgrade_gen_ids]}, down_id={out_downgrade_gen_ids}')
                    break

        overflow_diff += sum(adjust_gen_p[out_upgrade_gen_ids]) + sum(adjust_gen_p[out_downgrade_gen_ids])
    if len(overflow_line_ids) > 0:
        adjust_quantity = 10
        whole_upgrade_gen_ids = []
        whole_downgrade_gen_ids = []
        gen2lineInfM = gen_to_line_influence_matrix(obs, settings)
        for overflow_line_id in overflow_line_ids:
            upgrade_gen_ids = []   # up & down adjust generators of each overflow line
            downgrade_gen_ids = []
            # print(f'overflow_lines={overflow_line_ids}')
            lnname = settings.lnname[overflow_line_id]
            find_or = False
            find_ex = False
            for bus in settings.busname:
                for branch in obs.bus_branch[bus]:
                    if branch[:-3] == lnname and branch[-2:] == 'or':
                        find_or = True
                        bus_or = bus
                        gen_or = obs.bus_gen[bus_or]
                    elif branch[:-3] == lnname and branch[-2:] == 'ex':
                        find_ex = True
                        bus_ex = bus
                        gen_ex = obs.bus_gen[bus_ex]
                if find_ex and find_or:
                    # print(
                    #     f'overflow: {lnname}, status={obs.line_status[overflow_line_id]}, rho={obs.rho[overflow_line_id]:.2f} id:{overflow_line_id}, p_or={obs.p_or[overflow_line_id]:.2f}, bus_or={bus_or}, gen_or={gen_or}, p_ex={obs.p_ex[overflow_line_id]:.2f}, bus_ex={bus_ex}, gen_ex={gen_ex}')
                    break
            temp = np.where(gen2lineInfM[:, overflow_line_id] < 0)[0].tolist()
            if len(temp) > 0:
                for id in temp:
                    if id in open_gen_ids or id in ready_thermal_ids or id in renewable_adjustable_ids:
                        upgrade_gen_ids.append(id)
                        whole_upgrade_gen_ids.append(id)

            if len(upgrade_gen_ids) > 0:
                for upgrade_gen_id in upgrade_gen_ids:
                    if upgrade_gen_id in adjustable_gen_ids:
                        adjustable_gen_ids.remove(upgrade_gen_id)
                        open_gen_ids.remove(upgrade_gen_id)
                    elif upgrade_gen_id in closable_gen_ids:
                        closable_gen_ids.remove(upgrade_gen_id)
                        open_gen_ids.remove(upgrade_gen_id)
                    elif upgrade_gen_id in ready_thermal_ids:
                        ready_thermal_ids.remove(upgrade_gen_id)
                        closed_gen_ids.remove(upgrade_gen_id)
                    elif upgrade_gen_id in renewable_adjustable_ids:
                        renewable_adjustable_ids.remove(upgrade_gen_id)
                    # for _ in range(5):
                    # if overflow_line_id in out_line_ids and obs.steps_to_reconnect_line[overflow_line_id] == 16:
                    #     adjust_gen_p[upgrade_gen_id] += 50
                    if overflow_line_id not in out_line_ids:
                        adjust_gen_p[upgrade_gen_id] += adjust_quantity
                    if len(out_line_ids) > 0 and obs.rho[overflow_line_id] > 1.2:
                        adjust_gen_p[upgrade_gen_id] += 15
                    adjust_gen_p[upgrade_gen_id] = np.clip(adjust_gen_p[upgrade_gen_id], obs.action_space['adjust_gen_p'].low[upgrade_gen_id],
                                                           obs.action_space['adjust_gen_p'].high[upgrade_gen_id])
                    # print(
                    #     f'upgrade_gen:{settings.gen_name_list[upgrade_gen_id]}, id={upgrade_gen_id}, action={adjust_gen_p[upgrade_gen_id]}')

            temp = np.where(gen2lineInfM[:, overflow_line_id] > 0)[0].tolist()
            if len(temp) > 0:
                for id in temp:
                    if id in adjustable_gen_ids or id in renewable_adjustable_ids:
                        downgrade_gen_ids.append(id)
                        whole_downgrade_gen_ids.append(id)
            if len(downgrade_gen_ids) > 0:
                for downgrade_gen_id in downgrade_gen_ids:
                    if downgrade_gen_id in adjustable_gen_ids:
                        adjustable_gen_ids.remove(downgrade_gen_id)
                        open_gen_ids.remove(downgrade_gen_id)
                    elif downgrade_gen_id in renewable_adjustable_ids:
                        renewable_adjustable_ids.remove(downgrade_gen_id)
                    # for _ in range(5):
                    # if overflow_line_id in out_line_ids and obs.steps_to_reconnect_line[overflow_line_id] == 16:
                    #     adjust_gen_p[downgrade_gen_id] -= 50
                    if overflow_line_id not in out_line_ids:
                        adjust_gen_p[downgrade_gen_id] -= adjust_quantity
                    if len(out_line_ids) > 0 and obs.rho[overflow_line_id] > 1.2:
                        adjust_gen_p[downgrade_gen_id] -= 15
                    adjust_gen_p[downgrade_gen_id] = np.clip(adjust_gen_p[downgrade_gen_id],
                                                           obs.action_space['adjust_gen_p'].low[downgrade_gen_id],
                                                           obs.action_space['adjust_gen_p'].high[downgrade_gen_id])
                    # print(
                    #     f'downgrade_gen:{settings.gen_name_list[downgrade_gen_id]}, id={downgrade_gen_id}, action={adjust_gen_p[downgrade_gen_id]}')

        overflow_diff += sum(adjust_gen_p[whole_upgrade_gen_ids]) + sum(adjust_gen_p[whole_downgrade_gen_ids])

    # if len(lowrho_line_ids) > 0:
    #     whole_up_rho_gen_ids = []
    #     gen2lineInfM = gen_to_line_influence_matrix(obs, settings)
    #     for id in lowrho_line_ids:
    #         up_rho_gen_ids = []
    #         temp = np.where(gen2lineInfM[:, id] > 0)[0].tolist()
    #         if len(temp) > 0:
    #             for id in temp:
    #                 if id in open_gen_ids or id in renewable_adjustable_ids or id in ready_thermal_ids:
    #                     up_rho_gen_ids.append(id)
    #                     whole_up_rho_gen_ids.append(id)
    #         if len(up_rho_gen_ids) > 0:
    #             for up_rho_gen_id in up_rho_gen_ids:
    #                 if up_rho_gen_id in adjustable_gen_ids:
    #                     adjustable_gen_ids.remove(up_rho_gen_id)
    #                     open_gen_ids.remove(up_rho_gen_id)
    #                 elif up_rho_gen_id in closable_gen_ids:
    #                     closable_gen_ids.remove(up_rho_gen_id)
    #                     open_gen_ids.remove(up_rho_gen_id)
    #                 elif up_rho_gen_id in ready_thermal_ids:
    #                     ready_thermal_ids.remove(up_rho_gen_id)
    #                     closed_gen_ids.remove(up_rho_gen_id)
    #                 elif up_rho_gen_id in renewable_adjustable_ids:
    #                     renewable_adjustable_ids.remove(up_rho_gen_id)
    #                 adjust_gen_p[up_rho_gen_id] += 10
    #                 adjust_gen_p[up_rho_gen_id] = np.clip(adjust_gen_p[up_rho_gen_id],
    #                                                          obs.action_space['adjust_gen_p'].low[up_rho_gen_id],
    #                                                          obs.action_space['adjust_gen_p'].high[up_rho_gen_id])
    #
    #     rho_diff += sum(adjust_gen_p[whole_up_rho_gen_ids])

    # print(f'overflow_diff={overflow_diff:.2f}')
    delta_load -= overflow_diff
    delta_load -= rho_diff

    adjustable_thermal_action_high = obs.action_space['adjust_gen_p'].high[adjustable_gen_ids]
    adjustable_thermal_action_low = obs.action_space['adjust_gen_p'].low[adjustable_gen_ids]
    open_thermal_action_high = obs.action_space['adjust_gen_p'].high[open_gen_ids]
    open_thermal_action_low = obs.action_space['adjust_gen_p'].low[open_gen_ids]
    renewable_action_high = obs.action_space['adjust_gen_p'].high[renewable_adjustable_ids]
    renewable_action_low = obs.action_space['adjust_gen_p'].low[renewable_adjustable_ids]

    # if len(ready_thermal_ids) >= len(closable_gen_ids) and close_gen_num == 0 and len(closable_gen_ids) > 1 and sum(renewable_action_high) > 0:
    # if close_gen_num == 0 and len(closable_gen_ids) > 2:
    # renewable_redundancy = sum(renewable_action_high)
    # close_power = 0
    # while len(closable_gen_ids) > 0 and renewable_redundancy > 0:
    #     idx = np.argmin(np.asarray(settings.min_gen_p)[closable_gen_ids]).tolist()
    #     close_thermal_id = closable_gen_ids[idx]
    #     adjust_gen_p[close_thermal_id] = obs.action_space['adjust_gen_p'].low[close_thermal_id]
    #     close_power += adjust_gen_p[close_thermal_id]
    #     # close_gen_num += 1
    #     renewable_redundancy += adjust_gen_p[close_thermal_id]
    #     closable_gen_ids.remove(close_thermal_id)
    #     open_gen_ids.remove(close_thermal_id)
    # # else:
    # #     close_power = 0
    # delta_load -= close_power
    close_power = 0

    while len(ready_thermal_ids) > 0 and delta_load > 0:
        # TODO: find max power generator
        idx = np.argmax(obs.action_space['adjust_gen_p'].high[ready_thermal_ids]).tolist()
        restart_id = ready_thermal_ids[idx]
        adjust_gen_p[restart_id] = obs.action_space['adjust_gen_p'].high[restart_id]
        ready_thermal_ids.remove(restart_id)
        delta_load -= adjust_gen_p[restart_id]
        restart_flag = True

    if delta_load > 0:
        # print(f'sum_renewable_action_high={sum(renewable_action_high)}, '
        #       f'sum_thermal_action_high={sum(adjustable_thermal_action_high)}')
        redundant_balanced = 0
        if delta_load >= sum(renewable_action_high):
            adjust_gen_p[renewable_adjustable_ids] = renewable_action_high
        elif delta_load < sum(renewable_action_high):
            # average_degree = np.mean(gen_rho_inv[renewable_adjustable_ids])
            ratio = (delta_load - sum(renewable_action_low)) / (sum(renewable_action_high) - sum(renewable_action_low) + 1e-6)
            # ratio = ((gen_rho_inv[renewable_adjustable_ids] - average_degree) * delta_load - sum(renewable_action_low)) / (sum(renewable_action_high) - sum(renewable_action_low) + 1e-6)
            ratio = np.clip(ratio, 0, 1)
            adjust_gen_p[renewable_adjustable_ids] = ratio * (renewable_action_high - renewable_action_low) + renewable_action_low

            redundant_renewable = sum(renewable_action_high) - sum(adjust_gen_p[renewable_adjustable_ids])
            # print(f'redundant_renewable={redundant_renewable:.2f}')
            redundant_balanced = obs.gen_p[settings.balanced_id] - settings.min_gen_p[settings.balanced_id] - 50\
                                 # - balance_lower_redundancy
            if redundant_renewable <= np.abs(sum(adjustable_thermal_action_low)) + redundant_balanced - close_power:
                adjust_gen_p[renewable_adjustable_ids] = renewable_action_high
            elif redundant_renewable > np.abs(sum(adjustable_thermal_action_low)) + redundant_balanced - close_power:
                # average_degree = np.mean(gen_rho_inv[renewable_adjustable_ids])
                # ratio = ((gen_rho_inv[renewable_adjustable_ids] - average_degree) * (delta_load + np.abs(sum(adjustable_thermal_action_low)) + redundant_balanced - close_power) - sum(renewable_action_low)) / (
                ratio = ((delta_load + np.abs(sum(adjustable_thermal_action_low)) + redundant_balanced - close_power) - sum(renewable_action_low)) / (
                            sum(renewable_action_high) - sum(renewable_action_low) + 1e-6)
                ratio = np.clip(ratio, 0, 1)
                adjust_gen_p[renewable_adjustable_ids] = ratio * (renewable_action_high - renewable_action_low) + renewable_action_low

        thermal_object = delta_load - sum(adjust_gen_p[renewable_adjustable_ids]) + redundant_balanced
        open_thermal_action_high = obs.action_space['adjust_gen_p'].high[open_gen_ids]
        open_thermal_action_low = obs.action_space['adjust_gen_p'].low[open_gen_ids]

        if thermal_object <= sum(open_thermal_action_high) and thermal_object >= sum(open_thermal_action_low):
            # average_degree = sum(gen_degree[open_gen_ids]) / len(gen_degree[open_gen_ids])
            average_rho = np.mean(gen_rho_inv[open_gen_ids])
            # ratio = ((gen_degree[open_gen_ids] / average_degree) * thermal_object - sum(open_thermal_action_low)) / (sum(open_thermal_action_high) - sum(open_thermal_action_low) + 1e-6)
            # ratio = ((gen_rho_inv[open_gen_ids] / average_rho) * thermal_object - sum(open_thermal_action_low)) / (sum(open_thermal_action_high) - sum(open_thermal_action_low) + 1e-6)
            # ratio = np.clip(ratio, 0, 1)
            # adjust_gen_p[open_gen_ids] = ratio * (open_thermal_action_high - open_thermal_action_low) + open_thermal_action_low

            ratio = (thermal_object - sum(open_thermal_action_low)) / (sum(open_thermal_action_high) - sum(open_thermal_action_low) + 1e-6)
            ratio = np.clip(ratio, 0, 1)
            adjust_gen_p[open_gen_ids] = ratio * (open_thermal_action_high - open_thermal_action_low) + open_thermal_action_low
            if len(closable_gen_ids) > 0:
                readjust_thermal = 0
                for i in closable_gen_ids:
                    idx = gen_ids.index(i)
                    if adjust_gen_p[idx] < 0:  # ??????????0
                        readjust_thermal += -adjust_gen_p[idx]
                        adjust_gen_p[idx] = 0   # ??????????
                if readjust_thermal > 0:
                    # average_degree = sum(gen_degree[adjustable_gen_ids]) / len(gen_degree[adjustable_gen_ids])
                    average_rho = np.mean(gen_rho_inv[adjustable_gen_ids])
                    # ratio = ((gen_degree[adjustable_gen_ids] / average_degree) * thermal_object - sum(adjustable_thermal_action_low)) / (
                    # ratio = ((gen_rho_inv[adjustable_gen_ids] / average_rho) * thermal_object - sum(adjustable_thermal_action_low)) / (
                    #             sum(adjustable_thermal_action_high) - sum(adjustable_thermal_action_low) + 1e-6)
                    # ratio = np.clip(ratio, 0, 1)
                    # adjust_gen_p[adjustable_gen_ids] = ratio * (
                    #         adjustable_thermal_action_high - adjustable_thermal_action_low) + adjustable_thermal_action_low

                    ratio = (thermal_object - sum(adjustable_thermal_action_low)) / (
                                sum(adjustable_thermal_action_high) - sum(adjustable_thermal_action_low) + 1e-6)
                    ratio = np.clip(ratio, 0, 1)
                    adjust_gen_p[adjustable_gen_ids] = ratio * (
                                adjustable_thermal_action_high - adjustable_thermal_action_low) + adjustable_thermal_action_low

        elif thermal_object > sum(open_thermal_action_high):
            adjust_gen_p[open_gen_ids] = open_thermal_action_high
        elif thermal_object < sum(open_thermal_action_low):
            adjust_gen_p[open_gen_ids] = open_thermal_action_low
            if len(closable_gen_ids) > 0:
                for i in closable_gen_ids:
                    adjust_gen_p[gen_ids.index(i)] = 0  # ????????????????

        thermal_diff = thermal_object - sum(adjust_gen_p[open_gen_ids])
        # print(f'thermal_diff={thermal_diff:.2f}')
        if thermal_diff > 0.5:
            if len(ready_thermal_ids) > 0:
                while len(ready_thermal_ids) > 0:
                    need_bias = thermal_diff - obs.action_space['adjust_gen_p'].high[ready_thermal_ids]
                    idx = np.argmin(np.abs(need_bias)).tolist()
                    # TODO: find max power generator
                    # idx = np.argmax(obs.action_space['adjust_gen_p'].high[ready_thermal_ids]).tolist()
                    restart_id = ready_thermal_ids[idx]
                    if np.abs(thermal_diff - obs.action_space['adjust_gen_p'].high[restart_id]) < thermal_diff:
                        adjust_gen_p[restart_id] = obs.action_space['adjust_gen_p'].high[restart_id]
                        thermal_diff -= adjust_gen_p[restart_id]
                        ready_thermal_ids.remove(restart_id)
                    else:
                        break

    elif delta_load < 0:
        # TODO: ????????????????????????????
        if sum(renewable_action_high) < 0:
            adjust_gen_p[renewable_adjustable_ids] = renewable_action_high
            thermal_object = delta_load - sum(renewable_action_high)
            if sum(renewable_action_high) < delta_load:
                if thermal_object <= sum(open_thermal_action_high):
                    # average_degree = sum(gen_degree[open_gen_ids]) / len(gen_degree[open_gen_ids])
                    average_degree = np.mean(gen_rho_inv[open_gen_ids])
                    # ratio = ((gen_degree[open_gen_ids] / average_degree) * thermal_object - sum(open_thermal_action_low)) / (sum(open_thermal_action_high) - sum(open_thermal_action_low))
                    # ratio = ((gen_rho_inv[open_gen_ids] / average_degree) * thermal_object - sum(open_thermal_action_low)) / (sum(open_thermal_action_high) - sum(open_thermal_action_low))
                    ratio = (thermal_object - sum(open_thermal_action_low)) / (sum(open_thermal_action_high) - sum(open_thermal_action_low))
                    ratio = np.clip(ratio, 0, 1)
                    adjust_gen_p[open_gen_ids] = ratio * (open_thermal_action_high - open_thermal_action_low) + open_thermal_action_low
                elif thermal_object > sum(open_thermal_action_high):
                    adjust_gen_p[open_gen_ids] = open_thermal_action_high
                if len(closable_gen_ids) > 0:  # ????????
                    readjust_thermal = 0
                    for i in closable_gen_ids:
                        idx = gen_ids.index(i)
                        if adjust_gen_p[idx] < 0:  # ??????????0
                            readjust_thermal += -adjust_gen_p[idx]
                            adjust_gen_p[idx] = 0  # ??????????
                    if readjust_thermal > 0:
                        # average_degree = sum(gen_degree[adjustable_gen_ids]) / len(gen_degree[adjustable_gen_ids])
                        average_degree = np.mean(gen_rho_inv[adjustable_gen_ids])
                        # ratio = ((gen_degree[adjustable_gen_ids] / average_degree) * thermal_object - sum(adjustable_thermal_action_low)) / (
                        # ratio = ((gen_rho_inv[adjustable_gen_ids] / average_degree) * thermal_object - sum(adjustable_thermal_action_low)) / (
                        ratio = (thermal_object - sum(adjustable_thermal_action_low)) / (
                                sum(adjustable_thermal_action_high) - sum(adjustable_thermal_action_low) + 1e-6)
                        ratio = np.clip(ratio, 0, 1)
                        adjust_gen_p[adjustable_gen_ids] = ratio * (
                                adjustable_thermal_action_high - adjustable_thermal_action_low) + adjustable_thermal_action_low
                thermal_diff = thermal_object - sum(adjust_gen_p[open_gen_ids])
                # print(f'thermal_diff={thermal_diff:.2f}')
                if thermal_diff > 0.5:
                    if len(ready_thermal_ids) > 0:
                        while len(ready_thermal_ids) > 0:
                            need_bias = thermal_diff - obs.action_space['adjust_gen_p'].high[ready_thermal_ids]
                            idx = np.argmin(np.abs(need_bias)).tolist()
                            # TODO: find max power generator
                            # idx = np.argmax(obs.action_space['adjust_gen_p'].high[ready_thermal_ids]).tolist()
                            restart_id = ready_thermal_ids[idx]
                            if np.abs(thermal_diff - obs.action_space['adjust_gen_p'].high[restart_id]) < thermal_diff:
                                adjust_gen_p[restart_id] = obs.action_space['adjust_gen_p'].high[restart_id]
                                thermal_diff -= adjust_gen_p[restart_id]
                                ready_thermal_ids.remove(restart_id)
                            else:
                                break
            elif sum(renewable_action_high) >= delta_load:
                if thermal_object <= sum(adjustable_thermal_action_low):
                    adjust_gen_p[adjustable_gen_ids] = adjustable_thermal_action_low
                elif thermal_object > sum(adjustable_thermal_action_low):
                    # average_degree = sum(gen_degree[adjustable_gen_ids]) / len(gen_degree[adjustable_gen_ids])
                    average_degree = np.mean(gen_rho_inv[adjustable_gen_ids])
                    # ratio = ((gen_degree[adjustable_gen_ids] / average_degree) * thermal_object - sum(adjustable_thermal_action_low)) / (
                    # ratio = ((gen_rho_inv[adjustable_gen_ids] / average_degree) * thermal_object - sum(adjustable_thermal_action_low)) / (
                    ratio = (thermal_object - sum(adjustable_thermal_action_low)) / (
                            sum(adjustable_thermal_action_high) - sum(adjustable_thermal_action_low) + 1e-6)
                    ratio = np.clip(ratio, 0, 1)
                    adjust_gen_p[adjustable_gen_ids] = ratio * (
                            adjustable_thermal_action_high - adjustable_thermal_action_low) + adjustable_thermal_action_low
        elif sum(renewable_action_high) > 0:
            if delta_load < sum(adjustable_thermal_action_low):
                adjust_gen_p[adjustable_gen_ids] = adjustable_thermal_action_low
            elif delta_load >= sum(adjustable_thermal_action_low):
                # average_degree = sum(gen_degree[adjustable_gen_ids]) / len(gen_degree[adjustable_gen_ids])
                average_degree = np.mean(gen_rho_inv[adjustable_gen_ids])
                # ratio = ((gen_rho_inv[adjustable_gen_ids] / average_degree) * delta_load - sum(adjustable_thermal_action_low)) / (
                ratio = (delta_load - sum(adjustable_thermal_action_low)) / (
                            sum(adjustable_thermal_action_high) - sum(adjustable_thermal_action_low) + 1e-6)
                ratio = np.clip(ratio, 0, 1)
                adjust_gen_p[adjustable_gen_ids] = ratio * (
                            adjustable_thermal_action_high - adjustable_thermal_action_low) + adjustable_thermal_action_low
                if sum(renewable_action_high) > 0:
                    # print(f'sum_renewable_action_high={sum(renewable_action_high):.2f}')
                    redundant_thermal = sum(adjustable_thermal_action_low) - sum(adjust_gen_p[adjustable_gen_ids]) \
                                        + (settings.min_gen_p[17] - obs.gen_p[17] + 50) \
                                        + close_power
                    if np.abs(redundant_thermal) < sum(renewable_action_high):
                        adjust_gen_p[adjustable_gen_ids] = adjustable_thermal_action_low
                    elif np.abs(redundant_thermal) >= sum(renewable_action_high):
                        # average_degree = sum(gen_degree[adjustable_gen_ids]) / len(gen_degree[adjustable_gen_ids])
                        average_degree = np.mean(gen_rho_inv[adjustable_gen_ids])
                        # ratio = ((gen_rho_inv[adjustable_gen_ids] / average_degree) * (delta_load - sum(renewable_action_high)) - sum(adjustable_thermal_action_low)) / (
                        ratio = ((delta_load - sum(renewable_action_high)) - sum(adjustable_thermal_action_low)) / (
                                    sum(adjustable_thermal_action_high) - sum(adjustable_thermal_action_low) + 1e-6)
                        ratio = np.clip(ratio, 0, 1)
                        adjust_gen_p[adjustable_gen_ids] = ratio * (
                                    adjustable_thermal_action_high - adjustable_thermal_action_low) + adjustable_thermal_action_low

            renewable_object = delta_load - sum(adjust_gen_p[adjustable_gen_ids])
            # print(f'renewable_object={renewable_object:.2f}')
            if renewable_object >= sum(renewable_action_low) and renewable_object <= sum(renewable_action_high):
                # average_degree = np.mean(gen_rho_inv[renewable_adjustable_ids])
                # ratio = ((gen_rho_inv[renewable_adjustable_ids] - average_degree) * renewable_object - sum(renewable_action_low)) / (sum(renewable_action_high) - sum(renewable_action_low) + 1e-6)
                ratio = (renewable_object - sum(renewable_action_low)) / (sum(renewable_action_high) - sum(renewable_action_low) + 1e-6)
                ratio = np.clip(ratio, 0, 1)
                adjust_gen_p[renewable_adjustable_ids] = ratio * (renewable_action_high - renewable_action_low) + renewable_action_low
            elif renewable_object < sum(renewable_action_low):
                adjust_gen_p[renewable_adjustable_ids] = renewable_action_low
            elif renewable_object > sum(renewable_action_high):
                adjust_gen_p[renewable_adjustable_ids] = renewable_action_high

    return adjust_gen_p, restart_flag

def form_p_action(adjust_gen_thermal, adjust_gen_renewable, settings):
    thermal_dims = len(adjust_gen_thermal)
    thermal_ids = settings.thermal_ids
    renewable_dims = len(adjust_gen_renewable)
    renewable_ids = settings.renewable_ids
    adjust_gen_p = np.zeros(54)
    for i in range(thermal_dims):
        adjust_gen_p[thermal_ids[i]] = adjust_gen_thermal[i]
    for i in range(renewable_dims):
        adjust_gen_p[renewable_ids[i]] = adjust_gen_renewable[i]
    return adjust_gen_p

'''
voltage control of all generators, constraint to 1.05
'''
def voltage_action(obs, settings, type='reactive'):  # type = 'period' or '1'
    if type == 'period':
        err = np.maximum(np.asarray(settings.min_gen_v) - np.asarray(obs.gen_v), np.asarray(obs.gen_v) - np.asarray(settings.max_gen_v))  # restrict in legal range
        gen_num = len(err)
        action_high, action_low = obs.action_space['adjust_gen_v'].high, obs.action_space['adjust_gen_v'].low
        adjust_gen_v = np.zeros(54)
        for i in range(gen_num):
            if err[i] <= 0:
                continue
            elif err[i] > 0:
                if obs.gen_v[i] < settings.min_gen_v[i]:
                    if err[i] < action_high[i]:
                        adjust_gen_v[i] = err[i]
                    else:
                        adjust_gen_v[i] = action_high[i]
                elif obs.gen_v[i] > settings.max_gen_v[i]:
                    if - err[i] > action_low[i]:
                        adjust_gen_v[i] = - err[i]
                    else:
                        adjust_gen_v[i] = action_low[i]
    elif type == '1':
        # err = obs.gen_v - (np.asarray(settings.max_gen_v) + np.asarray(settings.min_gen_v)) / 2  # restrict at stable point 1
        err = np.asarray(obs.gen_v) - 1.05  # restrict at stable point 1.05
        gen_num = len(err)
        action_high, action_low = obs.action_space['adjust_gen_v'].high, obs.action_space['adjust_gen_v'].low
        adjust_gen_v = np.zeros(54)
        for i in range(gen_num):
            if err[i] < 0:
                if - err[i] > action_high[i]:
                    adjust_gen_v[i] = action_high[i]
                else:
                    adjust_gen_v[i] = - err[i]
            elif err[i] > 0:
                if - err[i] < action_low[i]:
                    adjust_gen_v[i] = action_low[i]
                else:
                    adjust_gen_v[i] = - err[i]

    elif type == 'overflow':
        outline_ids = []
        for line_idx in range(settings.num_line):
            if not obs.line_status[line_idx]:
                outline_ids.append(line_idx)

        overflow_line_ids = []
        for line_idx in range(settings.num_line):
            if obs.rho[line_idx] > settings.soft_overflow_bound - 0.1:
                overflow_line_ids.append(line_idx)

        # if len(overflow_line_ids) > 0:
        #     import ipdb
        #     ipdb.set_trace()
        #     bus_connected_to_overflowline = {}
        #     for line_id in overflow_line_ids:
        #         line_name = settings.lnname[line_id]
        #         bus_connected_to_overflowline[line_name] = []
        #         line_or = line_name + '_or'
        #         line_ex = line_name + '_ex'
        #         find_or = False
        #         find_ex = False
        #         for bus_name in settings.busname:
        #             connectors = obs.bus_branch[bus_name]
        #             if line_or in connectors:
        #                 find_or = True
        #                 import ipdb
        #                 ipdb.set_trace()
        #                 bus_connected_to_overflowline[line_name].append(bus_name)
        #                 gen_connected_to_or = obs.bus_gen[bus_name]
        #
        #                 # if gen_connected_to_or[0] is not '':
        #                 #     gen_idx = settings.gen_name_list.index(gen_connected_to_or[0])
        #                 #     gen_connected_to_outline[outline_name]['or'] = gen_idx
        #             elif line_ex in connectors:
        #                 find_ex = True
        #                 import ipdb
        #                 ipdb.set_trace()
        #                 bus_connected_to_overflowline[line_name].append(bus_name)
        #                 gen_connected_to_ex = obs.bus_gen[bus_name]
        #
        #                 # if gen_connected_to_ex[0] is not '':
        #                 #     gen_idx = settings.gen_name_list.index(gen_connected_to_ex[0])
        #                 #     gen_connected_to_outline[outline_name]['ex'] = gen_idx
        #
        #             if find_ex and find_or:
        #                 break
        #
        # if len(outline_ids) > 0:
        #     gen_connected_to_outline = {}
        #     for outline_id in outline_ids:
        #         outline_name = settings.lnname[outline_id]
        #         outline_or = outline_name + '_or'
        #         outline_ex = outline_name + '_ex'
        #         find_or = False
        #         find_ex = False
        #         for bus_name in settings.busname:
        #             connectors = obs.bus_branch[bus_name]
        #             if outline_or in connectors:
        #                 find_or = True
        #                 gen_connected_to_or = obs.bus_gen[bus_name]
        #                 if gen_connected_to_or[0] is not '':
        #                     gen_idx = settings.gen_name_list.index(gen_connected_to_or[0])
        #                     gen_connected_to_outline[outline_name]['or'] = gen_idx
        #             elif outline_ex in connectors:
        #                 find_ex = True
        #                 gen_connected_to_ex = obs.bus_gen[bus_name]
        #                 if gen_connected_to_ex[0] is not '':
        #                     gen_idx = settings.gen_name_list.index(gen_connected_to_ex[0])
        #                     gen_connected_to_outline[outline_name]['ex'] = gen_idx
        #
        #             if find_ex and find_or:
        #                 break

            # import ipdb
            # ipdb.set_trace()
        # print(f'max rho={max(obs.rho)}')
        # print(f'overflow_lines={overflow_line_ids}')
        if len(overflow_line_ids) > 0:
            adjust_gen_v = obs.action_space['adjust_gen_v'].high
        elif len(overflow_line_ids) == 0:
            err = np.asarray(obs.gen_v) - 1  # restrict at stable point 1.05
            gen_num = len(err)
            action_high, action_low = obs.action_space['adjust_gen_v'].high, obs.action_space['adjust_gen_v'].low
            adjust_gen_v = np.zeros(54)
            for i in range(gen_num):
                if err[i] < 0:
                    if - err[i] > action_high[i]:
                        adjust_gen_v[i] = action_high[i]
                    else:
                        adjust_gen_v[i] = - err[i]
                elif err[i] > 0:
                    if - err[i] < action_low[i]:
                        adjust_gen_v[i] = action_low[i]
                    else:
                        adjust_gen_v[i] = - err[i]
        # print(obs.gen_v)
    elif type == 'reactive':
        adjust_num = 0
        err = False
        adjust_gen_v = np.zeros(settings.num_gen)
        bus_v = [0 for _ in range(len(settings.busname))]
        adjustable_gen_ids = [i for i in range(settings.num_gen)]
        # first do voltage overstep adjust, then do reactive power overstep adjust
        # import ipdb
        # ipdb.set_trace()
        for line_idx in range(settings.num_line):
            line = settings.lnname[line_idx]
            line_or_v = obs.v_or[line_idx]
            line_ex_v = obs.v_ex[line_idx]
            line_or = line + '_or'
            line_ex = line + '_ex'
            find_or = False
            find_ex = False
            for bus_idx in range(len(settings.busname)):
                bus = settings.busname[bus_idx]
                connectors = obs.bus_branch[bus]
                if line_or in connectors:
                    find_or = True
                    bus_v[bus_idx] = line_or_v / 100
                if line_ex in connectors:
                    find_ex = True
                    bus_v[bus_idx] = line_ex_v / 100
                if find_ex and find_or:
                    break

        for bus_idx in range(len(settings.busname)):
            bus = settings.busname[bus_idx]
            gens = obs.bus_gen[bus]
            if len(gens) > 0 and gens[0] != '':
                gen_idx = settings.gen_name_list.index(gens[0])
                bus_v[bus_idx] = obs.gen_v[gen_idx]
            loads = obs.bus_load[bus]
            if len(loads) > 0 and loads[0] != '':
                load_idx = settings.ldname.index(loads[0])
                bus_v[bus_idx] = obs.load_v[load_idx]
                # print(f'load_v={load_v}')

        upgrade_gen_v_ids = []
        downgrade_gen_v_ids = []
        for gen_id in range(settings.num_gen):
            if obs.gen_q[gen_id] < settings.min_gen_q[gen_id]:
                err = True
                upgrade_gen_v_ids.append(gen_id)
                adjustable_gen_ids.remove(gen_id)
                # print(
                #     f'reactive below gen:{settings.gen_name_list[gen_id]}, id={gen_id}, LB={settings.min_gen_q[gen_id]}, q={obs.gen_q[gen_id]}, action_v={adjust_gen_v[gen_id]}')
            elif obs.gen_q[gen_id] > settings.max_gen_q[gen_id]:
                err = True
                downgrade_gen_v_ids.append(gen_id)
                adjustable_gen_ids.remove(gen_id)
                # print(
                #     f'reactive over gen:{settings.gen_name_list[gen_id]}, id={gen_id}, UB={settings.max_gen_q[gen_id]}, q={obs.gen_q[gen_id]}, action_v={adjust_gen_v[gen_id]}')

        # if adjust_num == 0:
        #     adjust_num += 1
        if len(upgrade_gen_v_ids) > 0:
            adjust_gen_v[upgrade_gen_v_ids] += 0.01
        elif len(downgrade_gen_v_ids) > 0:
            adjust_gen_v[downgrade_gen_v_ids] -= 0.01
        adjust_gen_v = np.clip(adjust_gen_v, obs.action_space['adjust_gen_v'].low,
                                   obs.action_space['adjust_gen_v'].high)

        # gen2busInfM = gen_to_bus_influence_matrix(obs, settings)
        over_v_upbound_bus = []
        below_v_lowbound_bus = []
        for bus_idx in range(len(settings.busname)):
            if bus_v[bus_idx] < settings.min_bus_v[bus_idx]:
                err = True
                below_v_lowbound_bus.append(bus_idx)
                # print(f'V_lower_bus:{settings.busname[bus_idx]}, bus_id:{bus_idx}, lower_bound={settings.min_bus_v[bus_idx]}, V={bus_v[bus_idx]}, gen_connected={obs.bus_gen[settings.busname[bus_idx]]}')
            elif bus_v[bus_idx] > settings.max_bus_v[bus_idx]:
                err = True
                over_v_upbound_bus.append(bus_idx)
                # print(
                #     f'V_lower_bus:{settings.busname[bus_idx]}, bus_id:{bus_idx}, upper_bound={settings.max_bus_v[bus_idx]}, V={bus_v[bus_idx]}, gen_connected={obs.bus_gen[settings.busname[bus_idx]]}')
        if adjust_num == 0:
            adjust_num += 1
            if len(over_v_upbound_bus) > 0:
                # adjust_gen_v -= 0.011
                err = np.asarray(obs.gen_v) - 1  # restrict at stable point 1.05
                gen_num = len(err)
                action_high, action_low = obs.action_space['adjust_gen_v'].high, obs.action_space['adjust_gen_v'].low
                adjust_gen_v = np.zeros(54)
                for i in range(gen_num):
                    if err[i] < 0:
                        if - err[i] > action_high[i]:
                            adjust_gen_v[i] = action_high[i]
                        else:
                            adjust_gen_v[i] = - err[i]
                    elif err[i] > 0:
                        if - err[i] < action_low[i]:
                            adjust_gen_v[i] = action_low[i]
                        else:
                            adjust_gen_v[i] = - err[i]
            elif len(below_v_lowbound_bus) > 0:
                # adjust_gen_v += 0.011
                err = np.asarray(obs.gen_v) - 1.04  # restrict at stable point 1.05
                gen_num = len(err)
                action_high, action_low = obs.action_space['adjust_gen_v'].high, obs.action_space['adjust_gen_v'].low
                adjust_gen_v = np.zeros(54)
                for i in range(gen_num):
                    if err[i] < 0:
                        if - err[i] > action_high[i]:
                            adjust_gen_v[i] = action_high[i]
                        else:
                            adjust_gen_v[i] = - err[i]
                    elif err[i] > 0:
                        if - err[i] < action_low[i]:
                            adjust_gen_v[i] = action_low[i]
                        else:
                            adjust_gen_v[i] = - err[i]
            else:
                err = np.asarray(obs.gen_v) - 1.02  # restrict at stable point 1.05
                gen_num = len(err)
                action_high, action_low = obs.action_space['adjust_gen_v'].high, obs.action_space['adjust_gen_v'].low
                adjust_gen_v = np.zeros(54)
                for i in range(gen_num):
                    if err[i] < 0:
                        if - err[i] > action_high[i]:
                            adjust_gen_v[i] = action_high[i]
                        else:
                            adjust_gen_v[i] = - err[i]
                    elif err[i] > 0:
                        if - err[i] < action_low[i]:
                            adjust_gen_v[i] = action_low[i]
                        else:
                            adjust_gen_v[i] = - err[i]
            adjust_gen_v = np.clip(adjust_gen_v, obs.action_space['adjust_gen_v'].low,
                                       obs.action_space['adjust_gen_v'].high)

        # if err:
        #     import ipdb
        #     ipdb.set_trace()
        #     print('false')
    return adjust_gen_v

def get_action_space(obs, parameters, settings):
    if parameters['only_power']:
        if parameters['only_thermal']:
            action_high = obs.action_space['adjust_gen_p'].high[settings.thermal_ids]
            action_low = obs.action_space['adjust_gen_p'].low[settings.thermal_ids]
        else:
            action_high = obs.action_space['adjust_gen_p'].high
            action_high = np.where(np.isinf(action_high), np.full_like(action_high, 0),
                               action_high)  # feed 0 to balanced generator threshold
            action_low = obs.action_space['adjust_gen_p'].low
            action_low = np.where(np.isinf(action_low), np.full_like(action_low, 0),
                              action_low)  # feed 0 to balanced generator threshold
    else:
        action_high = np.asarray(
            [obs.action_space['adjust_gen_p'].high, obs.action_space['adjust_gen_v'].high]).flatten()
        action_high = np.where(np.isinf(action_high), np.full_like(action_high, 0),
                               action_high)  # feed 0 to balanced generator threshold
        action_low = np.asarray([obs.action_space['adjust_gen_p'].low, obs.action_space['adjust_gen_v'].low]).flatten()
        action_low = np.where(np.isinf(action_low), np.full_like(action_low, 0),
                              action_low)  # feed 0 to balanced generator threshold
    return action_high, action_low

def get_state_from_obs(obs, settings):
    state_form = {'gen_p', 'gen_q', 'gen_v', 'target_dispatch', 'actual_dispatch', 'load_p', 'load_q', 'load_v', 'rho'}
    state = []
    for name in state_form:
        value = getattr(obs, name)
        value = preprocessing.minmax_scale(np.array(value, dtype=np.float32), feature_range=(0, 1), axis=0, copy=True)
        state.append(np.reshape(np.array(value, dtype=np.float32), (-1,)))
    state = np.concatenate(state)
    return state


class RuleAgent(BaseAgent):
    def __init__(
            self,
            settings,
            this_directory_path=None
        ):

        BaseAgent.__init__(self, settings.num_gen)

        self.settings = settings
        self.cnt = 0

        self.original_bus_branch = {}
        cc = np.load(this_directory_path+'/ori_bus_branch.npy', allow_pickle=True).tolist()
        for key, value in cc.items():
            self.original_bus_branch[key] = value

        self.last_p_or = np.zeros(settings.num_line)
        self.last_p_ex = np.zeros(settings.num_line)
        self.last_rho = np.zeros(settings.num_line)

    def act(self, obs, reward=0.0, done=False):

        adjust_gen_p, restart_flag = adjust_generator_p(obs, self.settings, self.original_bus_branch, self.last_p_or, self.last_p_ex, self.last_rho)
        adjust_gen_v = voltage_action(obs, self.settings)

        self.last_p_or = obs.p_or
        self.last_p_ex = obs.p_ex
        self.last_rho = obs.rho

        return form_action(adjust_gen_p, adjust_gen_v), restart_flag


