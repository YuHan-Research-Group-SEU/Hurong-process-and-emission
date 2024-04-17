import json
import pickle
import random
import numpy as np
import pandas as pd
import os
from numpy import *
import matplotlib.pyplot as plt
#import seaborn as sns
from multiprocessing.pool import Pool
import math
from matplotlib.collections import LineCollection


def find_lc_vehicles(vehicle_property, mainstream_lane_num, pos_lists, id_lists, speed, LC_speed_threshold1,
                     LC_speed_threshold2, merge_position, new_vehicles, time_idx, PUNN_info):
    lc_vehicles = {}

    for lane in range(len(id_lists)):  # 对每个车道循环
        for veh in id_lists[lane]:  # 对每辆车循环
            if veh not in PUNN_info.keys():
                PUNN_info[veh] = {'time_index': [], 'info': []}
            len_lane_record_veh = len(lane_record[veh])
            if veh not in new_vehicles and lane_record[veh][-1] == lane_record[veh][
                -min(int(3 / discrete_interval), len_lane_record_veh - 1)]:  # 避免短时间出现频繁换道的错误
                idx = id_lists[lane].index(veh)  # 车辆id的index
                pos = pos_lists[lane][idx]  # 记录车辆最新位置
                potential_lanes = []

                if lane in vehicle_property[veh][
                    "destination_lanes"] or pos < exit_position - 200:  # 已经在目标车道或汇入点600前 换到风格
                    lane_change_type = "free"  # 自由换道
                    thresh = LC_speed_threshold1 / 3.6
                else:
                    lane_change_type = "semi_free"  # 半自由换道
                    thresh = LC_speed_threshold2 / 3.6

                for other_lane in range(mainstream_lane_num):  # 记录相邻车道号
                    if other_lane != lane and abs(other_lane - lane) == 1:
                        potential_lanes.append(other_lane)

                for changeable_lane in potential_lanes:  # 对相邻的车道进行循环
                    for pos_idx in range(1, len(pos_lists[changeable_lane]) - 1):  # 对相邻车道的位置进行循环
                        if pos_lists[changeable_lane][pos_idx - 1] < pos and pos_lists[changeable_lane][pos_idx] > pos:  # 如果相邻车道有前车位置大于本车，后车位置小于本车
                            new_leader = id_lists[changeable_lane][pos_idx]  # 新前车
                            new_follower = id_lists[changeable_lane][pos_idx - 1]  # 新后车
                            leader_leader = id_lists[changeable_lane][pos_idx + 1]  # 新前车的前车
                            v_new_leader = speed[new_leader][-1]
                            v_leader_leader = speed[leader_leader][-1]
                            v_follower = speed[new_follower][-1]
                            s_lateral = pos_lists[changeable_lane][pos_idx] - pos_lists[changeable_lane][
                                pos_idx - 1]  # s_l目标车道前车位置-目标车道后车位置
                            s_eq = get_equilibrium_spacing(speed[veh][-1], vehicle_property, veh)  # s_eq平衡间距 输入本车速度、所有车的速度信息、本车id
                            # -----------------------------
                            if idx < len(id_lists[lane]) - 1:
                                k_l = 1 / (pos_lists[lane][idx + 1] - pos_lists[lane][idx])  # 本车道密度
                            else:
                                k_l = 1 / s_eq  # 无前车
                            k_l1 = 1 / s_lateral  # 目标车道密度
                            if speed[veh][-1] < average_parameters["cri_speed"]:
                                tao = average_parameters["tao_1"]
                            else:
                                tao = average_parameters["tao_2"]
                            if (v_new_leader + v_leader_leader) / 2 < average_parameters["cri_speed"]:
                                tao_1 = average_parameters["tao_1"]
                            else:
                                tao_1 = average_parameters["tao_2"]
                            Q = vehicle_property[veh]['free_speed'] / (
                                        tao * vehicle_property[veh]['free_speed'] + vehicle_property[veh][
                                    "jam_spacing"])
                            Q_1 = vehicle_property[new_follower]['free_speed'] / (
                                        tao_1 * vehicle_property[new_follower]['free_speed'] +
                                        vehicle_property[new_follower]["jam_spacing"])
                            lambda_kl = min(vehicle_property[veh]['free_speed'] * k_l, Q)
                            lambda_kl1 = min(vehicle_property[new_follower]['free_speed'] * k_l1, Q_1)
                            μ_kl1 = min(max(0, (1 / vehicle_property[new_follower]['jam_spacing'] - k_l1)) * (
                                        1 * vehicle_property[new_follower]['jam_spacing'] / (discrete_interval)), Q_1)
                            tau = 0.20  # 需要标定的参数
                            if lane_change_type == "free":
                                pai_ll1 = max((v_new_leader + v_leader_leader) / 2 - speed[veh][-1], 0) / (
                                            tau * vehicle_property[veh]['free_speed'])
                            if lane_change_type == "semi-free":
                                item_1 = max(((v_new_leader + v_leader_leader) / 2 - speed[veh][-1]) / (
                                            tau * vehicle_property[veh]['free_speed']), 0)
                                item_2 = 1 / discrete_interval
                                pai_ll1 = (exit_position - pos) / weaving_area * item_1 + (
                                            pos - (exit_position - weaving_area)) / weaving_area * item_2
                            P_ll1 = max(0, min(1, (
                                        min(1, μ_kl1 / lambda_kl1) * pai_ll1 * lambda_kl / vehicle_property[veh][
                                    'free_speed']) * discrete_interval * (1 / k_l)))
                            epsilon = random.rand()
                            if epsilon < P_ll1 and pos_lists[lane][idx] - pos_lists[changeable_lane][pos_idx - 1] > \
                                    vehicle_property[veh]['jam_spacing'] and pos_lists[changeable_lane][pos_idx] - \
                                    pos_lists[lane][idx] > vehicle_property[veh]['jam_spacing']:
                                action = False
                                if veh not in lc_vehicles:
                                    action = True
                                else:
                                    if P_ll1 > lc_vehicles[veh]["P_ll1_gain"] or lane_change_type == 'semi_free':
                                        action = True
                                if action == True:
                                    lc_vehicles[veh] = {}
                                    lc_vehicles[veh]["current_lane"] = lane
                                    lc_vehicles[veh]["lane_change_type"] = lane_change_type
                                    lc_vehicles[veh]["target_lane"] = changeable_lane
                                    lc_vehicles[veh]["speed_gain"] = (v_new_leader + v_leader_leader) / 2 - speed[veh][
                                        -1]
                                    lc_vehicles[veh]["epsilon"] = epsilon
                                    lc_vehicles[veh]["P_ll1_gain"] = P_ll1
                                    lc_vehicles[veh]["change_pos"] = pos
                                    lc_vehicles[veh]["change_time"] = time_idx * discrete_interval
                                    lc_vehicles[veh]['OD'] = vehicle_property[veh]["OD"]

                            PUNN_info[veh]['time_index'].append(time_idx)
                            PUNN_info[veh]['info'].append(
                                [lane, changeable_lane, lane_change_type, vehicle_property[veh]["OD"], pos, s_eq,
                                 s_lateral, round(1 / k_l, 2), (v_new_leader + v_leader_leader) / 2, speed[veh][-1],
                                 P_ll1])  # [车道，目标车道，换到风格，OD，位置，平衡间距，目标车道间距，本车道间距，目标车道速度，本车道速度,换道概率]
    return lc_vehicles


def get_equilibrium_spacing(v, vehicle_property, veh):
    if v <= vehicle_property[veh]["cri_spacing1"]:
        desired_spacing = vehicle_property[veh]["jam_spacing"] + 1 / (vehicle_property[veh]["cri_speed"] / (
                    vehicle_property[veh]["cri_spacing1"] - vehicle_property[veh]["jam_spacing"])) * v
    else:
        # desired_spacing =  1/((vehicle_property[veh]["free_speed"] - vehicle_property[veh]["cri_speed"]) / (vehicle_property[veh]["cri_spacing2"] - vehicle_property[veh]["cri_spacing1"])) * v
        desired_spacing = vehicle_property[veh]["jam_spacing"] + 1 / (vehicle_property[veh]["cri_speed"] / (
                    vehicle_property[veh]["cri_spacing1"] - vehicle_property[veh]["jam_spacing"])) * v
    return desired_spacing


def CF(vehicle_id, predecessor_id, position, speed, spacing, acceleration, lane_record, discrete_interval,
       vehicle_property, eta_all):
    if vehicle_id not in eta_all:
        eta_all[vehicle_id] = []
    if predecessor_id == "none":  # 前车id    目前加速度没有限制，暂且都记为0
        v_new = vehicle_property[vehicle_id]['free_speed']
        pos = position[vehicle_id][-1] + v_new * discrete_interval
        acc = 0
    else:
        s = spacing[vehicle_id][-1]
        ####free-flow term
        x = position[vehicle_id][-1]  # 上一时刻的位置
        v = speed[vehicle_id][-1]  # 上一时刻的速度
        x_max = position[predecessor_id][-1] - vehicle_property[vehicle_id]['jam_spacing']
        gG = 9.8 * 0
        a_max = vehicle_property[vehicle_id]['maximum_acc']
        v_c = int(vehicle_property[vehicle_id]['free_speed'] * (1 - gG / a_max))
        if v < average_parameters["cri_speed"]:
            tao = average_parameters["tao_1"]
        else:
            tao = average_parameters["tao_2"]
        if int(tao / discrete_interval) < len(speed[vehicle_id]) - 1 and int(tao / discrete_interval) < len(
                speed[predecessor_id]) - 1:  # 因为要用到tao时刻前的一些信息
            # free_flow_term
            v_t_tao = speed[vehicle_id][-(int(tao / discrete_interval))]
            x_leader = position[predecessor_id][-(int(tao / discrete_interval))]  # 前车上一tao时刻的位置
            x_hat = v_c * tao - (1 - math.exp(-tao * a_max / vehicle_property[vehicle_id]['free_speed'])) * (
                        v_c - v_t_tao)
            x_tao = position[vehicle_id][-(int(tao / discrete_interval))]
            free_flow_term = x_tao + min(vehicle_property[vehicle_id]['free_speed'] * tao, x_hat)
            # congestion_term
            v_leader = speed[predecessor_id][-1]
            v_leader_2 = speed[predecessor_id][-2]
            alpha = 0.4  # 决定η的范围
            epsilon_ = 0.01  # epsilon_*s_v约等于1km/h
            s_eq = get_equilibrium_spacing(v, vehicle_property, vehicle_id)
            s_v = get_equilibrium_spacing(v_leader, vehicle_property, predecessor_id)
            eta = 1
            if vehicle_property[vehicle_id]['drive_style'] == 'timid':
                if v_leader_2 - v_leader > 0.1 and eta_all[vehicle_id] == []:  # 前车减速
                    epsilon = epsilon_
                    eta = max(1, s / s_v) + epsilon * tao
                    eta = min(max(eta, 1), 1 + alpha)
                    eta_all[vehicle_id].append(eta)
                if eta_all[vehicle_id] != []:
                    if max(eta_all[vehicle_id]) < 1 + alpha:
                        epsilon = epsilon_
                        eta = max(1, s / s_v) + epsilon * tao
                        eta = min(max(eta, 1), 1 + alpha)
                    else:
                        epsilon = -epsilon_
                        eta = s / s_v + epsilon * tao
                        eta = max(min(eta, 1 + alpha), 1)
                    eta_all[vehicle_id].append(eta)
                    if max(eta_all[vehicle_id]) >= 1 + alpha and 0.98 < eta <= 1:
                        eta_all[vehicle_id] == []
            if vehicle_property[vehicle_id]['drive_style'] == 'aggressive':
                if v_leader_2 - v_leader > 0.1 and eta_all[vehicle_id] == []:  # 前车减速
                    epsilon = -epsilon_
                    eta = min(s / s_v, 1) + epsilon * tao
                    eta = max(min(eta, 1), 1 - alpha)
                    eta_all[vehicle_id].append(eta)
                if eta_all[vehicle_id] != []:
                    if min(eta_all[vehicle_id]) > 1 - alpha:
                        epsilon = -epsilon_
                        eta = min(s / s_v, 1) + epsilon * tao
                        eta = max(min(eta, 1), 1 - alpha)
                    else:
                        epsilon = epsilon_
                        eta = s / s_v + epsilon * tao
                        eta = min(max(eta, 1 - alpha), 1)
                    eta_all[vehicle_id].append(eta)
                    if min(eta_all[vehicle_id]) <= 1 - alpha and 0.98 < eta <= 1:
                        eta_all[vehicle_id] == []
            congestion_term = x_leader + tao * v_leader - eta * s_v
            x_time = max(x, min(free_flow_term, congestion_term, x_max,
                                x + vehicle_property[vehicle_id]['free_speed'] * discrete_interval))
            v_new = speed[predecessor_id][-(int(tao / discrete_interval))]
            if x_time != free_flow_term:
                v_new = max(0, min((x_time - x) / discrete_interval, vehicle_property[vehicle_id]['free_speed']))
            if x_time == congestion_term:  # 在拥堵的情况下，速度的传播递减
                s_v_leader_tao = get_equilibrium_spacing(speed[predecessor_id][-(int(tao / discrete_interval))],
                                                         vehicle_property, predecessor_id)
                v_new = max(0, speed[predecessor_id][-(int(tao / discrete_interval))] - epsilon_ * s_v_leader_tao)
            if lane_record[vehicle_id][-1] != lane_record[vehicle_id][
                -10]:  # 前车变化相关情况速度的处理。eg:论文没有说明换道车辆换道后速度的处理，已经不能用前车tau时刻前的速度
                v_new = max(0, (x_time - x) / discrete_interval)
            pos = None
            if v_leader > v_leader_2 and eta_all[vehicle_id] == []:  # 前车加速
                if s < s_v + vehicle_property[vehicle_id]["extension"]:
                    v_new = speed[vehicle_id][-1]
                    pos = x + v_new * discrete_interval
            if pos == None:
                pos = x_time
            acc = 0
        else:  # 车辆刚进入路段信息较少时的处理
            if int(tao / discrete_interval) < len(speed[predecessor_id]) - 1:
                pos = max(x, min(x + vehicle_property[vehicle_id]['free_speed'] * discrete_interval,
                                 position[predecessor_id][-(int(tao / discrete_interval))] - 7))
                v_new = speed[predecessor_id][-(int(tao / discrete_interval))]
                acc = 0
            else:
                s_eq = get_equilibrium_spacing(v, vehicle_property, vehicle_id)
                if s > s_eq:
                    acc = min(vehicle_property[vehicle_id]['maximum_acc'],
                              (vehicle_property[vehicle_id]['free_speed'] - speed[vehicle_id][-1]) / discrete_interval)
                    v_new = speed[vehicle_id][-1] + acc * discrete_interval
                    pos = min(position[vehicle_id][-1] + (speed[vehicle_id][-1] + v_new) / 2 * discrete_interval, x_max)
                if s < s_eq:
                    acc = max(vehicle_property[vehicle_id]['maximum_dcc'],
                              (0 - speed[vehicle_id][-1]) / discrete_interval)
                    v_new = speed[vehicle_id][-1] + acc * discrete_interval
                    pos = max(x, min(position[vehicle_id][-1] + (speed[vehicle_id][-1] + v_new) / 2 * discrete_interval,
                                     x_max))
                    v_new = (pos - x) / discrete_interval
                if s_eq == s:
                    acc = 0
                    v_new = speed[vehicle_id][-1]
                    pos = position[vehicle_id][-1] + v_new * discrete_interval
    return acc, v_new, pos


def vehicle_generation(mainstream_arrival_flow, ramp_arrival_flow):  # 车辆均衡到达的时间

    mainstream_arrival_headway = int(36000 / mainstream_arrival_flow) / 10
    ramp_arrival_headway = 3600 / ramp_arrival_flow
    mainstream_arrival_time = [round(mainstream_arrival_headway * i, 1) for i in range(mainstream_arrival_flow)]
    ramp_arrival_time = [round(ramp_arrival_headway * i, 1) for i in range(1, ramp_arrival_flow)]

    return mainstream_arrival_time, ramp_arrival_time


def get_random_vehicle_property(average_parameters, veh_id, vehicle_property, reaction_time_extension_list, OD_ratio,
                                origin, drive_style_ratio, time_idx):
    vehicle_property[veh_id] = {}
    epsilon = random.rand()
    if origin == "main":
        if epsilon < OD_ratio["o1_d1"]:
            vehicle_property[veh_id]["destination"] = 1
            vehicle_property[veh_id]["destination_lanes"] = [0, 1]
            OD = "o1_d1"
        else:
            vehicle_property[veh_id]["destination"] = 2
            vehicle_property[veh_id]["destination_lanes"] = [2]
            OD = "o1_d2"
    if origin == "ramp":
        if epsilon < OD_ratio["o2_d1"]:
            vehicle_property[veh_id]["destination"] = 1
            vehicle_property[veh_id]["destination_lanes"] = [0, 1]
            OD = "o2_d1"
        else:
            vehicle_property[veh_id]["destination"] = 2
            vehicle_property[veh_id]["destination_lanes"] = [2]
            OD = "o2_d2"
    if epsilon < drive_style_ratio['normal']:
        vehicle_property[veh_id]["drive_style"] = 'normal'
    if drive_style_ratio['normal'] <= epsilon < drive_style_ratio['normal'] + drive_style_ratio['timid']:
        vehicle_property[veh_id]["drive_style"] = 'timid'
    if epsilon >= drive_style_ratio['normal'] + drive_style_ratio['timid']:
        vehicle_property[veh_id]["drive_style"] = 'aggressive'
    # if 3000<time_idx<6000:
    #     vehicle_property[veh_id]["drive_style"] = 'aggressive'
    vehicle_property[veh_id]["jam_spacing"] = np.random.normal(average_parameters["jam_spacing"],
                                                               0.05 * average_parameters["jam_spacing"])
    vehicle_property[veh_id]["cri_spacing1"] = np.random.normal(average_parameters["cri_spacing1"],
                                                                0.05 * average_parameters["cri_spacing1"])
    vehicle_property[veh_id]["cri_spacing2"] = np.random.normal(average_parameters["cri_spacing2"],
                                                                0.05 * average_parameters["cri_spacing2"])
    vehicle_property[veh_id]["jam_spacing"] = np.random.normal(average_parameters["jam_spacing"],
                                                               0.05 * average_parameters["jam_spacing"])
    vehicle_property[veh_id]["cri_speed"] = np.random.normal(average_parameters["cri_speed"],
                                                             0.05 * average_parameters["cri_speed"])
    vehicle_property[veh_id]["free_speed"] = average_parameters["free_speed"]
    vehicle_property[veh_id]["dcc"] = average_parameters["dcc"]
    vehicle_property[veh_id]["maximum_dcc"] = average_parameters["maximum_dcc"]
    vehicle_property[veh_id]["maximum_acc"] = np.random.normal(average_parameters["maximum_acc"],
                                                               0.05 * average_parameters["maximum_acc"])
    reaction_time_extension = random.choice(reaction_time_extension_list)  # 反应延迟时间
    extended_spacing = 0.5 * vehicle_property[veh_id][
        "maximum_acc"] * reaction_time_extension * reaction_time_extension  # 1/2 a t**2
    normal_reaction_time = (vehicle_property[veh_id]["cri_spacing1"] - vehicle_property[veh_id]["jam_spacing"]) / \
                           vehicle_property[veh_id]["cri_speed"]
    vehicle_property[veh_id]["normal_extension"] = 0.5 * vehicle_property[veh_id][
        "maximum_acc"] * normal_reaction_time * normal_reaction_time
    vehicle_property[veh_id]["extension"] = extended_spacing
    vehicle_property[veh_id]["OD"] = OD
    vehicle_property[veh_id]["begin_frame_index"] = time_idx
    vehicle_property[veh_id]["origin"] = origin
    return vehicle_property


def find_leader(id_lists, pos_lists):
    leaders = {}
    sorted_id_lists = {}
    sorted_pos_lists = {}
    spacing = {}
    for idx in range(len(pos_lists)):
        sorted_pairs = sorted(zip(pos_lists[idx], id_lists[idx]))
        list1 = [v1 for v1, v2 in sorted_pairs]
        list2 = [v2 for v1, v2 in sorted_pairs]
        sorted_id_lists[idx] = list2
        sorted_pos_lists[idx] = list1

    for idx in range(len(sorted_id_lists)):
        for id in sorted_id_lists[idx]:
            if sorted_id_lists[idx].index(id) != len(sorted_id_lists[idx]) - 1:
                leader = sorted_id_lists[idx][sorted_id_lists[idx].index(id) + 1]
                leaders[id] = leader
                spacing[id] = sorted_pos_lists[idx][sorted_id_lists[idx].index(id) + 1] - sorted_pos_lists[idx][
                    sorted_id_lists[idx].index(id)]
            else:
                leaders[id] = "none"
                spacing[id] = "none"

    return sorted_pos_lists, sorted_id_lists, leaders, spacing


def state_update(position, speed, acceleration, lane_record, spacing, pos_time, speed_time, acceleration_time,
                 lane_time, spacing_time, id_lists, new_vehicles):
    for idx in range(len(id_lists)):
        for veh in id_lists[idx]:
            if veh not in new_vehicles:
                position[veh].append(pos_time[veh])
                speed[veh].append(speed_time[veh])
                acceleration[veh].append(acceleration_time[veh])
                lane_record[veh].append(lane_time[veh])
                spacing[veh].append(spacing_time[veh])
            else:
                spacing[veh].append(spacing_time[veh])

    return position, speed, acceleration, lane_record, spacing


def find_merging_follower(pos_lists, id_lists, merge_position, mainstream_lane_num):
    merge_follower = id_lists[mainstream_lane_num - 1][-2]
    merge_spacing_follower = merge_position - pos_lists[mainstream_lane_num - 1][-2]
    merge_leader = id_lists[mainstream_lane_num - 1][-1]
    merge_spacing_leader = pos_lists[mainstream_lane_num - 1][-1] - merge_position

    for idx in range(len(pos_lists[mainstream_lane_num - 1]) - 1):
        if pos_lists[mainstream_lane_num - 1][idx] < merge_position and pos_lists[mainstream_lane_num - 1][
            idx + 1] > merge_position:
            merge_follower = id_lists[mainstream_lane_num - 1][idx]
            merge_spacing_follower = merge_position - pos_lists[mainstream_lane_num - 1][idx]
            merge_leader = id_lists[mainstream_lane_num - 1][idx + 1]
            merge_spacing_leader = pos_lists[mainstream_lane_num - 1][idx + 1] - merge_position
    return merge_leader, merge_follower, merge_spacing_leader, merge_spacing_follower


def traffic_dynamics(discrete_interval, position, speed, acceleration, lane_record, spacing, vehicle_property,
                     reaction_time_extension_list):
    lc_vehicles_all = {}  # 记录所有变道的车辆
    all_time_pos = {}  # 记录所有车辆时间位置信息，判断流率
    eta_all = {}  # 存储在前车减速过程中的eta变化情况
    PUNN_info = {}  # 存储神经网络所需要的相关信息
    num_steps = int(simulation_duration / discrete_interval)
    mainstream_arrival_time, ramp_arrival_time = vehicle_generation(mainstream_arrival_flow, ramp_arrival_flow)
    veh_id = 0
    finished_vehicles = []
    accumulate_time = []
    leaders = {}
    id_lists = {}
    pos_lists = {}
    for lane_idx in range(mainstream_lane_num):
        id_lists[lane_idx] = []  #
        pos_lists[lane_idx] = []

    for time_idx in range(num_steps):
        print(time_idx)
        if time_idx == 0:
            for lane in range(mainstream_lane_num):
                vehicle_property = get_random_vehicle_property(average_parameters, veh_id, vehicle_property,
                                                               reaction_time_extension_list, OD_ratio, "main",
                                                               drive_style_ratio, time_idx)
                position[veh_id] = [0]
                acceleration[veh_id] = [0]
                speed[veh_id] = [vehicle_property[veh_id]["free_speed"]]  # 初始速度变化
                lane_record[veh_id] = [lane]
                spacing[veh_id] = ["none"]
                veh_id += 1

        else:
            new_vehicles = []
            if round(time_idx * discrete_interval, 1) in mainstream_arrival_time:
                for lane in range(mainstream_lane_num):
                    vehicle_property = get_random_vehicle_property(average_parameters, veh_id, vehicle_property,
                                                                   reaction_time_extension_list, OD_ratio, "main",
                                                                   drive_style_ratio, time_idx)  # 此处修改了代码，将ramp改为main
                    position[veh_id] = [0]
                    acceleration[veh_id] = [0]
                    speed[veh_id] = [vehicle_property[veh_id]["free_speed"]]
                    lane_record[veh_id] = [lane]
                    spacing[veh_id] = []
                    new_vehicles.append(veh_id)
                    veh_id += 1

            if round(time_idx * discrete_interval, 1) in ramp_arrival_time:  ###有问题，如果距离后车的距离极小-0

                vehicle_property = get_random_vehicle_property(average_parameters, veh_id, vehicle_property,
                                                               reaction_time_extension_list, OD_ratio, "ramp",
                                                               drive_style_ratio, time_idx)
                merge_leader, merge_follower, merge_spacing_leader, merge_spacing_follower = find_merging_follower(
                    pos_lists, id_lists, merge_position, mainstream_lane_num)
                position[veh_id] = [merge_position]
                acceleration[veh_id] = [0]
                if merge_spacing_leader > average_parameters["cri_spacing2"] and merge_spacing_follower > \
                        average_parameters["cri_spacing2"]:
                    merge_speed = average_parameters["free_speed"]
                    speed[veh_id] = [merge_speed]
                else:
                    if merge_spacing_leader <= average_parameters["cri_spacing2"] and merge_spacing_follower > \
                            average_parameters["cri_spacing2"]:
                        merge_speed = max(0, speed[merge_leader][-1] - merge_speed_reduction)
                        speed[veh_id] = [merge_speed]
                    if merge_spacing_leader > average_parameters["cri_spacing2"] and merge_spacing_follower <= \
                            average_parameters["cri_spacing2"]:
                        merge_speed = max(0, speed[merge_follower][-1] - merge_speed_reduction)
                        speed[veh_id] = [merge_speed]
                    if merge_spacing_leader <= average_parameters["cri_spacing2"] and merge_spacing_follower <= \
                            average_parameters["cri_spacing2"]:
                        merge_speed = (speed[merge_follower][-1] + speed[merge_leader][-1]) / 2
                        speed[veh_id] = [merge_speed]
                speed[veh_id] = [merge_speed]
                lane_record[veh_id] = [mainstream_lane_num - 1]
                spacing[veh_id] = []
                new_vehicles.append(veh_id)
                veh_id += 1

            lc_vehicles = find_lc_vehicles(vehicle_property, mainstream_lane_num, pos_lists, id_lists, speed,
                                           LC_speed_threshold1, LC_speed_threshold2, merge_position, new_vehicles,
                                           time_idx, PUNN_info)
            for key in lc_vehicles.keys():
                if key not in lc_vehicles_all:
                    lc_vehicles_all[key] = []
                lc_vehicles_all[key].append(lc_vehicles[key])
            pos_time = {}  # 每辆车最新的位置
            speed_time = {}
            acceleration_time = {}
            lane_time = {}
            id_lists = {}
            pos_lists = {}
            for lane_idx in range(mainstream_lane_num):
                id_lists[lane_idx] = []
                pos_lists[lane_idx] = []

            for veh in position:
                if veh not in finished_vehicles and veh not in new_vehicles:
                    if veh in lc_vehicles:

                        lane_time[veh] = lc_vehicles[veh]["target_lane"]
                        acceleration_time[veh] = 0
                        speed_time[veh] = speed[veh][-1]
                        pos = position[veh][-1] + speed[veh][-1] * discrete_interval
                        pos_time[veh] = pos
                        lane_id = lc_vehicles[veh]["target_lane"]
                        # print(veh,time_idx,lane_record[veh][-1],lane_time[veh])
                    else:
                        lane_time[veh] = lane_record[veh][-1]
                        if veh in leaders:
                            leader_id = leaders[veh]
                        else:
                            leader_id = "none"

                        acc, v, pos = CF(veh, leader_id, position, speed, spacing, acceleration, lane_record,
                                         discrete_interval, vehicle_property, eta_all)
                        acceleration_time[veh] = acc
                        speed_time[veh] = v
                        pos_time[veh] = pos
                        lane_id = lane_record[veh][-1]
                    if pos > mainstream_length:
                        finished_vehicles.append(veh)
                        accumulate_time.append(time_idx)
                    id_lists[lane_id].append(veh)
                    pos_lists[lane_id].append(pos)
                    if veh not in all_time_pos:
                        all_time_pos[veh] = {'time': [], 'pos': []}
                    all_time_pos[veh]['time'].append(time_idx)
                    all_time_pos[veh]['pos'].append(pos)
            for veh in new_vehicles:
                lane_id = lane_record[veh][-1]
                pos = position[veh][-1]
                id_lists[lane_id].append(veh)
                pos_lists[lane_id].append(pos)

            pos_lists, id_lists, leaders, spacing_time = find_leader(id_lists, pos_lists)
            position, speed, acceleration, lane_record, spacing = state_update(position, speed, acceleration,
                                                                               lane_record, spacing, pos_time,
                                                                               speed_time, acceleration_time, lane_time,
                                                                               spacing_time, id_lists, new_vehicles)

    # 准备绘图统计信息
    total_info = {}
    total_info['position'] = position
    total_info['speed'] = speed
    total_info['lane_record'] = lane_record
    total_info['vehicle_property'] = vehicle_property
    accumulate_curve = {"finished_vehicles": finished_vehicles, 'accumulate_time': accumulate_time}
    return total_info, accumulate_curve, all_time_pos, lc_vehicles_all


def lane_changing_count(lc_vehicle, count_time_period):
    count_all = 0
    count_1 = 0
    count_2 = 0
    time_begin = count_time_period[0]
    time_end = count_time_period[1]
    for key, value in lc_vehicle.items():
        judge = True
        for i in range(0, len(value)):
            if time_begin < value[i]['change_time'] < time_end:
                if judge == True:
                    count_all = count_all + 1
                    judge = False
                if value[i]["lane_change_type"] == "free":
                    count_1 = count_1 + 1
                if value[i]["lane_change_type"] == "semi_free":
                    count_2 = count_2 + 1
    print('换道次数:', '自由换道', count_1, '强制换道', count_2, '换道车辆', count_all)


def flow_rate_count(all_time_pos, count_pos, count_time_period):
    flow_count = 0
    time_begin = count_time_period[0]
    time_end = count_time_period[1]
    for key, value in all_time_pos.items():
        if value['pos'][0] < count_pos and value['pos'][-1] > count_pos:
            if value['time'][-1] < time_begin / discrete_interval or value['time'][0] > time_end / discrete_interval:
                continue
            else:
                flow_count = flow_count + 1
    print("五分钟的流量", flow_count, '时段总流量', len(all_time_pos))


def smoothWithsEMA(lsdata, T, dt=0.1):
    """
    平滑数据使用对称指数移动平均法
    :return:
    """
    Na = len(lsdata)
    deta = T / dt
    outData = []
    for i in range(Na):
        D = min([3 * deta, i, Na - i - 1])
        lsgt = []
        lsxe = []
        for k in range(int(i - D), int(i + D + 1)):
            gt = pow(np.e, -abs(i - k) / deta)
            xe = lsdata[k] * gt
            lsgt.append(gt)
            lsxe.append(xe)
        outX = sum(lsxe) / sum(lsgt)
        outData.append(outX)
    return outData


def plot_all_cumulative_curves(all_cum_num_ls, save_folder, color_dict, slanted_cumulative_curve=True, capacity=1500):
    plt.figure(figsize=(16, 12))
    lane_num = 3
    min_start_unix_time = min([x[1][0] for x in all_cum_num_ls])
    first_cal = True
    flow_time_interval = 60  # second
    for cum_num_dict, start_unix_time in all_cum_num_ls:
        for line_name, cum_num in cum_num_dict.items():
            start_x = (start_unix_time[0] - min_start_unix_time) / 1000  # 0
            x = np.linspace(start_x, start_x + (len(cum_num) - 1) * start_unix_time[1] / 1000,
                            len(cum_num))  # [0,0.1,0.2.....8998]
            res_cumsum = np.cumsum(cum_num)
            res = smoothWithsEMA(res_cumsum, 0.5)

            if slanted_cumulative_curve:
                capacity_flow = [capacity / 3600 * start_unix_time[1] / 1000 * lane_num] * len(cum_num)
                res_cap = np.cumsum(capacity_flow)
                res = res - res_cap
            plt.plot(x, res, color='blue', linewidth=0.5)
            plt.xlabel('Time (s)')
            plt.ylabel('N')
            plt.title('\n Slanted cumulative curves\n' + 'capacity%d' % capacity)
            file_path = os.path.join(save_folder, '_cumulative_curves_capacity%d' % capacity + '.jpg')
            plt.savefig('胆小%d.jpg' % capacity, dpi=1000)
            plt.show()


def get_vehicles_from_lane(target_lane_id):
    lines_ls = []
    speed_ls = []
    detaT = discrete_interval
    for key in total_info['position'].keys():  # 对每一辆车循环
        lane_id = total_info["lane_record"][key]
        begin_frame = total_info['vehicle_property'][key]["begin_frame_index"]
        for i in range(len(lane_id) - 1):  # 对车辆的车道进行循环
            if lane_id[i] == target_lane_id and lane_id[i + 1] == target_lane_id:
                x1 = (begin_frame + i) * detaT
                x2 = (begin_frame + i + 1) * detaT
                y1 = total_info['position'][key][i]
                y2 = total_info['position'][key][i + 1]
                speed = total_info['speed'][key][i]
                speed_ls.append(abs(speed))
                lines_ls.append([(x1, y1), (x2, y2)])
    return lines_ls, speed_ls


def plot_line(lines, color_speed, lane_id, figsize=(15, 5), start_time=None):
    '''
    绘制轨迹时空轨迹图
    '''
    fig = plt.figure(figsize=figsize)
    # fig, ax = plt.subplots(figsize=(15, 5))
    ax = fig.add_subplot()
    lines_sc = LineCollection(lines, array=np.array(color_speed), cmap="jet_r", linewidths=0.3)  # 每条线的数据
    ax.add_collection(lines_sc)
    lines_sc.set_clim(vmin=0, vmax=30)
    cb = fig.colorbar(lines_sc)
    ax.autoscale()
    cb.ax.set_title('speed [m/s]', fontsize=8)
    plt.xlabel("Time [s]")
    plt.ylabel("Location [m]")
    plt.title('Lane_id:%s' % lane_id)
    # plt.grid(None)
    # plt.savefig('./images/%d.jpg' %lane_id, dpi=750, bbox_inches = 'tight')
    # plt.savefig('lane_id%d.jpg'%lane_id, dpi=1000)
    plt.show()


if __name__ == "__main__":

    simulation_duration = 900
    mainstream_length = 1030
    merge_position = 400  # 汇入的位置
    exit_position = 800  # 分叉的位置
    weaving_area = 200  # 交织区长200
    mainstream_arrival_flow = 2000
    mainstream_lane_num = 3
    ramp_arrival_flow = 600

    average_parameters = {}
    average_parameters["jam_spacing"] = 7 #已知
    average_parameters["cri_spacing1"] = 23
    average_parameters["cri_spacing2"] = 36.15
    average_parameters["cri_speed"] = 14
    average_parameters["free_speed"] = 22 #已知
    average_parameters["maximum_acc"] = 1.0  #加速度已知
    average_parameters["dcc"] = -1  #减速度
    average_parameters["maximum_dcc"] = -3.0  #最大减速度
    average_parameters["tao_1"] = 1 / (average_parameters["cri_speed"] / (
                average_parameters["cri_spacing1"] - average_parameters["jam_spacing"]))
    average_parameters["tao_2"] = 1 / (average_parameters["cri_speed"] / (
                average_parameters["cri_spacing1"] - average_parameters["jam_spacing"]))
    # average_parameters["tao_2"] =1/( (average_parameters["free_speed"]-average_parameters["cri_speed"])/(average_parameters["cri_spacing2"]-average_parameters["cri_spacing1"]))

    with open("extension_list.json", "r") as json_file:
        reaction_time_extension_list = json.load(json_file)

    discrete_interval = 0.1
    LC_speed_threshold1 = 20
    LC_speed_threshold2 = 2
    merge_speed_reduction = 1
    OD_ratio = {}
    OD_ratio["o1_d1"] = 0.4
    OD_ratio["o1_d2"] = 0.6
    OD_ratio["o2_d1"] = 0.4
    OD_ratio["o2_d2"] = 0.6
    drive_style_ratio = {}
    drive_style_ratio["normal"] = 0.4
    drive_style_ratio["timid"] = 0.3
    drive_style_ratio["aggressive"] = 0.3
    position = {}
    speed = {}
    acceleration = {}
    vehicle_property = {}
    lane_record = {}
    spacing = {}
    total_info, accumulate_curve, all_time_pos, lc_vehicles_all = traffic_dynamics(discrete_interval, position, speed,
                                                                                   acceleration, lane_record, spacing,
                                                                                   vehicle_property,
                                                                                   reaction_time_extension_list)

    # 统计数据
    count_time_period = [300, 600]  # 统计的时间段300s-600s
    det_line_ls = 1000  # 检测位置
    lane_changing_count(lc_vehicles_all, count_time_period)  # 信息，统计的时间段
    flow_rate_count(all_time_pos, det_line_ls, count_time_period)  # 信息，统计的位置，统计的时间段

    # 绘制累计曲线图
    save_folder = '/accumulative save fold'
    lane_ls = [0, 1, 2]
    capacity_ls = [2000, 2100, 2200]
    section_region = list(range(0, 1050, 50))
    section_region_ls = [[x, y] for x, y in zip(section_region[:-1],
                                                section_region[1:])]  # [[0, 50], [50, 100], [100, 150],...[950, 1000]]
    color_dict = {600: (0.0, 1.0, 0.0)}
    new_det_line_ls = [det_line_ls]
    new_section_region_ls = section_region_ls
    all_xy = []
    all_speed = []
    cumulative_num_ls = {}
    for i in range(0, int(simulation_duration / 0.1)):
        if i in accumulate_curve['accumulate_time']:
            cumulative_num_ls[i] = accumulate_curve['accumulate_time'].count(i)
        else:
            continue
    cum_num_dict = {}
    cumulative_num_ls = [cumulative_num_ls]
    for cumulative_num, det_line in zip(cumulative_num_ls, new_det_line_ls):
        max_cum = max(cumulative_num.keys())
        cum_num = [0] * (max_cum + 1)
        for frame_index, veh_num in cumulative_num.items():
            cum_num[frame_index] = veh_num
        cum_num_dict[det_line] = cum_num
    all_xy.append([cum_num_dict, [0, 100]])
    for capacity in capacity_ls:
        plot_all_cumulative_curves(all_xy, save_folder, color_dict=color_dict, slanted_cumulative_curve=True,
                                   capacity=capacity)
    # 绘制轨迹图
    for lane_id in lane_ls:
        lines_ls, speed_ls = get_vehicles_from_lane(lane_id)
        plot_line(lines_ls, speed_ls, lane_id)
