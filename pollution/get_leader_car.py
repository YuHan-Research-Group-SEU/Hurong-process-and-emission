#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time : 2023-04-27 21:04
@Author : Xinkai Ji
@contact: jhjxk@hotmail.com
@File : multi_video_trajectory_connection.py
@Software: PyCharm
@desc: k&q
'''
import os
import pickle
import json

import numpy as np
import pandas as pd
import copy

import math

df_veh = None
start_frame_index = None
end_frame_index = None
all_lane = None
save_folder = None
video_name = None
all_vehicle = None


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
            gt = pow(math.e, -abs(i - k) / deta)
            xe = lsdata[k] * gt
            lsgt.append(gt)
            lsxe.append(xe)
        outX = sum(lsxe) / sum(lsgt)
        outData.append(outX)
    return outData

def get_tppkl_data(tppkl_path):
    global start_frame_index
    global end_frame_index
    global all_lane
    global df_veh
    global save_folder
    global video_name  # 前面应该已经定义为全局变量了
    global all_vehicle

    with open(tppkl_path, 'rb') as f:  # 可自动关闭已经打开的文件
        vehicles_data = pickle.load(f)  # vehicles_data是dict数据类型
    print('load:%s' % tppkl_path)

    df_data = []

    for veh_id, veh_data in vehicles_data.items():
        temp_driving_name = list(veh_data['drivingline'].keys())[0]  # 选取一个drivingline名字
        drivingline = veh_data['drivingline'][temp_driving_name]
        for i in range(len(veh_data['frame_index'])):
            frame_index = int(veh_data['frame_index'][i])
            drivingline_dist_x = drivingline[i][0]
            #drivingline_dist_y = drivingline[i][1]
            lane_id = veh_data['lane_id'][i]
            temp_d = [frame_index, veh_id, lane_id, drivingline_dist_x]
            df_data.append(temp_d)
    #将所有车的信息存到dataframe中
    df_veh = pd.DataFrame(df_data,
                          columns=['frame_index', 'vehicle_id', 'lane_id', 'drivingline_dist_x'])
    start_frame_index = int(df_veh['frame_index'].min())
    end_frame_index = int(df_veh['frame_index'].max())
    all_lane = df_veh['lane_id'].unique()
    all_vehicle = df_veh['vehicle_id'].unique()

def get_CF (frame_index_c):
    new_df_ls = []
    for lane_id in all_lane:
        if lane_id == -1:
            continue
        if lane_id > 19:
            ascending = False
        else:
            ascending = True
        lane_veh = df_veh[(df_veh['frame_index'] == frame_index_c) & (df_veh['lane_id'] == lane_id)]. \
            sort_values (by='drivingline_dist_x', ascending=ascending)  # ascending=True升序
        for i in range (len (lane_veh)):
            current_veh_id = lane_veh.iloc[i]['vehicle_id']
            dist_x = lane_veh.iloc[i]['drivingline_dist_x']
            #dist_y = lane_veh.iloc[i]['drivingline_dist_y']
            #if i == 0:
                #following_veh_id = None
                #following_dist_x = None
                #following_dist_y = None
            if i == len (lane_veh) - 1:
                leader_veh_id = None
                leader_dist_x = None
                leader_dist_y = None
            #if i > 0 and i < len (lane_veh):
                #following_veh_id = lane_veh.iloc[i - 1]['vehicle_id']
                #following_dist_x = lane_veh.iloc[i - 1]['drivingline_dist_x']
                #following_dist_y = lane_veh.iloc[i - 1]['drivingline_dist_y']
            if (i < len (lane_veh) - 1):
                leader_veh_id = lane_veh.iloc[i + 1]['vehicle_id']
                leader_dist_x = lane_veh.iloc[i + 1]['drivingline_dist_x']
                print(leader_dist_x)
                #leader_dist_y = lane_veh.iloc[i + 1]['drivingline_dist_y']

            new_line = [frame_index_c, current_veh_id, lane_id, dist_x, leader_veh_id, leader_dist_x]
            new_df_ls.append (new_line)
    return new_df_ls

#会有空值
def get_CF_trajectory(couple_frame_df):
    dict_CF = {}
    for i in all_vehicle:
        veh_data_CF= {}
        lane_id = []
        frame_index=[]
        dist_x=[]
        leader_veh_id = []
        leader_dist_x = []
        this_veh = couple_frame_df[(couple_frame_df['current_veh_id'] == i)]
        this_veh = this_veh.sort_values (by=['frame_index'], ascending=[True])
        for j in range(len(this_veh)-1):
            frame_index.append(this_veh.iloc[j]['frame_index'])
            lane_id.append(this_veh.iloc[j]['lane_id'])
            dist_x.append(this_veh.iloc[j]['dist_x'])
            leader_veh_id.append(this_veh.iloc[j]['leader_veh_id'])
            leader_dist_x.append(this_veh.iloc[j]['leader_dist_x'])
        veh_data_CF['frame_index'] = frame_index
        veh_data_CF['lane_id'] = lane_id
        veh_data_CF['dist_x'] = dist_x
        veh_data_CF['leader_veh_id'] = leader_veh_id
        veh_data_CF['leader_dist_x'] = leader_dist_x
        dict_CF[i] = veh_data_CF
    return dict_CF




def run_main(pkl_file):
    all_lane = [1,2,3,4,5,20,21,22,23,24]
    get_tppkl_data(pkl_file)
    frame_ls = list (range (start_frame_index, end_frame_index + 1))
    couple_frame_ls = []
    for frame_index_c in frame_ls:
        couple_frame = get_CF (frame_index_c)
        couple_frame_ls += couple_frame
    couple_frame_df = pd.DataFrame(couple_frame_ls,
                           columns=['frame_index', 'current_veh_id', 'lane_id', 'dist_x', 'leader_veh_id', 'leader_dist_x'])
    #print(couple_frame_df)

    dict_CF = get_CF_trajectory(couple_frame_df)
    #print(dict_CF)
    file_name = "/data3/liyitong/YingTian_process/F1/my_dict.pkl"

    # 使用 pickle.dump() 将字典保存到文件
    with open (file_name, "wb") as file:
        pickle.dump (dict_CF, file)

    print (f"字典已保存到 {file_name}")






if __name__ == '__main__':
    tppkl_ls = ['/data3/DJIData/YingtianStreet/20220708/Y1-Y2/M-20220708_Y1_A_F1_1-S-20220708_Y2_A_F1_1/new_tp_result_first_frame_M-20220708_Y1_A_F1_1-S-20220708_Y2_A_F1_1.tppkl']
    for tppkl_path in tppkl_ls:
        run_main(tppkl_path)
