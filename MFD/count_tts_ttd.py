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


def run_main(pkl_file):
    '''
    运行的入口函数
    :param multi_video_config:
    :return:
    '''

    with open(pkl_file, 'rb') as f:
        vehicle_data: object = pickle.load(f)

    #print(vehicle_data)
    #for key, value in vehicle_data.items():
        #print(key, value)
    lane_id_ls = [1,2,3,4,5,6,20,21,22,23,24,25,26]
    # lane_id_ls = [70,71,72,73,80,81,82,83]
    # lane_id_ls = [90,91,92,93,100,101,102,103]
    # lane_id_ls = [30,31,32,33,34,35,40,40,41,42,43,44,45]
    #lane_id_ls = [50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 65]
    driving_line_min = -100
    driving_line_max = 10000

    tts_ls = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ttd_ls = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    count_minute = list(range(20))
    tts = 0
    ttd = 0
    for lane_id_det in lane_id_ls:
        for veh_id, veh_data in vehicle_data.items():
            print(veh_data.keys())
            start_unix_time, ns_detaT = veh_data['start_unix_time']
            detaT = ns_detaT / 1000
            # every car in circulate
            frame_index = veh_data['frame_index']
            veh_frame_max = max(frame_index)
            # print(veh_frame_max)
            lane_id = veh_data['lane_id']
            start_unix_time, ns_detaT = veh_data['start_unix_time']
            driving_name = None
            print(veh_data.keys())
            #if driving_name:
                #drivingline = veh_data['drivingline'][driving_name]
                #drivingline_dist = smoothWithsEMA([x[0] for x in drivingline], 0.3, detaT)
            #else:
                #assert len(list(veh_data['drivingline'].keys())) == 1, 'give the drivingline name'
                #temp_driving_name = 'A2'#list(veh_data['drivingline'].keys())[0]
                #temp_driving_name = list(veh_data['drivingline'].keys())[0]
                #drivingline = veh_data['drivingline'][temp_driving_name]

                #drivingline_dist = smoothWithsEMA([x[0] for x in drivingline], 0.3, detaT)

            tt_frame = 0
            tts = 0

            # for i in range(len(lane_id)-1):
            # print(lane_id)
            # if (lane_id[i] == lane_id_det ) :
            #   if((drivingline_dist[i]>=driving_line_min) and (drivingline_dist[i]<=driving_line_max)):
            #      tt_frame = tt_frame + 1
            # print(tt_frame)
            # if frame_index[i]%600 == 0:
            #   minute = frame_index[i]//600

            #  minute=int(minute)
            #  tts = tt_frame/10
            # tts_ls[minute] += tts
            # tt_frame = 0
            #   if (frame_index[i] == veh_frame_max and frame_index[i]%600 != 0):
            # end = frame_index[i]//600 + 1
            #  end = int(end)
            # tts = tt_frame/10
            # tts_ls[end] += tts
            #  tt_frame = 0

            ttd = 0  # 不在每一帧统计位移，而是找出行驶的头和尾进行统计
            driving_frame_list = []
            driving_track_list = []

            for j in count_minute:
                driving_track_list = []
                driving_frame_list = []
                frame_start = j * 600
                frame_end = (j + 1) * 600
                for i in range(len(lane_id) - 1):
                    if (lane_id[i] == lane_id_det):
                        if ((drivingline_dist[i] >= driving_line_min) and (drivingline_dist[i] <= driving_line_max)):
                            if ((frame_index[i] >= frame_start) and (frame_index[i] < frame_end)):
                                driving_track_list.append(drivingline_dist[i])
                                driving_frame_list.append(frame_index[i])
                if (len(driving_track_list) >= 15):  # 除去在道路上时间较短的十字路口对向车辆
                    driving_track_list_len = len(driving_track_list) - 1
                    # print( driving_track_list[0])
                    # print(driving_track_list_len)
                    ttd = abs(driving_track_list[driving_track_list_len] - driving_track_list[0])
                    tts = (abs(driving_frame_list[driving_track_list_len] - driving_frame_list[0])) / 10

                    ttd_ls[j] += ttd
                    tts_ls[j] += tts

                    ttd = 0
                    tts = 0

                driving_track_list = []

    # tts_ls_df = pd.DataFrame(tts_ls, columns=['tts'])
    # ttd_ls_df = pd.DataFrame(ttd_ls, columns=['ttd'])
    # 保存到本地excel

    output_excel = {'tts': [], 'ttd': []}
    output_excel['tts'] = tts_ls
    output_excel['ttd'] = ttd_ls
    out_put = pd.DataFrame(output_excel)
    out_put.to_csv("/data3/liyitong/HuRong_process/MFD/E3/tts_ttd_E3_F7.csv")
    # tts_ls_df.to_excel("/data3/liyitong/HuRong_process/MFD/0512/tts_ls_E1_F5.xlsx")
    # ttd_ls_df.to_excel("/data3/liyitong/HuRong_process/MFD/0512/ttD_ls_E1_F5.xlsx")
    print(tts_ls)
    print(ttd_ls)
    print("end")

    # get k in every minute


if __name__ == '__main__':
    pkl_file = '/data3/liyitong/stitch_tppkl_multi_20220617_D1toA1_F1.tppkl'
    run_main(pkl_file)


