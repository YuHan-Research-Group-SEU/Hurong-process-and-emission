import os
import pickle
import json

import numpy as np
import pandas as pd
import copy

import math
import pickle
import numpy as np
import pandas as pd
#from toolbox.trajectory_process_CF import TrajectoryProcess,unixtime2time
import numpy as np
import pandas as pd
import math
import time
from datetime import datetime
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from toolbox.Vehicle import VehicleCollection
from scipy import interpolate
import numpy as np
#from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pywt

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


def plot_line (savepath, lines, color_speed, figsize=(15, 5), start_time=None, points=None):
    fig = plt.figure (figsize=figsize)
    # fig, ax = plt.subplots(figsize=(15, 5))
    ax = fig.add_subplot ()
    lines_sc = LineCollection (lines, array=np.array (color_speed), cmap="jet_r", linewidths=0.2)
    ax.add_collection (lines_sc)
    lines_sc.set_clim (vmin=0, vmax=120)
    cb = fig.colorbar (lines_sc)
    if not start_time is None:
        plt.title ('Start time:%s' % start_time, fontsize=20)
    if not points is None:
        size = 1

        lc_points = points.get ('lc', None)
        if not lc_points is None:
            plt.scatter (lc_points[0], lc_points[1], c='k', marker='x', label='lane changing point', s=size)
        plt.legend (loc='upper right', fontsize=16)
    ax.autoscale ()
    plt.xticks (fontsize=18)
    plt.yticks (fontsize=18)
    # 设置colorbar刻度的字体大小
    cb.ax.tick_params (labelsize=18)
    cb.ax.set_title ('speed [km/h]', fontsize=18)
    plt.xlabel ("Time [s]", fontsize=18)
    plt.ylabel ("Location [m]", fontsize=18)

    # plt.grid(None)
    # plt.show()
    plt.savefig (savepath, dpi=1000, bbox_inches='tight')
    print ('save_img:%s' % savepath)

def get_vehicles_from_lane (vehicles_data,target_lane_id,output_points=False):
    lines_ls = []
    speed_ls = []
    if output_points:
        start_points = [[], []]
        finish_points = [[], []]
        lc_points = [[], []]
    for veh_id, veh_data in vehicles_data.items ():
        detaT = 0.1
        lane_id = veh_data['lane_id']
        frame_index = veh_data['frame_index']

        drivingline = veh_data['dist_x']
        drivingline_dist = smoothWithsEMA ([x for x in drivingline], 0.3, detaT)
        for i in range (len (lane_id) - 1):
            x1 = frame_index [i] * detaT
            x2 = frame_index [i + 1] * detaT
            if lane_id [i] == target_lane_id and lane_id [i + 1] == target_lane_id:
                y1 = drivingline_dist [i]
                y2 = drivingline_dist [i + 1]
                speed = abs((y2 - y1) / detaT * 3.6)
                speed_ls.append (speed)
                lines_ls.append ([(x1, y1), (x2, y2)])
            if output_points:
                if lane_id [i] != lane_id [i + 1] and (
                        lane_id [i] == target_lane_id or lane_id [i + 1] == target_lane_id):
                    if lane_id [i] == target_lane_id:
                        lc_points[0].append (x1)
                        lc_points[1].append (drivingline_dist [i])
                    else:
                        lc_points[0].append (x2)
                        lc_points[1].append (drivingline_dist [i + 1])

        if (output_points &len(frame_index)>0):
            x_s = frame_index [0] * detaT
            x_e = frame_index [-1] * detaT
            start_points [0].append (x_s)
            start_points [1].append (drivingline_dist [0])
            finish_points [0].append (x_e)
            finish_points [1].append (drivingline_dist [-1])
    if output_points:
        points = {'start': start_points, 'finish': finish_points, 'lc': lc_points}
        return lines_ls, speed_ls, points
    return lines_ls, speed_ls


#def count_SDAI(loaded_dict):
    #for for veh_id, veh_data in loaded_dict ():

def plot_CF(vehicles_data):
    length_ls = []
    speed_ls = []
    lane_id_ls = []
    frame_index_ls = []
    for veh_id, veh_data in vehicles_data.items ():
        length_ls_car = []
        speed_ls_car = []
        detaT = 0.1
        lane_id = veh_data['lane_id']
        frame_index = veh_data['frame_index']

        drivingline = veh_data['dist_x']
        drivingline_leader = veh_data['leader_dist_x']
        for i in range (len (lane_id) - 1):

            speed = abs ((drivingline[i + 1] - drivingline[i]) / detaT * 3.6)
            if (speed <120):
                length = abs(drivingline[i]-drivingline_leader[i])
                #print(drivingline[i])
                #print(drivingline_leader[i])
                #print(length)
                speed = abs((drivingline[i+1] - drivingline[i]) / detaT * 3.6)
                length_ls_car.append(length)
                speed_ls_car.append(speed)
        if len (length_ls_car)!=0:

            average = sum (length_ls_car) / len (length_ls_car)
            length_ls.append(average)
            average = sum (speed_ls_car) / len (speed_ls_car)
            speed_ls.append (average)
    plt.scatter (speed_ls, length_ls, label='Data Points', color='blue', marker='o',alpha=0.5)

    # 添加标签和标题
    plt.xlabel ('X-axis')
    plt.ylabel ('Y-axis')
    plt.title ('Scatter Plot Example')
    plt.show()

def get_middle_trajectory(start,end,veh_id,veh_data):
    new_dict = {}
    lane_id_new = []
    frame_index_new = []
    dist_x_new = []
    leader_x_new = []
    leader_veh_id_new = []
    veh_id_new = []

    lane_id = veh_data['lane_id']
    frame_index = veh_data['frame_index']
    drivingline = veh_data['dist_x']
    drivingline_leader = veh_data['leader_dist_x']
    leader_veh_id = veh_data['leader_veh_id']

    for i in range(start, end + 1):
        lane_id_new.append(lane_id[i])
        frame_index_new.append(frame_index[i])
        dist_x_new.append(drivingline[i])
        leader_veh_id_new.append(leader_veh_id[i])
        leader_x_new.append(drivingline_leader[i])
        veh_id_new.append(veh_id)
    new_dict['frame_index'] = frame_index_new
    new_dict['lane_id'] = lane_id_new
    new_dict['dist_x'] = dist_x_new
    new_dict['leader_veh_id'] = leader_veh_id_new
    new_dict['leader_dist_x'] = leader_x_new
    new_dict['veh_id'] = veh_id_new
    return new_dict

def find_deceleration_acc (vehicles_data):
    #speed_ls = []
    veh_data_new = {}
    deceleration_num = 0
    for veh_id, veh_data in vehicles_data.items ():
        detaT = 0.1

        lane_id = veh_data['lane_id']
        drivingline = veh_data['dist_x']
        drivingline_dist = smoothWithsEMA ([x for x in drivingline], 0.3, detaT)

        start = -1
        end = -1
        for i in range (len (lane_id) - 2):
            if lane_id [i] == lane_id [i+1] and lane_id [i] ==lane_id [i+2]:
                y1 = drivingline_dist [i]
                y2 = drivingline_dist [i + 1]
                y3 = drivingline_dist [i + 2]
                speed_1 = abs((y2 - y1) / detaT * 3.6)
                speed_2 = abs((y3 - y2) / detaT * 3.6)
                acc = (speed_2-speed_1) / detaT
                #print(speed)
                #print(acc)
                if start == -1 and acc <= -1000:#找出减速并加速的起始点
                    start = i
                if end ==-1 and acc >= 1000:
                    end = i
                if (start != -1 & end !=-1 & start < end):
                    #print("yes")
                    start = -1
                    end = -1
                    new_dict = get_middle_trajectory(start,end,veh_id,veh_data)
                    deceleration_num += 1
                    veh_data_new[deceleration_num] = new_dict

    return veh_data_new

def find_deceleration (vehicles_data):
    new_veh_data = {}

    for veh_id, veh_data in vehicles_data.items ():
        new_dict = {}
        detaT = 0.1
        drivingline_dist = veh_data['dist_x']
        #drivingline_dist = smoothWithsEMA([x for x in drivingline], 0.3, detaT)
        lane_id = veh_data['lane_id']
        frame_index = veh_data['frame_index']
        drivingline_leader = veh_data['leader_dist_x']
        leader_veh_id = veh_data['leader_veh_id']

        lane_id_new = []
        frame_index_new = []
        dist_x_new = []
        leader_x_new = []
        leader_veh_id_new = []
        veh_id_new = []
        for i in range(len(lane_id)-1):
            if lane_id[i] == lane_id[i+1]:
                y1 = drivingline_dist[i]
                y2 = drivingline_dist[i + 1]
                speed = abs((y2 - y1) / detaT * 3.6)
                #acc = (speed_2-speed_1) / detaT
                if speed <= 12:
                    lane_id_new.append (lane_id[i])
                    frame_index_new.append (frame_index[i])
                    dist_x_new.append (drivingline_dist[i])
                    leader_veh_id_new.append (leader_veh_id[i])
                    leader_x_new.append (drivingline_leader[i])
                    veh_id_new.append (veh_id)
        new_dict['frame_index'] = frame_index_new
        new_dict['lane_id'] = lane_id_new
        new_dict['dist_x'] = dist_x_new
        new_dict['leader_veh_id'] = leader_veh_id_new
        new_dict['leader_dist_x'] = leader_x_new
        new_dict['veh_id'] = veh_id_new
        new_veh_data[veh_id] = new_dict
    return new_veh_data

def get_every_deceleration(vehicles_data):
    new_veh_dict = {}
    deceleration_num = 0
    for veh_id, veh_data in vehicles_data.items ():
        new_dict = {}
        detaT = 0.1
        drivingline_dist = veh_data['dist_x']
        #drivingline_dist = smoothWithsEMA([x for x in drivingline], 0.3, detaT)
        lane_id = veh_data['lane_id']
        frame_index = veh_data['frame_index']
        drivingline_leader = veh_data['leader_dist_x']
        #drivingline_leader = smoothWithsEMA([x for x in leader_x], 0.3, detaT)
        leader_veh_id = veh_data['leader_veh_id']

        lane_id_new_1 = []
        frame_index_new_1 = []
        dist_x_new_1 = []
        leader_x_new_1 = []
        leader_veh_id_new_1 = []
        veh_id_new_1 = []

        dece_time = 0
        for i in range (len (lane_id) - 1):
            if frame_index[i + 1] - frame_index[i] <= 10:
                dece_time += 1
                #print(dece_time)
            if frame_index[i + 1] - frame_index[i] >= 30:
                if dece_time >= 50 and (abs(drivingline_dist[i-dece_time+1]-drivingline_dist[i]) > 5):
                    deceleration_num += 1
                    #print("save")
                    for k in range(dece_time):
                        lane_id_new_1.insert (0,lane_id[i-k])
                        frame_index_new_1.insert (0,frame_index[i-k])
                        dist_x_new_1.insert (0,drivingline_dist[i-k])
                        leader_veh_id_new_1.insert (0,leader_veh_id[i-k])
                        leader_x_new_1.insert (0,drivingline_leader[i-k])
                        veh_id_new_1.append (veh_id)
                    dece_time = 0
                    new_dict['frame_index'] = frame_index_new_1
                    new_dict['lane_id'] = lane_id_new_1
                    new_dict['dist_x'] = dist_x_new_1
                    new_dict['leader_veh_id'] = leader_veh_id_new_1
                    new_dict['leader_dist_x'] = leader_x_new_1
                    new_dict['veh_id'] = veh_id_new_1
                    new_veh_dict[deceleration_num] = new_dict
                else:
                    dece_time = 0
    return new_veh_dict
def find_deceleration_walevet (vehicles_data):
    new_veh_dict = { }
    deceleration_num = 0
    for veh_id , veh_data in vehicles_data.items ( ) :
        new_dict = { }
        detaT = 0.1
        drivingline_dist = veh_data ['dist_x']
        # drivingline_dist = smoothWithsEMA([x for x in drivingline], 0.3, detaT)
        lane_id = veh_data ['lane_id']
        frame_index = veh_data ['frame_index']
        drivingline_leader = veh_data ['leader_dist_x']
        # drivingline_leader = smoothWithsEMA([x for x in leader_x], 0.3, detaT)
        leader_veh_id = veh_data ['leader_veh_id']

        lane_id_new_1 = []
        frame_index_new_1 = []
        dist_x_new_1 = []
        leader_x_new_1 = []
        leader_veh_id_new_1 = []
        veh_id_new_1 = []

        speed_ls = []

        dece_veh = 0
        for i in range (len (lane_id) - 10) :
            y1 = drivingline_dist [i]
            y2 = drivingline_dist [i + 1]
            speed = abs ((y2 - y1) / detaT * 3.6)
            speed_ls.append (speed)
        time = np.arange(0, len(speed_ls))
        time_x = time/10
        # for i in range (len (time)) :
        #     time[i] = time[i]/10

        speed_array = np.array(speed_ls)
        wavelet = 'mexh'  # 墨西哥帽小波
        widths = np.arange(10, 25)  # 尺度范围，可以根据需要调整
        if len(speed_ls)>=20:
            coefficients , frequencies = pywt.cwt (speed_array , widths , wavelet)

            plt.figure (figsize = (12 , 6))

            # 原始速度时间序列
            plt.subplot (3 , 1 , 1)
            plt.plot (time_x , speed_array , label = 'original speed')
            # plt.xticks (time_x,time)
            plt.xticks (np.arange (min (time_x) , max (time_x)
                                   + 10 , 10) , rotation = 45)

            plt.title ('raw velocity time series')
            plt.xlabel ('time(s)')
            plt.ylabel ('speed(km/s)')
            plt.legend ( )

            # 墨西哥帽小波拟合图
            plt.subplot (3 , 1 , 2)
            plt.imshow (np.abs (coefficients) , aspect = 'auto' , extent = [0 , 10 , 1 , 31] , cmap = 'jet' ,
                        interpolation = 'bilinear')
            # plt.colorbar (label = 'amplitude')
            plt.title ('Continuous Wavelet Transform(Mexican hat wavelet)')
            plt.xlabel ('time(s)')
            plt.ylabel ('arrange')

            energy = np.abs (coefficients) ** 2
            energy_sum = np.sum (energy , axis = 0)

            # 可视化能量分布
            plt.subplot (3 , 1 , 3)
            plt.plot (time_x , energy_sum)
            plt.xticks (np.arange (min (time_x) , max (time_x)
                                   + 10 , 10) , rotation = 45)
            plt.ylim (0,20000)
            # plt.xticks (time_x,time)
            plt.title ('energy')
            plt.xlabel ('time(s)')
            plt.ylabel ('energy')

            plt.tight_layout ( )
            plt.show ( )


def get_every_deceleration_walevet(vehicles_data):
    new_veh_dict = { }
    deceleration_num = 0
    for veh_id , veh_data in vehicles_data.items ( ) :
        new_dict = { }
        detaT = 0.1
        drivingline_dist = veh_data ['dist_x']
        # drivingline_dist = smoothWithsEMA([x for x in drivingline], 0.3, detaT)
        lane_id = veh_data ['lane_id']
        frame_index = veh_data ['frame_index']
        drivingline_leader = veh_data ['leader_dist_x']
        # drivingline_leader = smoothWithsEMA([x for x in leader_x], 0.3, detaT)
        leader_veh_id = veh_data ['leader_veh_id']

        lane_id_new_1 = []
        frame_index_new_1 = []
        dist_x_new_1 = []
        leader_x_new_1 = []
        leader_veh_id_new_1 = []
        veh_id_new_1 = []

        speed_ls = []

        dece_veh = 0
        for i in range(len(lane_id) -1 ):
            y1 = drivingline_dist [i]
            y2 = drivingline_dist [i + 1]
            speed = abs((y2 - y1) / detaT * 3.6)
            speed_ls.append(speed)

        wavelet = 'mexh'  # 墨西哥帽小波
        widths = 20  # 尺度范围，可以根据需要调整
        coefficients , frequencies = pywt.cwt (speed_ls , widths , wavelet)

        plt.figure (figsize = (12 , 6))

        # 原始速度时间序列
        plt.subplot (2 , 1 , 1)
        plt.plot (time , speed , label = 'original speed')
        plt.title ('raw velocity time series')
        plt.xlabel ('time')
        plt.ylabel ('speed')
        plt.legend ( )

        # 墨西哥帽小波拟合图
        plt.subplot (2 , 1 , 2)
        plt.imshow (np.abs (coefficients) , aspect = 'auto' , extent = [0 , 10 , 1 , 31] , cmap = 'jet' ,
                    interpolation = 'bilinear')
        plt.colorbar (label = 'amplitude')
        plt.title ('Continuous Wavelet Transform(Mexican hat wavelet)')
        plt.xlabel ('time')
        plt.ylabel ('arrange')

        plt.tight_layout ( )
        plt.show ( )
                #     new_dict ['frame_index'] = frame_index_new_1
                #     new_dict ['lane_id'] = lane_id_new_1
                #     new_dict ['dist_x'] = dist_x_new_1
                #     new_dict ['leader_veh_id'] = leader_veh_id_new_1
                #     new_dict ['leader_dist_x'] = leader_x_new_1
                #     new_dict ['veh_id'] = veh_id_new_1
                #     new_veh_dict [deceleration_num] = new_dict
                # else :
                #     dece_time = 0
    return new_veh_dict





def count_SDAI(deceleration):
    jam_spacing= 7  # 已知
    #jam_spacing = 6
    cri_speed = 14
    cri_spacing = 23
    SDAI_dict = {}
    tao_stan = (cri_spacing - jam_spacing)/cri_speed
    SDAI_ls = []
    #veh_ls = []
    deceleration_num_ls = []
    for deceleration_num, veh_data in deceleration.items ():
        detaT = 0.1
        SDAI_i =0
        SDAI_veh = 0
        lane_id = veh_data['lane_id']
        frame_index = veh_data['frame_index']
        drivingline_dist = veh_data['dist_x']
        drivingline_leader_dist = veh_data['leader_dist_x']
        for i in range(len (lane_id) - 1):
            length = abs(drivingline_leader_dist[i]-drivingline_dist[i])
            speed = abs((drivingline_dist[i + 1] - drivingline_dist[i]) / detaT * 3.6)
            #if length <= 3.5:
                #length = 5
            tao_i = abs(length - jam_spacing)/speed
            yita = tao_i / tao_stan

            SDAI_sec = abs(yita-1) * detaT
            SDAI_i += SDAI_sec
        #print(SDAI_i,len((lane_id)))
        SDAI_veh = SDAI_i/(len (lane_id) - 1) * 10
        #print(SDAI_veh)
        if not math.isnan(SDAI_veh):
            if SDAI_veh <= 1 and SDAI_veh >=0 :
                SDAI_ls.append(SDAI_veh)
                deceleration_num_ls.append(deceleration_num)
    return SDAI_ls,deceleration_num_ls

#注意单位需要换算成英里
def count_pollution(deceleration):

    NO_VSP = [0.00029,0.000223,0.000174,0.000719,0.001136,0.001587,0.00237,
              0.004098,0.006124,0.007313,0.013178,0.012663,0.015387,0.020308]
    HC_VSP = [0.000548,0.000222,0.000272,0.000472,0.000754,0.000702,0.000944,
              0.001443,0.001708,0.002605,0.003523,0.007653,0.006667,0.006574]
    CO2_VSP = [1.566819,1.443564,1.470553,2.611318,3.523681,4.650741,5.635386,
               6.599677,7.647334,8.808448,11.67061,14.52036, 15.65327,17.36653]
    CO_VSP = [0.017699,0.008608,0.008479,0.014548,0.025709,0.025212,0.04113,
              0.076601,0.129248,0.150578,0.355223,0.881642, 0.755155,0.904851]
    NO_ls = []
    HC_ls = []
    CO2_ls = []
    CO_ls = []

    deceleration_num_ls = []

    for deceleration_num, veh_data in deceleration.items ():
        detaT = 0.1
        frame_index = veh_data['frame_index']
        drivingline_dist = veh_data['dist_x']
        drivingline_leader_dist = veh_data['leader_dist_x']
        lane_id = veh_data['lane_id']

        NO_veh = 0
        HC_veh = 0
        CO2_veh = 0
        CO_veh = 0

        for i in range (len (lane_id) - 2):
            y1 = drivingline_dist [i]
            y2 = drivingline_dist [i + 1]
            y3 = drivingline_dist [i + 2]
            speed_1 = abs ((y2 - y1) / detaT * 2.236936)
            speed_2 = abs ((y3 - y2) / detaT * 2.236936)
            acc = (speed_2 - speed_1) / detaT
            VSP = speed_1 * (1.1 * acc + 0.132) + 0.000302 * speed_1 * speed_1 * speed_1
            if VSP < -2:
                NO = NO_VSP[0] * detaT
                HC = HC_VSP [0] * detaT
                CO2 = CO2_VSP [0] * detaT
                CO = CO_VSP [0] * detaT
            elif VSP >= -2 and VSP < 0:
                NO = NO_VSP[1] * detaT
                HC = HC_VSP [1] * detaT
                CO2 = CO2_VSP [1] * detaT
                CO = CO_VSP [1] * detaT
            elif VSP >= 0 and VSP < 1:
                NO = NO_VSP[2] * detaT
                HC = HC_VSP [2] * detaT
                CO2 = CO2_VSP [2] * detaT
                CO = CO_VSP [2] * detaT
            elif VSP >= 1 and VSP < 4:
                NO = NO_VSP[3] * detaT
                HC = HC_VSP [3] * detaT
                CO2 = CO2_VSP [3] * detaT
                CO = CO_VSP [3] * detaT
            elif VSP >= 4 and VSP < 7:
                NO = NO_VSP[4] * detaT
                HC = HC_VSP [4] * detaT
                CO2 = CO2_VSP [4] * detaT
                CO = CO_VSP [4] * detaT
            elif VSP >= 7 and VSP < 10:
                NO = NO_VSP[5] * detaT
                HC = HC_VSP [5] * detaT
                CO2 = CO2_VSP [5] * detaT
                CO = CO_VSP [5] * detaT
            elif VSP >= 10 and VSP < 13:
                NO = NO_VSP[6] * detaT
                HC = HC_VSP [6] * detaT
                CO2 = CO2_VSP [6] * detaT
                CO = CO_VSP [6] * detaT
            elif VSP >= 13 and VSP < 16:
                NO = NO_VSP[7] * detaT
                HC = HC_VSP [7] * detaT
                CO2 = CO2_VSP [7] * detaT
                CO = CO_VSP [7] * detaT
            elif VSP >= 16 and VSP < 19:
                NO = NO_VSP[8] * detaT
                HC = HC_VSP [8] * detaT
                CO2 = CO2_VSP [8] * detaT
                CO = CO_VSP [8] * detaT
            elif VSP >= 19 and VSP < 23:
                NO = NO_VSP[9] * detaT
                HC = HC_VSP [9] * detaT
                CO2 = CO2_VSP [9] * detaT
                CO = CO_VSP [9] * detaT
            elif VSP >= 23 and VSP < 28:
                NO = NO_VSP[10] * detaT
                HC = HC_VSP [10] * detaT
                CO2 = CO2_VSP [10] * detaT
                CO = CO_VSP [10] * detaT
            elif VSP >= 28 and VSP <33:
                NO = NO_VSP[11] * detaT
                HC = HC_VSP [11] * detaT
                CO2 = CO2_VSP [11] * detaT
                CO = CO_VSP [11] * detaT
            elif VSP >= 33 and VSP < 39:
                NO = NO_VSP[12] * detaT
                HC = HC_VSP [12] * detaT
                CO2 = CO2_VSP [12] * detaT
                CO = CO_VSP [12] * detaT
            elif VSP >= 39:
                #print("yes")
                NO = NO_VSP[13] * detaT
                HC = HC_VSP [13] * detaT
                CO2 = CO2_VSP [13] * detaT
                CO = CO_VSP [13] * detaT
            NO_veh += NO
            HC_veh += HC
            CO2_veh += CO2
            CO_veh += CO
        NO_veh = NO_veh/(len(lane_id) - 2) * 10
        HC_veh = HC_veh / (len (lane_id) - 2) * 10
        CO2_veh = CO2_veh / (len (lane_id) - 2) * 10
        CO_veh = CO_veh / (len (lane_id) - 2) * 10

        NO_ls.append(NO_veh)
        HC_ls.append(HC_veh)
        CO2_ls.append(CO2_veh)
        CO_ls.append(CO_veh)
        deceleration_num_ls.append(deceleration_num)
    return deceleration_num_ls,NO_ls,HC_ls,CO2_ls,CO_ls


def run_main(pkl_file):
    with open (pkl_file, "rb") as file:
        loaded_dict = pickle.load(file)
    deceleration = find_deceleration (loaded_dict)
    a1 = find_deceleration_walevet(loaded_dict)
    print(len(loaded_dict))
    lane_id_ls = [1,2,3,4,5]
    #for target_lane_id in lane_id_ls:
        #print(deceleration)
        #lines_ls, speed_ls = get_vehicles_from_lane(deceleration,target_lane_id,output_points=False)
        #fig_path = '/data3/liyitong/HuRong_process/laneold%d.jpg' % target_lane_id
        #plot_line (fig_path, lines_ls, speed_ls)

    deceleration_2 = get_every_deceleration(deceleration)
    #deceleration_3 = get_every_deceleration_walevet()
    #for target_lane_id in lane_id_ls:
        #print(deceleration)
        #lines_ls, speed_ls = get_vehicles_from_lane(deceleration_2,target_lane_id,output_points=False)
        #fig_path = '/data3/liyitong/HuRong_process/lane_dece%d.jpg' % target_lane_id
        #plot_line (fig_path, lines_ls, speed_ls)

    SDAI_ls, dece_SDAI_num_ls = count_SDAI(deceleration_2)
    print(SDAI_ls)
    print (len(dece_SDAI_num_ls))

    dece_polu_num_ls, NO_ls, HC_ls, CO2_ls, CO_ls = count_pollution(deceleration_2)
    print(len(dece_polu_num_ls))
    print(NO_ls)
    #count_SDAI(pkl_file)
    #plot_CF(loaded_dict)
    NO_ls_new = []
    HC_ls_new = []
    CO2_ls_new = []
    CO_ls_new = []
    SDAI_new = []
    for i in range(len(dece_polu_num_ls)):
        for j in range(len(dece_SDAI_num_ls)):
            if dece_polu_num_ls[i] ==dece_SDAI_num_ls[j]:
                NO_ls_new.append(NO_ls[i])
                HC_ls_new.append(HC_ls[i])
                CO2_ls_new.append(CO2_ls[i])
                CO_ls_new.append(CO_ls[i])
                SDAI_new.append(SDAI_ls[j])
    plt.figure (figsize=(9, 7))
    plt.scatter (SDAI_new, CO_ls_new, label='Data Points', color='blue', marker='o',alpha = 0.5)

    # 添加标题和标签
    plt.xticks (fontsize=15)
    plt.yticks (fontsize=15)
    plt.title (' SDAI and major pollutant emissions ',fontsize=20)
    plt.xlabel ('SDAI(s)',fontsize=20)
    plt.ylabel ('CO(g)',fontsize=20)


    plt.show ()

if __name__ == '__main__':
    tppkl_ls = ['/data3/liyitong/HuRong_process/B1/20220616_0815_B1_F4_371_1_Num_5/my_dict.pkl']
    for tppkl_path in tppkl_ls:
        run_main(tppkl_path)