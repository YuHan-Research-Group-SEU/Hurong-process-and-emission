import pickle
from typing import Dict, Any

import numpy as np
import pandas as pd
from toolbox.trajectory_process_1 import TrajectoryProcess,unixtime2time
import openpyxl
import pickle
import numpy as np
import pandas as pd
from toolbox.trajectory_process_1 import TrajectoryProcess,unixtime2time
import numpy as np
import pandas as pd
import math
import time
from datetime import datetime
import pickle
import sklearn
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from toolbox.Vehicle import VehicleCollection
from scipy import interpolate

def smoothWithsEMA(lsdata, T, dt=0.1):
    """
    s�pn(��p��sG�
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

def get_vehicles_from_movements (vehicles_data,moevment_car_id_ls, traget_lane_id_ls,traget_lane_points,driving_name=None, x_is_unixtime=False, output_points=False):

    lines_ls = []
    speed_ls = []
    if output_points:
        start_points = [[], []]
        finish_points = [[], []]
        lc_points = [[], []]
    #print(moevment_car_id_ls)
    keys = vehicles_data.keys ()
    #print(keys)
    yes = 0
    for car_id_movement in moevment_car_id_ls:
        #print(car_id_movement)
        for veh_id, veh_data in vehicles_data.items ():
            #print (car_id_movement,veh_id)
            if car_id_movement == veh_id:
                yes += 1
                start_unix_time, ns_detaT = veh_data ['start_unix_time']
                detaT = ns_detaT / 1000
                lane_id = veh_data ['lane_id']
                frame_index = veh_data ['frame_index']
                # print(['drivingline'][0])
                if driving_name:
                    drivingline = veh_data ['drivingline']
                else:
                    assert len (list (veh_data ['drivingline'].keys ())) == 1, 'give the drivingline name'
                    temp_driving_name = list (veh_data ['drivingline'].keys ()) [0]
                    # print(temp_driving_name)
                    drivingline = veh_data ['drivingline'] [temp_driving_name]
                drivingline_dist = smoothWithsEMA ([x [0] for x in drivingline], 0.3, detaT)

                for i in range (len (lane_id) - 1):
                    print ("yes")
                    if x_is_unixtime:
                        x1 = (frame_index [i] - frame_index [0]) * ns_detaT + start_unix_time
                        x2 = (frame_index [i + 1] - frame_index [0]) * ns_detaT + start_unix_time
                    else:
                        x1 = frame_index [i] * detaT
                        x2 = frame_index [i + 1] * detaT
                    y1 = drivingline_dist [i]
                    y2 = drivingline_dist [i + 1]
                    speed = (y2 - y1) / detaT * 3.6
                    if speed > -5 and lane_id[i] in traget_lane_id_ls:
                        speed_ls.append (speed)
                        lines_ls.append ([(x1, y1), (x2, y2)])
                    if output_points and (lane_id[i] in traget_lane_points or lane_id[i + 1] in traget_lane_points):
                        if lane_id [i] != lane_id [i + 1]:
                            lc_points [0].append (x2)
                            lc_points [1].append (drivingline_dist [i + 1])
                    if output_points and (lane_id[i] in traget_lane_points or lane_id[i + 1] in traget_lane_points):
                        if x_is_unixtime:
                            x_s = start_unix_time
                            x_e = (frame_index [-1] - frame_index [0]) * ns_detaT + start_unix_time
                        else:
                            x_s = frame_index [0] * detaT
                            x_e = frame_index [-1] * detaT
                        start_points [0].append (x_s)
                        start_points [1].append (drivingline_dist [0])
                        finish_points [0].append (x_e)
                        finish_points [1].append (drivingline_dist [-1])
    if output_points:
        points = {'start': start_points, 'finish': finish_points, 'lc': lc_points}
        return lines_ls, speed_ls, points

    '''
          if output_points:
                        if lane_id[i] != lane_id[i + 1]:
                            lc_points[0].append (x2)
                            lc_points[1].append (drivingline_dist[i + 1])
                if output_points:
                    if x_is_unixtime:
                        x_s = start_unix_time
                        x_e = (frame_index[-1] - frame_index[0]) * ns_detaT + start_unix_time
                    else:
                        x_s = frame_index[0] * detaT
                        x_e = frame_index[-1] * detaT
                    start_points[0].append (x_s)
                    start_points[1].append (drivingline_dist[0])
                    finish_points[0].append (x_e)
                    finish_points[1].append (drivingline_dist[-1])
            if output_points:
                points = {'start': start_points, 'finish': finish_points, 'lc': lc_points}
                return lines_ls, speed_ls, points
            
                 '''

    return lines_ls, speed_ls



def plot_line(savepath, lines, color_speed, figsize=(15, 5), start_time=None, points=None):
    '''
        �6h��zh��
        :param savepath:
        :param lines:
        :param color_speed:
        :param figsize:
        :param start_time:
        :return:
    '''
    fig = plt.figure (figsize = figsize)
    # fig, ax = plt.subplots(figsize=(15, 5))
    ax = fig.add_subplot ()
    lines_sc = LineCollection (lines, array = np.array (color_speed), cmap = "jet_r", linewidths = 0.2)
    ax.add_collection (lines_sc)
    lines_sc.set_clim (vmin = 0, vmax = 120)
    cb = fig.colorbar (lines_sc)
    if not start_time is None:
        plt.title ('Start time:%s' % start_time, fontsize = 20)
    if not points is None:
        size = 1

        lc_points = points.get ('lc', None)
        if not lc_points is None:
            plt.scatter (lc_points [0], lc_points [1], c = 'k', marker = 'x', label = 'lane changing point', s = size)
        plt.legend (loc = 'upper right', fontsize = 16)
    ax.autoscale ()
    plt.xticks (fontsize = 18)
    plt.yticks (fontsize = 18)
    # 设置colorbar刻度的字体大小
    cb.ax.tick_params (labelsize = 18)
    cb.ax.set_title ('speed [km/h]', fontsize = 18)
    plt.xlabel ("Time [s]", fontsize = 18)
    plt.ylabel ("Location [m]", fontsize = 18)

    # plt.grid(None)
    # plt.show()
    plt.savefig (savepath, dpi = 1000, bbox_inches = 'tight')
    print ('save_img:%s' % savepath)


def run_main(pkl_file,movement_file):
    '''
    运行的入口函数
    :param multi_video_config:
    :return:
    '''
    with open(pkl_file,'rb') as f:
        vehicle_data = pickle.load(f)
    with open(movement_file,'rb') as f:
        movement_dict = pickle.load(f)
    #print(movement_dict)
    movement_car_id_ls = movement_dict['left_7_to_2']
    print(movement_car_id_ls)
    traget_lane_id_ls = [7,50,2]
    traget_lane_points = [7,2]
    lines_ls, speed_ls, points= get_vehicles_from_movements(vehicle_data,movement_car_id_ls,traget_lane_id_ls,traget_lane_points,output_points=True)
    fig_path = '/data3/liyitong/HuRong_process/intersection/D2/20220616_0700_D2_F1_370_1_Num_4/D2_left_7_to_2.jpg'
    plot_line (fig_path, lines_ls, speed_ls,points=points)



if __name__ == '__main__':
    pkl_file ='/data3/liyitong/HuRong_process/intersection/D2/20220616_0700_D2_F1_370_1_Num_4/tp_result_20220616_0700_D2_F1_left_7_to_2.tppkl'
    movement_file = '/data3/liyitong/HuRong_process/intersection/D2/20220616_0700_D2_F1_370_1_Num_4/movements_veh_id.pkl'
    run_main(pkl_file,movement_file)
