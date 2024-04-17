import pickle
import numpy as np
import pandas as pd
import sklearn as skl
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

from toolbox.trajectory_process_1 import TrajectoryProcess,unixtime2time



# 画出雷达数据
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

def get_vehicles_choose ( vehicles_data,lane_id_ls,driving_name=None ,x_is_unixtime=False ,output_points=False) :
    lines_ls = []
    speed_ls = []
    if output_points :
        start_points = [[] , []]
        finish_points = [[] , []]
        lc_points = [[] , []]
    for veh_id , veh_data in vehicles_data.items () :
        #if veh_id in veh_choose :
        start_unix_time , ns_detaT = veh_data ['start_unix_time']
        detaT = ns_detaT / 1000
        lane_id = veh_data ['lane_id']
        frame_index = veh_data ['frame_index']
        speed_y = veh_data['speed_y']
        if driving_name :
            drivingline = veh_data ['drivingline']
        else :
            assert len (list (veh_data ['drivingline'].keys ( ))) == 1 , 'give the drivingline name'
            temp_driving_name = list (veh_data ['drivingline'].keys ( )) [0]
            # print(temp_driving_name)
            drivingline = veh_data ['drivingline'] [temp_driving_name]
        # print(type(drivingline[0]))
        #drivingline_dist = smoothWithsEMA ([x [0] for x in drivingline] , 0.3 , detaT)
        drivingline_dist = []
        for i in drivingline:
            drivingline_dist.append(i[0])
        for i in range (len (lane_id) - 1) :
            if x_is_unixtime :
                x1 = (frame_index [i] - frame_index [0]) * ns_detaT + start_unix_time
                x2 = (frame_index [i + 1] - frame_index [0]) * ns_detaT + start_unix_time

            else :
                x1 = frame_index [i] * detaT
                x2 = frame_index [i + 1] * detaT
            if lane_id [i] in lane_id_ls :
                y1 = drivingline_dist [i]
                y2 = drivingline_dist [i + 1]
                #speed = abs ((y2 - y1) / detaT * 3.6)
                speed_ls.append (abs(speed_y[i])*3.6)
                lines_ls.append ([(x1 , y1) , (x2 , y2)])
            if output_points :
                if lane_id [i] != lane_id [i + 1] :
                    lc_points [0].append (x2)
                    lc_points [1].append (drivingline_dist [i + 1])
        if output_points :
            if x_is_unixtime :
                x_s = start_unix_time
                x_e = (frame_index [-1] - frame_index [0]) * ns_detaT + start_unix_time
            else :
                x_s = frame_index [0] * detaT
                x_e = frame_index [-1] * detaT
            start_points [0].append (x_s)
            start_points [1].append (drivingline_dist [0])
            finish_points [0].append (x_e)
            finish_points [1].append (drivingline_dist [-1])
    if output_points :
        points = { 'start' :start_points , 'finish' :finish_points , 'lc' :lc_points }
        return lines_ls , speed_ls , points
    return lines_ls , speed_ls


def plot_line (savepath , lines , color_speed , figsize=(20, 5) , start_time=None , points=None) :

    fig = plt.figure (figsize = figsize)
    # fig, ax = plt.subplots(figsize=(15, 5))
    ax = fig.add_subplot ()
    lines_sc = LineCollection (lines , array = np.array (color_speed) , cmap = "jet_r" , linewidths = 0.2)
    ax.add_collection (lines_sc)
    lines_sc.set_clim (vmin = 0 , vmax = 120)
    cb = fig.colorbar (lines_sc)
    if not start_time is None :
        plt.title ('Start time:%s' % start_time , fontsize = 20)
    if not points is None :
        size = 1
        lc_points = points.get ('lc' , None)
        if not lc_points is None :
            plt.scatter (lc_points [0] , lc_points [1] , c = 'k' , marker = 'x' , label = 'lane changing point' ,
                         s = size)
        plt.legend (loc = 'upper right' , fontsize = 16)
    ax.autoscale ()
    plt.xticks (fontsize = 18)
    plt.yticks (fontsize = 18)
    # 设置colorbar刻度的字体大小
    cb.ax.tick_params (labelsize = 18)
    cb.ax.set_title ('speed [km/h]' , fontsize = 18)
    plt.xlabel ("Time [s]" , fontsize = 18)
    plt.ylabel ("Location [m]" , fontsize = 18)

    # plt.grid(None)
    # plt.show()
    plt.savefig (savepath , dpi = 1000 , bbox_inches = 'tight')
    print ('save_img:%s' % savepath)

def get_vehicles_from_lane(vehicles_data,target_lane_id, driving_name=None, x_is_unixtime=False, output_points=False):
    lines_ls = []
    speed_ls = []
    if output_points :
        start_points = [[] , []]
        finish_points = [[] , []]
        lc_points = [[] , []]
    for veh_id , veh_data in vehicles_data.items () :
        # if veh_id not in [508,553]:
        #     continue
        start_unix_time , ns_detaT = veh_data ['start_unix_time']
        detaT = ns_detaT / 1000
        lane_id = veh_data ['lane_id']
        frame_index = veh_data ['frame_index']
        # print(detaT)
        speed_y = veh_data['speed_y']
        # print(veh_data.keys())
        if driving_name :
            drivingline = veh_data ['drivingline']
        else :
            assert len (list (veh_data ['drivingline'].keys ())) == 1 , 'give the drivingline name'
            temp_driving_name = list (veh_data ['drivingline'].keys ()) [0]
            # print(temp_driving_name)
            drivingline = veh_data ['drivingline'] [temp_driving_name]
        # print(type(drivingline[0]))
        #drivingline_dist = smoothWithsEMA ([x [0] for x in drivingline] , 0.3 , detaT)
        drivingline_dist = []
        for i in drivingline:
            drivingline_dist.append(i[0])

        for i in range (len (lane_id) - 2) :
            if x_is_unixtime :
                x1 = (frame_index [i] - frame_index [0]) * ns_detaT + start_unix_time
                x2 = (frame_index [i + 1] - frame_index [0]) * ns_detaT + start_unix_time

            else :
                x1 = frame_index [i] * detaT
                x2 = frame_index [i + 1] * detaT
            if lane_id [i] == target_lane_id and lane_id [i + 1] == target_lane_id :
                y1 = drivingline_dist [i]
                y2 = drivingline_dist [i + 1]
                speed = abs(speed_y[i])*3.6
                #speed = abs ((y2 - y1) / detaT * 3.6)
                if abs(y1-y2) < 500 and abs(x1-x2)<500:    #如果间隔太远，取消之间的连线
                    speed_ls.append (speed)
                    lines_ls.append ([[x1 , y1] , [x2 , y2]])
            if output_points :
                if lane_id [i] != lane_id [i + 1] and (
                        lane_id [i] == target_lane_id or lane_id [i + 1] == target_lane_id) :
                    if lane_id [i] == target_lane_id :
                        lc_points [0].append (x1)
                        lc_points [1].append (drivingline_dist [i])
                    else :
                        lc_points [0].append (x2)
                        lc_points [1].append (drivingline_dist [i + 1])
        if output_points :
            if x_is_unixtime :
                x_s = start_unix_time
                x_e = (frame_index [-1] - frame_index [0]) * ns_detaT + start_unix_time
            else :
                x_s = frame_index [0] * detaT
                x_e = frame_index [-1] * detaT
            start_points [0].append (x_s)
            start_points [1].append (drivingline_dist [0])
            finish_points [0].append (x_e)
            finish_points [1].append (drivingline_dist [-1])
    if output_points :
        points = { 'start' : start_points , 'finish' : finish_points , 'lc' : lc_points }
        return lines_ls , speed_ls , points
    return lines_ls , speed_ls

def run_main(pkl_file):
    '''
    运行的入口函数
    :param multi_video_config:
    :return:
    '''
    yes = 0
    no = 0

    with open(pkl_file,'rb') as f:
        vehicle_data = pickle.load(f)


    lane_id_ls = [1,2,3,4,5,6,7,8]
    for lane_id in lane_id_ls:
        lines_ls, speed_ls,points = get_vehicles_from_lane(vehicle_data,lane_id, x_is_unixtime=False, output_points=True)
        fig_path = '/data3/liyitong/HuRong_process/sichuan_%d.jpg' %lane_id
        plot_line(fig_path, lines_ls, speed_ls)

    # lines_ls, speed_ls = get_vehicles_choose( vehicle_data,lane_id_ls)
    # fig_path = '/data3/liyitong/HuRong_process/sichuan_%d.jpg'
    # plot_line(fig_path, lines_ls, speed_ls)
        #print(lines_ls[1])

    pkl_file_1 = '/data3/liyitong/radar_sichuan/TRACK_K2_300_HEAD_2023121811.pkl'
    with open (pkl_file_1 , 'rb') as f :
        vehicle_data_1 = pickle.load (f)
    list_1 = list (vehicle_data_1.keys ( ))
    pkl_file_2 = '/data3/liyitong/radar_sichuan/TRACK_K1_828_HEAD_2023121811.pkl'
    with open (pkl_file_2 , 'rb') as f :
        vehicle_data_2 = pickle.load (f)
    list_2 = list (vehicle_data_2.keys ( ))
    yes = []

    # for li in list_1 :
    #     if li in list_2 :
    #         yes.append (li)

    lines_ls , speed_ls = get_vehicles_choose (vehicle_data_1, lane_id_ls, yes)
    fig_path = '/data3/liyitong/HuRong_process/sichuan_%d.jpg'


if __name__ == '__main__':
    pkl_file = '/data3/liyitong/radar_sichuan/TRACK_K1_230_HEAD_2023121811.pkl'
    run_main(pkl_file)
    # start_index,end_index = longestConsecutive(nums)
    #
    # print(start_index,end_index)
    # print(nums)
    # print(nums[start_index:end_index+1])