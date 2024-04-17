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
import sklearn
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import re

from toolbox.trajectory_process_1 import TrajectoryProcess,unixtime2time

def extract_numbers(input_string):
    # 使用正则表达式匹配连续的数字
    numbers = re.findall(r'\d+', input_string)

    # 将匹配到的数字列表转换为整数
    numbers = [int(num) for num in numbers]

    return numbers

def smoothWithsEMA (lsdata, T, dt=0.1):
    """
    s�pn(��p��sG�
    :return:
    """
    Na = len (lsdata)
    deta = T / dt
    outData = []
    for i in range (Na):
        D = min ([3 * deta, i, Na - i - 1])
        lsgt = []
        lsxe = []
        for k in range (int (i - D), int (i + D + 1)):
            gt = pow (math.e, -abs (i - k) / deta)
            xe = lsdata [k] * gt
            lsgt.append (gt)
            lsxe.append (xe)
        outX = sum (lsxe) / sum (lsgt)
        outData.append (outX)
    return outData


def get_vehicles_from_lane (vehicles_data,driving_name=None , x_is_unixtime=False , output_points=False):
    lines_ls = []
    speed_ls = []
    if output_points:
        start_points = [[] , []]
        finish_points = [[] , []]
        lc_points = [[] , []]
    for veh_id , veh_data in vehicles_data.items ():
        # if veh_id not in [508,553]:
        #     continue
        start_unix_time , ns_detaT = veh_data ['start_unix_time']
        detaT = ns_detaT / 1000
        lane_id = veh_data ['lane_id']
        frame_index = veh_data ['frame_index']
        # print(detaT)

        # print(veh_data.keys())
        if driving_name:
            drivingline = veh_data ['drivingline']
        else:
            assert len (list (veh_data ['drivingline'].keys ())) == 1 , 'give the drivingline name'
            temp_driving_name = list (veh_data ['drivingline'].keys ()) [0]
            # print(temp_driving_name)
            drivingline = veh_data ['drivingline'] [temp_driving_name]
        # print(type(drivingline[0]))
        drivingline_dist = smoothWithsEMA ([x [0] for x in drivingline] , 0.3 , detaT)
        #drivingline_dist = [x [0] for x in drivingline]
        for i in range (len (lane_id) - 1):
            if x_is_unixtime:
                x1 = (frame_index [i] - frame_index [0]) * ns_detaT + start_unix_time
                x2 = (frame_index [i + 1] - frame_index [0]) * ns_detaT + start_unix_time

            else:
                x1 = frame_index [i] * detaT
                x2 = frame_index [i + 1] * detaT
            #if lane_id [i] == target_lane_id and lane_id [i + 1] == target_lane_id:
            y1 = drivingline_dist [i]
            y2 = drivingline_dist [i + 1]
            speed = abs ((y2 - y1) / detaT * 3.6)
            speed_ls.append (speed)
            lines_ls.append ([(x1 , y1) , (x2 , y2)])
            if output_points:
                if lane_id [i] != lane_id [i + 1]:

                    lc_points [0].append (x1)
                    lc_points [1].append (drivingline_dist [i])

        if output_points:
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
        points = {'start': start_points , 'finish': finish_points , 'lc': lc_points}
        return lines_ls , speed_ls , points
    return lines_ls , speed_ls


def plot_line (savepath ,lines ,color_speed ,figsize=(15 ,5) ,start_time=None ,points=None):
    fig = plt.figure (figsize = figsize)
    # fig, ax = plt.subplots(figsize=(15, 5))
    ax = fig.add_subplot ()
    lines_sc = LineCollection (lines ,array = np.array (color_speed) ,cmap = "jet_r" ,linewidths = 0.2)
    ax.add_collection (lines_sc)
    lines_sc.set_clim (vmin = 0 ,vmax = 120)
    cb = fig.colorbar (lines_sc)
    if not start_time is None:
        plt.title ('Start time:%s' % start_time ,fontsize = 20)
    if not points is None:
        size = 1
        # start_points = points.get('start',None)
        # if not start_points is None:
        #     plt.scatter(start_points[0],start_points[1],c='k',marker='^',label='starting point',s=size)
        # end_points = points.get('finish', None)
        # if not end_points is None:
        #     plt.scatter(end_points[0], end_points[1], c='k', marker='s',label='finishing point',s=size)
        lc_points = points.get ('lc' ,None)
        if not lc_points is None:
            plt.scatter (lc_points [0] ,lc_points [1] ,c = 'k' ,marker = 'x' ,label = 'lane changing point' ,s = size)
        plt.legend (loc = 'upper right' ,fontsize = 16)
    ax.autoscale ()
    plt.xticks (fontsize = 18)
    plt.yticks (fontsize = 18)
    # 设置colorbar刻度的字体大小
    cb.ax.tick_params (labelsize = 18)
    cb.ax.set_title ('speed [km/h]' ,fontsize = 18)
    plt.xlabel ("Time [s]" ,fontsize = 18)
    plt.ylabel ("Location [m]" ,fontsize = 18)

    # plt.grid(None)
    # plt.show()
    plt.savefig (savepath ,dpi = 1000 ,bbox_inches = 'tight')
    print ('save_img:%s' % savepath)

def delete_other_intersection(intersection_vehicles_data,target_lane_id):
    if 'lane_dist' in intersection_vehicles_data:
        del intersection_vehicles_data ['lane_dist']
    lane_id = intersection_vehicles_data['lane_id']

    frame_index = intersection_vehicles_data['frame_index']
    drivingline = intersection_vehicles_data['drivingline']['mainroad']
    pixel_cpos_x = intersection_vehicles_data['pixel_cpos_x']
    pixel_cpos_y = intersection_vehicles_data['pixel_cpos_y']
    geo_cpos_x = intersection_vehicles_data['geo_cpos_x']
    geo_cpos_y = intersection_vehicles_data['geo_cpos_y']
    start_unix_time = intersection_vehicles_data['start_unix_time']
    vehicle_length = intersection_vehicles_data['vehicle_length']
    detaT=intersection_vehicles_data['detaT']
    delete_index = []
    yes = 0
    for i in range((len(lane_id)-1)):
        if lane_id[i] in target_lane_id:
            yes+=0
        else:
            delete_index.append(i)
    delete_index.sort (reverse = True)
    for index in delete_index:
        del lane_id [index]
        frame_index = np.delete (frame_index ,index)
        del drivingline [index]
        del pixel_cpos_x [index]
        del pixel_cpos_y [index]
        del geo_cpos_x [index]
        del geo_cpos_y [index]

    this_intersection_veh = {}
    drivingline_dict = {}
    drivingline_dict['mainroad'] = drivingline

    this_intersection_veh ['lane_id'] = lane_id
    this_intersection_veh ['frame_index'] = frame_index
    this_intersection_veh ['drivingline'] = drivingline_dict
    this_intersection_veh ['pixel_cpos_x'] = pixel_cpos_x
    this_intersection_veh ['pixel_cpos_y'] = pixel_cpos_y
    this_intersection_veh ['geo_cpos_x'] = geo_cpos_x
    this_intersection_veh ['geo_cpos_y'] = geo_cpos_y
    this_intersection_veh ['start_unix_time'] = start_unix_time
    this_intersection_veh ['vehicle_length'] = vehicle_length
    this_intersection_veh ['detaT'] = detaT

    return this_intersection_veh
def run_main(tppkl_list,save_path,movement_file,target_lane_id_ori):
    one_intersection = {}
    with open(movement_file,'rb') as f:
        intersection_movements = pickle.load(f)

    for movement, movement_veh_id_list in intersection_movements.items():
        target_lane_id = target_lane_id_ori
        result = extract_numbers (movement)
        target_lane_id.extend(result)
        for tppkl_file in tppkl_list:
            if movement in tppkl_file:
                this_movement_veh = {}
                with open (tppkl_file, 'rb') as f:
                    intersection_vehicles = pickle.load (f)
                for movement_veh_id in movement_veh_id_list:
                    #this_intersection_veh = delete_other_intersection(intersection_vehicles[movement_veh_id],target_lane_id)
                    #不需要删除了，因为画图的时候就没有画另一个交叉口的drivingline
                    this_movement_veh[movement_veh_id] = intersection_vehicles[movement_veh_id]
                one_intersection[movement] = this_movement_veh
    print(one_intersection.keys())
    with open (save_path, "wb") as file:
        pickle.dump (one_intersection, file)

#以下为画图
    for movement, movement_veh in one_intersection.items():
        fig_path = '/data3/liyitong/HuRong_process/intersection/intersection_D2_one/0616_F1/D2_one_0616_F1_%s.jpg' % movement

        lines_ls, speed_ls, points = get_vehicles_from_lane (movement_veh,  output_points = True)

        plot_line (fig_path, lines_ls, speed_ls, points = points)

if __name__ == '__main__':
    tppkl_list = ['/data3/liyitong/HuRong_process/intersection/D2/20220616_0700_D2_F1_370_1_Num_4/tp_result_20220616_0700_D2_F1_left_1_to_4.tppkl',
                  '/data3/liyitong/HuRong_process/intersection/D2/20220616_0700_D2_F1_370_1_Num_4/tp_result_20220616_0700_D2_F1_left_3_to_6.tppkl',
                  '/data3/liyitong/HuRong_process/intersection/D2/20220616_0700_D2_F1_370_1_Num_4/tp_result_20220616_0700_D2_F1_left_5_to_8.tppkl',
                  '/data3/liyitong/HuRong_process/intersection/D2/20220616_0700_D2_F1_370_1_Num_4/tp_result_20220616_0700_D2_F1_left_7_to_2.tppkl',
                  '/data3/liyitong/HuRong_process/intersection/D2/20220616_0700_D2_F1_370_1_Num_4/tp_result_20220616_0700_D2_F1_right_1_to_8.tppkl',
                  '/data3/liyitong/HuRong_process/intersection/D2/20220616_0700_D2_F1_370_1_Num_4/tp_result_20220616_0700_D2_F1_right_3_to_2.tppkl',
                  '/data3/liyitong/HuRong_process/intersection/D2/20220616_0700_D2_F1_370_1_Num_4/tp_result_20220616_0700_D2_F1_right_5_to_4.tppkl',
                  '/data3/liyitong/HuRong_process/intersection/D2/20220616_0700_D2_F1_370_1_Num_4/tp_result_20220616_0700_D2_F1_right_7_to_6.tppkl',
                  '/data3/liyitong/HuRong_process/intersection/D2/20220616_0700_D2_F1_370_1_Num_4/tp_result_20220616_0700_D2_F1_straight_1_to_6.tppkl',
                  '/data3/liyitong/HuRong_process/intersection/D2/20220616_0700_D2_F1_370_1_Num_4/tp_result_20220616_0700_D2_F1_straight_3_to_8.tppkl',
                  '/data3/liyitong/HuRong_process/intersection/D2/20220616_0700_D2_F1_370_1_Num_4/tp_result_20220616_0700_D2_F1_straight_5_to_2.tppkl',
                  '/data3/liyitong/HuRong_process/intersection/D2/20220616_0700_D2_F1_370_1_Num_4/tp_result_20220616_0700_D2_F1_straight_7_to_4.tppkl']

    save_path = '/data3/liyitong/HuRong_process/intersection/intersection_D2_one/0616_F1/intersection_D2_one_20220616_0700_D2_F1.pkl'

    movement_file = "/data3/liyitong/HuRong_process/intersection/D2/20220616_0700_D2_F1_370_1_Num_4/movements_veh_id.pkl"

    target_lane_id = [50]

    run_main(tppkl_list,save_path,movement_file,target_lane_id)
    # start_index,end_index = longestConsecutive(nums)
    #
    # print(start_index,end_index)
    # print(nums)
    # print(nums[start_index:end_index+1])