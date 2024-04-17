import pickle
import numpy as np
import pandas as pd
import sklearn as skl
import time
from datetime import datetime
from toolbox.trajectory_process_1 import TrajectoryProcess,unixtime2time


def run_main(pkl_file):
    '''
    运行的入口函数
    :param multi_video_config:
    :return:
    '''
    yes = 0
    no = 0
    start_unix_time_small = 999999999999999999999999999
    with open(pkl_file,'rb') as f:
        vehicle_data = pickle.load(f)
    for veh_id ,veh_data in vehicle_data.items():
        start_unix_time = veh_data['start_unix_time'][0]
        if start_unix_time <= start_unix_time_small:
            start_unix_time_small = start_unix_time
    #print(start_unix_time_small)
    veh_choose = []
    start_unix_time_big = start_unix_time_small +250*1000

    for veh_id ,veh_data in vehicle_data.items():
        start_unix_time = veh_data['start_unix_time'][0]
        drivingline_min = veh_data['drivingline']['mainroad'][0][0]
        drivingline_max = veh_data ['drivingline'] ['mainroad'] [-1] [0]
        if (start_unix_time <= start_unix_time_big) and (drivingline_min<=10) and (drivingline_max>=3800):
            veh_choose.append(veh_id)
    #print(veh_choose)

    tp = TrajectoryProcess()
    tp.vehicles_data = vehicle_data


    lane_id_ls = [1,2,3,4,5]
    #for lane_id in lane_id_ls:
    lines_ls, speed_ls = tp.get_vehicles_choose(lane_id_ls,veh_choose, x_is_unixtime=False, output_points=False)

    fig_path = '/data3/liyitong/HuRong_process/connect_4000_%d.jpg'
    tp.plot_line(fig_path, lines_ls, speed_ls)
if __name__ == '__main__':
    pkl_file = '/home/liyitong/Workspace/GCVTM/data/output/20220617/stitch_test/multi_20220617_D1toA1_F2/stitch_tppkl_multi_20220617_D1toA1_F2_new5.tppkl'
    run_main(pkl_file)
    # start_index,end_index = longestConsecutive(nums)
    #
    # print(start_index,end_index)
    # print(nums)
    # print(nums[start_index:end_index+1])