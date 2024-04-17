import pickle
import numpy as np
import pandas as pd
import sklearn as skl

from toolbox.trajectory_process_1 import TrajectoryProcess,unixtime2time


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

    tp = TrajectoryProcess()
    tp.vehicles_data = vehicle_data

    lane_id_ls = [1,2,3,4,5,6,7,20,21,22,23,24,121]
    for lane_id in lane_id_ls:
        lines_ls, speed_ls, points = tp.get_vehicles_from_lane(lane_id, x_is_unixtime=False, output_points=True)
        #print(lines_ls)
        fig_path = '/data3/liyitong/HuRong_process/B2_%d.jpg' % lane_id
        tp.plot_line(fig_path, lines_ls, speed_ls, points=points)
        #print(lines_ls[1])

if __name__ == '__main__':
    pkl_file = '/data3/liyitong/HuRong_process/B1/20220616_0700_B1_F1_373_1_Num_4/tp_result_20220616_0700_B1_F1_373_1.tppkl'
    run_main(pkl_file)
    # start_index,end_index = longestConsecutive(nums)
    #
    # print(start_index,end_index)
    # print(nums)
    # print(nums[start_index:end_index+1])