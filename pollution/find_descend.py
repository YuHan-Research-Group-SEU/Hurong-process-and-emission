import pickle
import numpy as np
import pandas as pd
from toolbox.trajectory_process_1 import TrajectoryProcess,unixtime2time


def run_main(pkl_file):
    '''
    运行的入口函数
    :param multi_video_config:
    :return:
    '''
    with open(pkl_file,'rb') as f:
        vehicle_data = pickle.load(f)
    #for veh_id ,veh_data in vehicle_data.items():
        #print(type(veh_id))
        #print(type(veh_data))

        #print(vehicle_data['lane_id'])


    tp = TrajectoryProcess()
    tp.vehicles_data = vehicle_data

    lane_id_ls = [1,2,3,4,5]
    for lane_id in lane_id_ls:
        lines_ls, speed_ls, points = tp.get_vehicles_from_lane(lane_id, x_is_unixtime=False, output_points=True)
        #print(lines_ls)
        fig_path = '/data3/liyitong/HuRong_process/laneold%d.jpg' % lane_id
        tp.plot_line(fig_path, lines_ls, speed_ls, points=points)
        print(lines_ls[1])

if __name__ == '__main__':
    pkl_file = '/data3/liyitong/HuRong_process/B1/20220616_0815_B1_F4_371_1_Num_5/tp_result_20220616_0815_B1_F4_371_1.tppkl'
    # nums = [1,2,3,5,6,7,9,10,11,12,13,14,18]
    run_main(pkl_file)
    # start_index,end_index = longestConsecutive(nums)
    #
    # print(start_index,end_index)
    # print(nums)
    # print(nums[start_index:end_index+1])