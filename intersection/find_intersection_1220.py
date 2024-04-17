import pickle
from typing import Dict, Any

import numpy as np
import pandas as pd
from toolbox.trajectory_process_1 import TrajectoryProcess,unixtime2time
import openpyxl



def run_main(pkl_file):
    '''
    运行的入口函数
    :param multi_video_config:
    :return:
    '''
    intersection_dict = {}
    intersection_find = 50
    movement_ls = ['straight_1_to_6','straight_3_to_8','straight_5_to_2','straight_7_to_4',
                   'right_1_to_8','right_3_to_2','right_5_to_4','right_7_to_6',
                   'left_1_to_4','left_3_to_6','left_5_to_8','left_7_to_2']

    veh_id_ls = []
    straight_1_to_6 = []
    straight_3_to_8 = []
    straight_5_to_2 = []
    straight_7_to_4 = []
    right_1_to_8 = []
    right_3_to_2 = []
    right_5_to_4 = []
    right_7_to_6 = []
    left_1_to_4 = []
    left_3_to_6 = []
    left_5_to_8 =[]
    left_7_to_2 = []

    with open(pkl_file,'rb') as f:
        vehicle_data = pickle.load(f)
    #count = 0
    #veh = 0
    for veh_id ,veh_data in vehicle_data.items():
        #veh += 1
        lane_id_intersection = []
        lane_id = veh_data ['lane_id']
        #print(lane_id)
        frame_index = veh_data ['frame_index']
        drivingline = veh_data['drivingline']['mainroad']
        for i in range(len(frame_index)):
            #lane_id_last = -1
            lane_id_now = lane_id[i]
            if i == 0:
                lane_id_last = lane_id[i]
                if (drivingline[i][0] <= 390) and (lane_id_now != -1):
                    lane_id_intersection.append(lane_id_last)

            if lane_id_now != lane_id_last:
                lane_id_last = lane_id_now
                if (lane_id_now != -1) and (drivingline [i] [0] <= 390):
                    lane_id_intersection.append (lane_id_now)
        #print(lane_id_intersection)
        #以上找到了车辆经过的车道

        index_ls = []
        if len(lane_id_intersection) == 0:
            continue
        if len(lane_id_intersection) >= 2:        #至少有一次经过交叉口，及另一个其他进出口道的编号
            for j in range(len(lane_id_intersection)):
                if lane_id_intersection[j] == intersection_find:
                    index_ls.append(j)

            #if len(index_ls) == 1:
                #index = index_ls[0]
            #elif len (index_ls) > 1:             #如果多次经过交叉口，先还是以最后一次为准
                #index = index = index_ls[-1]
            for times in range(len(index_ls)):
                index = index_ls[times]
                if (index - 1 >= 0) and (index + 1 < len(lane_id_intersection)):
                    from_lane = lane_id_intersection [index - 1]
                    to_lane = lane_id_intersection [index + 1]
                    if from_lane == 1 or from_lane == 2:
                        if to_lane == 5 or to_lane == 6:
                            straight_1_to_6.append (veh_id)
                        if to_lane == 7 or to_lane == 8:
                            right_1_to_8.append (veh_id)
                        if to_lane == 3 or to_lane == 4:
                            left_1_to_4.append (veh_id)
                    if from_lane == 3 or from_lane == 4:
                        if to_lane == 7 or to_lane == 8:
                            straight_3_to_8.append (veh_id)
                        if to_lane == 1 or to_lane == 2:
                            right_3_to_2.append (veh_id)
                        if to_lane == 5 or to_lane == 6:
                            left_3_to_6.append (veh_id)
                    if from_lane == 5 or from_lane == 6:
                        if to_lane == 1 or to_lane == 2:
                            straight_5_to_2.append (veh_id)
                        if to_lane == 3 or to_lane == 4:
                            right_5_to_4.append (veh_id)
                        if to_lane == 7 or to_lane == 8:
                            left_5_to_8.append (veh_id)
                    if from_lane == 7 or from_lane == 8:
                        if to_lane == 3 or to_lane == 4:
                            straight_7_to_4.append (veh_id)
                        if to_lane == 5 or to_lane == 6:
                            right_7_to_6.append (veh_id)
                        if to_lane == 1 or to_lane == 2:
                            left_7_to_2.append (veh_id)
                elif index - 1 < 0:
                    to_lane = lane_id_intersection [index + 1]
                    if to_lane == 1 or to_lane == 2:
                        straight_5_to_2.append (veh_id)
                    if to_lane == 3 or to_lane == 4:
                        straight_7_to_4.append (veh_id)
                    if to_lane == 5 or to_lane == 6:
                        straight_1_to_6.append (veh_id)
                    if to_lane == 7 or to_lane == 8:
                        straight_3_to_8.append (veh_id)
                elif index + 1 >= len(lane_id_intersection):
                    from_lane = lane_id_intersection [index - 1]
                    if from_lane == 1 or from_lane == 2:
                        straight_1_to_6.append (veh_id)
                    if from_lane == 3 or from_lane == 4:
                        straight_3_to_8.append (veh_id)
                    if from_lane == 5 or from_lane == 6:
                        straight_5_to_2.append (veh_id)
                    if from_lane == 7 or from_lane == 8:
                        straight_7_to_4.append (veh_id)
        if len (lane_id_intersection) == 1:
            lane_id_one = lane_id_intersection[0]
            if lane_id_one == 1 :
                straight_1_to_6.append(veh_id)
            elif lane_id_one == 2:
                straight_5_to_2.append(veh_id)
            elif lane_id_one == 3:
                straight_3_to_8.append(veh_id)
            elif lane_id_one == 4:
                straight_7_to_4.append(veh_id)
            elif lane_id_one == 5:
                straight_5_to_2.append(veh_id)
            elif lane_id_one == 6:
                straight_1_to_6.append(veh_id)
            elif lane_id_one == 7:
                straight_7_to_4.append(veh_id)
            elif lane_id_one == 8:
                straight_3_to_8.append(veh_id)


    intersection_dict ['straight_1_to_6'] = straight_1_to_6
    intersection_dict ['straight_3_to_8'] = straight_3_to_8
    intersection_dict ['straight_5_to_2'] = straight_5_to_2
    intersection_dict ['straight_7_to_4'] = straight_7_to_4
    intersection_dict ['right_1_to_8'] = right_1_to_8
    intersection_dict ['right_3_to_2'] = right_3_to_2
    intersection_dict ['right_5_to_4'] = right_5_to_4
    intersection_dict ['right_7_to_6'] = right_7_to_6
    intersection_dict ['left_1_to_4'] = left_1_to_4
    intersection_dict ['left_3_to_6'] = left_3_to_6
    intersection_dict ['left_5_to_8'] = left_5_to_8
    intersection_dict ['left_7_to_2'] = left_7_to_2

    file_name = "/data3/liyitong/HuRong_process/intersection/D2/20220616_0700_D2_F1_370_1_Num_4/movements_veh_id.pkl"

    # 使用 pickle.dump() 将字典保存到文件
    with open (file_name, "wb") as file:
        pickle.dump (intersection_dict, file)
    #movement_ls = ['straight_1_to_6','straight_3_to_8','straight_5_to_2','straight_7_to_4',
                   #'right_1_to_8','right_3_to_2','right_5_to_4','right_7_to_6'
                   #'left_1_to_4','left_3_to_6','left_5_to_8','left_7_to_2']

    #df.to_excel ("/data3/liyitong/HuRong_process/lane_id.xlsx", index=True)  # index=False表示不保存行索引
if __name__ == '__main__':

    pkl_file = '/data3/liyitong/HuRong_process/intersection/D2/20220616_0700_D2_F1_370_1_Num_4/tp_result_20220616_0700_D2_F1_370_1.tppkl'
    run_main(pkl_file)
    # start_index,end_index = longestConsecutive(nums)
    #
    # print(start_index,end_index)
    # print(nums)
    # print(nums[start_index:end_index+1])