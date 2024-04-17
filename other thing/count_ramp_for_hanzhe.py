import pickle
import numpy as np
import pandas as pd
import sklearn as skl
import math

from toolbox.trajectory_process_1 import TrajectoryProcess,unixtime2time
import datetime

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

def count_speed_volume_section(vehicle_data,start_frame,end_frame,target_lane_id,section):
    speed_ls = []
    volume = 0
    speed = 0
    for veh_id , veh_data in vehicle_data.items ( ) :
        frame_index = veh_data ['frame_index']
        drivingline = veh_data ['drivingline'] ['mainroad']
        lane_id = veh_data ['lane_id']
        pixel_cpos_x = veh_data ['pixel_cpos_x']
        pixel_cpos_y = veh_data ['pixel_cpos_y']
        geo_cpos_x = veh_data ['geo_cpos_x']
        geo_cpos_y = veh_data ['geo_cpos_y']
        detaT = veh_data ['detaT']
        start_unix_time = veh_data ['start_unix_time']
        drivingline_dist = smoothWithsEMA ([x [0] for x in drivingline] , 0.3 , detaT)

        for i in range(len(frame_index) - 2):
            if frame_index[i] >= start_frame and frame_index[i] <= end_frame and lane_id[i] in target_lane_id:
                if drivingline_dist[i] <= section and drivingline_dist[i + 1] >= section:
                    volume += 1

                    angle = math.radians (8.32)   #角度变成弧度
                    # 计算正弦值
                    cos_value = math.cos (angle)
                    #print(cos_value)
                    speed_this_veh = abs(drivingline_dist[i - 2] - drivingline_dist[ i + 2])/0.4 * 3.6
                    speed_ls.append(speed_this_veh)
    speed = sum (speed_ls) / len (speed_ls)
    return volume , speed

def run_main(pkl_file):

    with open(pkl_file,'rb') as f:
        vehicle_data = pickle.load(f)
    start_unix_time_min = 165542939629099999999
    start_unix_time_max = 0

    frame_min = 165542939629099999
    frame_max = 0
    start_unix_time_max = 0
    for veh_id,veh_data in vehicle_data.items():
        frame_index = veh_data ['frame_index']
        drivingline = veh_data ['drivingline'] ['mainroad']
        lane_id = veh_data ['lane_id']
        pixel_cpos_x = veh_data ['pixel_cpos_x']
        pixel_cpos_y = veh_data ['pixel_cpos_y']
        geo_cpos_x = veh_data ['geo_cpos_x']
        geo_cpos_y = veh_data ['geo_cpos_y']
        detaT = veh_data ['detaT']

        # print(geo_cpos_x[0])
        # print(geo_cpos_y[0])
        start_unix_time = veh_data ['start_unix_time']
        if start_unix_time[0] <= start_unix_time_min:
            start_unix_time_min = start_unix_time[0]
        if start_unix_time[0] >= start_unix_time_max :
            start_unix_time_max = start_unix_time[0]


        if frame_index[0] <= frame_min:
            frame_min = frame_index[0]
        if frame_index[-1] >= frame_max:
            frame_max = frame_index[-1]


    period = frame_max//6000 + 1
    period = int(period)
    list_frame = []
    for i in range(period):
        list_frame.append(i * 6000)
    list_frame.append(frame_max)

    print("整个视频开始时间")
    start_true = datetime.datetime.utcfromtimestamp (start_unix_time_min/1000)
    print(start_true)
    # print(frame_min,frame_max)
    # normal_time_min = datetime.datetime.utcfromtimestamp (start_unix_time_min)
    #
    # print (normal_time_min)
    target_lane_id_ls = [1,2,3,4,5]
    section = 100
    for j in range(len(list_frame) - 1):
        period_start = start_unix_time_min + list_frame[j] * 100
        period_end = start_unix_time_min + list_frame [j + 1] * 100

        period_start = datetime.datetime.utcfromtimestamp (period_start/1000)
        period_end = datetime.datetime.utcfromtimestamp (period_end/1000)
        print("开始时间")
        print(period_start)
        print("结束时间")
        print (period_end)
        print("持续时间")
        print((list_frame [j + 1]-list_frame[j])/10)
        for target_lane_id_int in target_lane_id_ls:
            target_lane_id = [target_lane_id_int]
            volume,speed = count_speed_volume_section (vehicle_data,list_frame[j] , list_frame[j + 1] , target_lane_id , section)
            print(target_lane_id)
            print(volume)
            print(speed)



if __name__ == '__main__':

    # 注意在计算速度时，需要考虑匝道角度问题，需要检查代码
    # file = '/data3/liyitong/HuRong_process/C1/20220616_0700_C1_F1_370_1_Num_5/tp_result_20220616_0700_C1_F1_370_1.tppkl'
    file = '/data3/liyitong/HuRong_process/C1/20220616_0725_C1_F2_370_1_Num_5/tp_result_20220616_0725_C1_F2_370_1.tppkl'
    file = '/data3/liyitong/HuRong_process/C1/20220616_0750_C1_F3_371_1_Num_5/tp_result_20220616_0750_C1_F3_371_1.tppkl'
    file = '/data3/liyitong/HuRong_process/C1/20220616_0815_C1_F4_370_1_Num_5/tp_result_20220616_0815_C1_F4_370_1.tppkl'
    file = '/data3/liyitong/HuRong_process/C1/20220616_0845_C1_F5_371_1_Num_5/tp_result_20220616_0845_C1_F5_371_1.tppkl'


    run_main(file)
    # start_index,end_index = longestConsecutive(nums)
    #
    # print(start_index,end_index)
    # print(nums)
    # print(nums[start_index:end_index+1])