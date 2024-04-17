import pickle
import numpy as np
import pandas as pd
import sklearn as skl
import math
from toolbox.trajectory_process_1 import TrajectoryProcess,unixtime2time

def smoothWithsEMA(lsdata, T, dt=0.1):

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

def delete_horizontal_line_head_or_tail(vehicle_data,region,time_period):
    for veh_id,veh_data in vehicle_data.items():
        frame_index = veh_data ['frame_index']
        drivingline = veh_data ['drivingline'] ['mainroad']

        lane_id = veh_data ['lane_id']

        pixel_cpos_x = veh_data ['pixel_cpos_x']
        pixel_cpos_y = veh_data ['pixel_cpos_y']
        # geo_cpos_x = veh_data ['geo_cpos_x']
        # geo_cpos_y = veh_data ['geo_cpos_y']
        detaT = veh_data ['detaT']
        drivingline_dist = smoothWithsEMA ([x [0] for x in drivingline] , 0.3 , detaT)

        delete_index_list = []
        for i in range(len(lane_id) - 1):
            if drivingline_dist[i] > region[0] and drivingline_dist[i] < region[1]:
                if frame_index[i] > time_period[0] and frame_index[i] < time_period[1]:
                    speed = abs((drivingline_dist[i+1] - drivingline_dist[i])/detaT * 3.6)
                    if speed <= 15:
                        delete_index_list.append(i)

        if len(delete_index_list) >= 2:
            max_index_delete = max(delete_index_list)
            min_index_delete = min(delete_index_list)

            if abs(drivingline_dist[0] - drivingline_dist[max_index_delete]) >= 10:
                while abs (drivingline_dist [ 0 ] - drivingline_dist [ max_index_delete ])>=8 and (max_index_delete>min_index_delete) :
                    max_index_delete -= 1
                if abs (drivingline_dist [ 0 ] - drivingline_dist [ max_index_delete ])<= 10:
                    veh_data [ 'frame_index' ] = frame_index [ max_index_delete : ]
                    veh_data [ 'drivingline' ] [ 'mainroad' ] = drivingline [ max_index_delete : ]
                    veh_data [ 'lane_id' ] = lane_id [ max_index_delete : ]
                    veh_data [ 'pixel_cpos_x' ] = pixel_cpos_x [ max_index_delete : ]
                    veh_data [ 'pixel_cpos_y' ] = pixel_cpos_y [ max_index_delete : ]
                    # veh_data [ 'geo_cpos_x' ] = geo_cpos_x [ max_index_delete : ]
                    # veh_data [ 'geo_cpos_y' ] = geo_cpos_y [ max_index_delete : ]
            else:
                veh_data [ 'frame_index' ] = frame_index [ max_index_delete : ]
                veh_data [ 'drivingline' ] [ 'mainroad' ] = drivingline [ max_index_delete : ]
                veh_data [ 'lane_id' ] = lane_id [ max_index_delete : ]
                veh_data [ 'pixel_cpos_x' ] = pixel_cpos_x [ max_index_delete : ]
                veh_data [ 'pixel_cpos_y' ] = pixel_cpos_y [ max_index_delete : ]
                # veh_data [ 'geo_cpos_x' ] = geo_cpos_x [ max_index_delete : ]
                # veh_data [ 'geo_cpos_y' ] = geo_cpos_y [ max_index_delete : ]

            if abs(drivingline_dist[-1] - drivingline_dist[min_index_delete]) >= 10:
                while abs ( drivingline_dist [-1] - drivingline_dist [min_index_delete]) >= 8 and (min_index_delete<max_index_delete) :
                    min_index_delete += 1
                if abs ( drivingline_dist [-1] - drivingline_dist [min_index_delete]) <= 9:
                    veh_data ['frame_index'] = frame_index [:min_index_delete]
                    veh_data ['drivingline'] ['mainroad'] = drivingline [:min_index_delete]
                    veh_data ['lane_id'] = lane_id [:min_index_delete]
                    veh_data ['pixel_cpos_x'] = pixel_cpos_x [:min_index_delete]
                    veh_data ['pixel_cpos_y'] = pixel_cpos_y [:min_index_delete]
                    # veh_data ['geo_cpos_x'] = geo_cpos_x [:min_index_delete]
                    # veh_data ['geo_cpos_y'] = geo_cpos_y [:min_index_delete]
            else:
                veh_data [ 'frame_index' ] = frame_index [ :min_index_delete ]
                veh_data [ 'drivingline' ] [ 'mainroad' ] = drivingline [ :min_index_delete ]
                veh_data [ 'lane_id' ] = lane_id [ :min_index_delete ]
                veh_data [ 'pixel_cpos_x' ] = pixel_cpos_x [ :min_index_delete ]
                veh_data [ 'pixel_cpos_y' ] = pixel_cpos_y [ :min_index_delete ]
                # veh_data [ 'geo_cpos_x' ] = geo_cpos_x [ :min_index_delete ]
                # veh_data [ 'geo_cpos_y' ] = geo_cpos_y [ :min_index_delete ]

    return vehicle_data

def delete_short_trajectory_in_region(vehicle_data,region):
    delete_car = []
    for veh_id,veh_data in vehicle_data.items():
        frame_index = veh_data ['frame_index']
        drivingline_dist = veh_data ['drivingline'] ['mainroad']

        #drivingline_dist = smoothWithsEMA ([x [0] for x in drivingline] , 0.3 , detaT)
        if drivingline_dist[0][0] <= drivingline_dist[-1][0]:
            if drivingline_dist[0][0] >= region[0] and drivingline_dist[-1][0] <= region[1]:
                delete_car.append(veh_id)
        if drivingline_dist [0] [0]>=drivingline_dist [-1] [0] :
            if drivingline_dist [0][0] <= region [1] and drivingline_dist [-1][0]>=region [0] :
                delete_car.append (veh_id)
        if len(drivingline_dist) <= 10:
            delete_car.append (veh_id)
    unique_set = set (delete_car)
    car_to_delete = list (unique_set)
    for car in car_to_delete:
        del vehicle_data[car]
    return vehicle_data

def delete_horizontal_line_cut(vehicle_data,region,time_period):
    yes = 0
    veh_id_ls = list(vehicle_data.keys())
    max_veh_id = max(veh_id_ls) + 10
    new_veh_id_ls = []
    new_veh_data_ls = []
    for veh_id,veh_data in vehicle_data.items():
        frame_index = veh_data ['frame_index']
        drivingline = veh_data ['drivingline'] ['mainroad']

        lane_id = veh_data ['lane_id']

        pixel_cpos_x = veh_data ['pixel_cpos_x']
        pixel_cpos_y = veh_data ['pixel_cpos_y']
        geo_cpos_x = veh_data ['geo_cpos_x']
        geo_cpos_y = veh_data ['geo_cpos_y']
        detaT = veh_data ['detaT']
        drivingline_dist = smoothWithsEMA ([x [0] for x in drivingline] , 0.3 , detaT)

        delete_index_list = [ ]
        for i in range (len (lane_id) - 1) :
            if drivingline_dist [i]>region [0] and drivingline_dist [i]<region [1] and frame_index [i]>time_period [0] and frame_index [i]<time_period [1] :
                yes += 1
                speed = abs ((drivingline_dist [i + 1] - drivingline_dist [i]) / detaT * 3.6)
                if speed <= 15 :      #速度阈值设置处
                    delete_index_list.append (i)

        if len (delete_index_list)>=1 :
            max_index_delete = max (delete_index_list)
            min_index_delete = min (delete_index_list)
            if len (delete_index_list)<= 10:
                max_index_delete = max (delete_index_list)
                min_index_delete = min (delete_index_list)
            if len(frame_index [: min_index_delete]) >= 40:
                veh_data ['frame_index'] = frame_index [: min_index_delete]
                veh_data ['drivingline'] ['mainroad'] = drivingline [: min_index_delete]
                veh_data ['lane_id'] = lane_id [: min_index_delete]
                veh_data ['pixel_cpos_x'] = pixel_cpos_x [: min_index_delete]
                veh_data ['pixel_cpos_y'] = pixel_cpos_y [: min_index_delete]
                veh_data ['geo_cpos_x'] = geo_cpos_x [: min_index_delete]
                veh_data ['geo_cpos_y'] = geo_cpos_y [: min_index_delete]

            if len (frame_index [max_index_delete:])>=40 :
                new_veh_data = {}
                new_veh_data ['frame_index'] = frame_index [max_index_delete:]
                drivingline_new = {}
                drivingline_new['mainroad'] = drivingline [max_index_delete:]
                new_veh_data ['drivingline'] = drivingline_new
                new_veh_data ['pixel_cpos_x'] = pixel_cpos_x [max_index_delete:]
                new_veh_data ['lane_id'] = lane_id [max_index_delete:]
                new_veh_data ['pixel_cpos_y'] = pixel_cpos_y [max_index_delete:]
                new_veh_data ['geo_cpos_x'] = geo_cpos_x [max_index_delete:]
                new_veh_data ['geo_cpos_y'] = geo_cpos_y [max_index_delete:]
                new_veh_data ['start_unix_time'] = (frame_index[max_index_delete],100)
                new_veh_data ['vehicle_length'] = veh_data ['vehicle_length']
                new_veh_data ['detaT'] = 0.1

                max_veh_id += 10
                new_veh_id_ls.append(max_veh_id)
                new_veh_data_ls.append(new_veh_data)

    for index in range(len(new_veh_id_ls)):
        new_veh_id = new_veh_id_ls[index]
        vehicle_data[new_veh_id] = new_veh_data_ls[index]
    print(yes)
    return vehicle_data


def run_main(pkl_file):
    with open(pkl_file,'rb') as f:
        vehicle_data = pickle.load(f)
    region_1 = [460,500]
    time_period_1 = [0 , 9000]
    vehicle_data_new_1 = delete_horizontal_line_head_or_tail(vehicle_data,region_1,time_period_1)

    # region_2 = [290 , 360]
    # vehicle_data_new_2 = delete_short_trajectory_in_region(vehicle_data,region_2)

    # region_3 = [396,404]
    # time_period_3 = [7280 , 7380]
    # vehicle_data_new_3 = delete_horizontal_line_cut (vehicle_data,region_3,time_period_3)

    with open ('/data3/liyitong/HuRong_process/A2/20220616_0700_A2_F1_374_1_Num_4/tp_result_20220616_0700_A2_F1_374_1_new_3.tppkl' ,'wb') as f :
        pickle.dump (vehicle_data_new_1 , f)

if __name__ == '__main__':
    pkl_file = '/data3/liyitong/HuRong_process/A2/20220616_0700_A2_F1_374_1_Num_4/tp_result_20220616_0700_A2_F1_374_1_new_2.tppkl'
    run_main(pkl_file)