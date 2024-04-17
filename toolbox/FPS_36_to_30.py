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


def save_data2pkl(vehicles_data, file_name):
    '''
    保存tppkl文件
    :param vehicles_data:
    :param file_name:
    :return:
    '''
    print('start save data:%s'%file_name)
    with open(file_name, 'wb') as f:
        pickle.dump(vehicles_data, f)



def run_main(pkl_file):
    '''
    运行的入口函数
    :param multi_video_config:
    :return:
    '''
    with open(pkl_file,'rb') as f:
        vehicle_data = pickle.load(f)
    #print(vehicle_data)
    new_vehicle_data = {}
    for veh_id, veh_data in vehicle_data.items ():
        frame_index_new = []
        drivingline_new = []
        lane_id_new = []
        #lane_dist_new = []
        pixel_cpos_x_new = []
        pixel_cpos_y_new = []
        geo_cpos_x_new = []
        geo_cpos_y_new = []
        #start_unix_time_new = ()
        #vehicle_length_new = 0.0
        detaT = 0.1

        frame_index = veh_data['frame_index']
        drivingline = veh_data['drivingline']['mainroad']
        #print(drivingline)
        lane_id = veh_data['lane_id']
        #lane_dist = veh_data['lane_dist']
        pixel_cpos_x = veh_data['pixel_cpos_x']
        pixel_cpos_y = veh_data['pixel_cpos_y']
        geo_cpos_x = veh_data['geo_cpos_x']
        geo_cpos_y = veh_data['geo_cpos_y']

        unix_time = veh_data['start_unix_time'][0]
        m_second = 100
        tup = (unix_time,m_second)
        start_unix_time = tup
        vehicle_length = veh_data['vehicle_length']

        start_unix_time_new = start_unix_time
        vehicle_length_new = vehicle_length

        sec = (len(frame_index) // 12)
        #print (sec)
        #flag = False
        for i in range(sec + 1):
            #flag = False
            for j in range(10):
                end = 12 * i + j + 2
                #print(end)
                if end >= len(frame_index):
                    #print("yes")
                    #flag = True
                    break
                if j == 0:
                    #print("yes")
                    frame = 10 * i + j + (frame_index[0]//12 * 10) + frame_index[0] % 12
                    frame_index_new.append(frame)

                    line_0 = drivingline[12 * i + j][0] + 0.2 * (drivingline[12 * i + j + 1][0] - drivingline[12 * i + j][0])
                    line_1 = drivingline[12 * i + j][1] + 0.2 * (drivingline[12 * i + j + 1][1] - drivingline[12 * i + j][1])
                    line = [line_0,line_1]
                    drivingline_new.append(line)

                    l_id = lane_id[12 * i + j]
                    lane_id_new.append(l_id)

                    pixel_x = pixel_cpos_x[12 * i + j] + 0.2 * (pixel_cpos_x[12 * i + j + 1] - pixel_cpos_x[12 * i + j])
                    pixel_y = pixel_cpos_y[12 * i + j] + 0.2 * (pixel_cpos_y[12 * i + j + 1] - pixel_cpos_y[12 * i + j])
                    geo_x = geo_cpos_x[12 * i + j] + 0.2 * (geo_cpos_x[12 * i + j + 1] - geo_cpos_x[12 * i + j])
                    geo_y = geo_cpos_y[12 * i + j] + 0.2 * (geo_cpos_y[12 * i + j + 1] - geo_cpos_y[12 * i + j])

                    #print(geo_y)
                    pixel_cpos_x_new.append(pixel_x)
                    pixel_cpos_y_new.append(pixel_y)
                    geo_cpos_x_new.append(geo_x)
                    geo_cpos_y_new.append(geo_y)

                elif j == 1:
                    #print ("yes")
                    frame = 10 * i + j + (frame_index[0]//12 * 10) + frame_index[0] % 12
                    frame_index_new.append(frame)

                    line_0 = drivingline [12 * i + j] [0] + 0.4 * (drivingline [12 * i + j + 1] [0] - drivingline [12 * i + j] [0])
                    line_1 = drivingline [12 * i + j] [1] + 0.4 * (drivingline [12 * i + j + 1] [1] - drivingline [12 * i + j] [1])
                    line = [line_0, line_1]
                    drivingline_new.append (line)

                    l_id = lane_id [12 * i + j]
                    lane_id_new.append (l_id)

                    pixel_x = pixel_cpos_x [12 * i + j] + 0.4 * (pixel_cpos_x [12 * i + j + 1] - pixel_cpos_x [12 * i + j])
                    pixel_y = pixel_cpos_y [12 * i + j] + 0.4 * (pixel_cpos_y [12 * i + j + 1] - pixel_cpos_y [12 * i + j])
                    geo_x = geo_cpos_x [12 * i + j] + 0.4 * (geo_cpos_x [12 * i + j + 1] - geo_cpos_x [12 * i + j])
                    geo_y = geo_cpos_y [12 * i + j] + 0.4 * (geo_cpos_y [12 * i + j + 1] - geo_cpos_y [12 * i + j])

                    pixel_cpos_x_new.append (pixel_x)
                    pixel_cpos_y_new.append (pixel_y)
                    geo_cpos_x_new.append (geo_x)
                    geo_cpos_y_new.append (geo_y)

                elif j == 2:
                    frame = 10 * i + j+ (frame_index[0]//12 * 10) + frame_index[0] % 12
                    frame_index_new.append(frame)

                    line_0 = drivingline [12 * i + j] [0] + 0.6 * (drivingline [12 * i + j + 1] [0] - drivingline [12 * i + j] [0])
                    line_1 = drivingline [12 * i + j] [1] + 0.6 * (drivingline [12 * i + j + 1] [1] - drivingline [12 * i + j] [1])
                    line = [line_0, line_1]
                    drivingline_new.append (line)

                    l_id = lane_id [12 * i + j + 1]
                    lane_id_new.append (l_id)

                    pixel_x = pixel_cpos_x [12 * i + j] + 0.6 * (pixel_cpos_x [12 * i + j + 1] - pixel_cpos_x [12 * i + j])
                    pixel_y = pixel_cpos_y [12 * i + j] + 0.6 * (pixel_cpos_y [12 * i + j + 1] - pixel_cpos_y [12 * i + j])
                    geo_x = geo_cpos_x [12 * i + j] + 0.6 * (geo_cpos_x [12 * i + j + 1] - geo_cpos_x [12 * i + j])
                    geo_y = geo_cpos_y [12 * i + j] + 0.6 * (geo_cpos_y [12 * i + j + 1] - geo_cpos_y [12 * i + j])

                    pixel_cpos_x_new.append (pixel_x)
                    pixel_cpos_y_new.append (pixel_y)
                    geo_cpos_x_new.append (geo_x)
                    geo_cpos_y_new.append (geo_y)

                elif j == 3:
                    frame = 10 * i + j+ (frame_index[0]//12 * 10) + frame_index[0] % 12
                    frame_index_new.append(frame)

                    line_0 = drivingline [12 * i + j] [0] + 0.8 * (drivingline [12 * i + j + 1] [0] - drivingline [12 * i + j] [0])
                    line_1 = drivingline [12 * i + j] [1] + 0.8 * (drivingline [12 * i + j + 1] [1] - drivingline [12 * i + j] [1])
                    line = [line_0, line_1]
                    drivingline_new.append (line)

                    l_id = lane_id [12 * i + j + 1]
                    lane_id_new.append (l_id)

                    pixel_x = pixel_cpos_x [12 * i + j] + 0.8 * (pixel_cpos_x [12 * i + j + 1] - pixel_cpos_x [12 * i + j])
                    pixel_y = pixel_cpos_y [12 * i + j] + 0.8 * (pixel_cpos_y [12 * i + j + 1] - pixel_cpos_y [12 * i + j])
                    geo_x = geo_cpos_x [12 * i + j] + 0.8 * (geo_cpos_x [12 * i + j + 1] - geo_cpos_x [12 * i + j])
                    geo_y = geo_cpos_y [12 * i + j] + 0.8 * (geo_cpos_y [12 * i + j + 1] - geo_cpos_y [12 * i + j])

                    pixel_cpos_x_new.append (pixel_x)
                    pixel_cpos_y_new.append (pixel_y)
                    geo_cpos_x_new.append (geo_x)
                    geo_cpos_y_new.append (geo_y)

                elif j == 4:
                    frame = 10 * i + j+ (frame_index[0]//12 * 10) + frame_index[0] % 12
                    frame_index_new.append(frame)

                    line_0 = drivingline [12 * i + j + 1] [0]
                    line_1 = drivingline [12 * i + j + 1] [1]
                    line = [line_0, line_1]
                    drivingline_new.append (line)

                    l_id = lane_id [12 * i + j + 1]
                    lane_id_new.append (l_id)

                    pixel_x = pixel_cpos_x [12 * i + j + 1]
                    pixel_y = pixel_cpos_y [12 * i + j + 1]
                    geo_x = geo_cpos_x [12 * i + j + 1]
                    geo_y = geo_cpos_y [12 * i + j + 1]

                    pixel_cpos_x_new.append (pixel_x)
                    pixel_cpos_y_new.append (pixel_y)
                    geo_cpos_x_new.append (geo_x)
                    geo_cpos_y_new.append (geo_y)

                elif j == 5:
                    frame = 10 * i + j+ (frame_index[0]//12 * 10) + frame_index[0] % 12
                    frame_index_new.append(frame)

                    line_0 = drivingline [12 * i + j + 1] [0] + 0.2 * (drivingline [12 * i + j + 2] [0] - drivingline [12 * i + j + 1] [0])
                    line_1 = drivingline [12 * i + j + 1] [1] + 0.2 * (drivingline [12 * i + j + 2] [1] - drivingline [12 * i + j + 1] [1])
                    line = [line_0, line_1]
                    drivingline_new.append (line)

                    l_id = lane_id [12 * i + j + 1]
                    lane_id_new.append (l_id)

                    pixel_x = pixel_cpos_x [12 * i + j + 1] + 0.2 * (pixel_cpos_x [12 * i + j + 2] - pixel_cpos_x [12 * i + j + 1])
                    pixel_y = pixel_cpos_y [12 * i + j + 1] + 0.2 * (pixel_cpos_y [12 * i + j + 2] - pixel_cpos_y [12 * i + j + 1])
                    geo_x = geo_cpos_x [12 * i + j + 1] + 0.2 * (geo_cpos_x [12 * i + j + 2] - geo_cpos_x [12 * i + j + 1])
                    geo_y = geo_cpos_y [12 * i + j + 1] + 0.2 * (geo_cpos_y [12 * i + j + 2] - geo_cpos_y [12 * i + j + 1])

                    pixel_cpos_x_new.append (pixel_x)
                    pixel_cpos_y_new.append (pixel_y)
                    geo_cpos_x_new.append (geo_x)
                    geo_cpos_y_new.append (geo_y)

                elif j == 6:
                    frame = 10 * i + j+ (frame_index[0]//12 * 10) + frame_index[0] % 12
                    frame_index_new.append(frame)

                    line_0 = drivingline [12 * i + j + 1] [0] + 0.4 * (drivingline [12 * i + j + 2] [0] - drivingline [12 * i + j + 1] [0])
                    line_1 = drivingline [12 * i + j + 1] [1] + 0.4 * (drivingline [12 * i + j + 2] [1] - drivingline [12 * i + j + 1] [1])
                    line = [line_0, line_1]
                    drivingline_new.append (line)

                    l_id = lane_id [12 * i + j + 1]
                    lane_id_new.append (l_id)

                    pixel_x = pixel_cpos_x [12 * i + j + 1] + 0.4 * (pixel_cpos_x [12 * i + j + 2] - pixel_cpos_x [12 * i + j + 1])
                    pixel_y = pixel_cpos_y [12 * i + j + 1] + 0.4 * (pixel_cpos_y [12 * i + j + 2] - pixel_cpos_y [12 * i + j + 1])
                    geo_x = geo_cpos_x [12 * i + j + 1] + 0.4 * (geo_cpos_x [12 * i + j + 2] - geo_cpos_x [12 * i + j + 1])
                    geo_y = geo_cpos_y [12 * i + j + 1] + 0.4 * (geo_cpos_y [12 * i + j + 2] - geo_cpos_y [12 * i + j + 1])

                    pixel_cpos_x_new.append (pixel_x)
                    pixel_cpos_y_new.append (pixel_y)
                    geo_cpos_x_new.append (geo_x)
                    geo_cpos_y_new.append (geo_y)

                elif j == 7:
                    frame = 10 * i + j+ (frame_index[0]//12 * 10) + frame_index[0] % 12
                    frame_index_new.append(frame)

                    line_0 = drivingline [12 * i + j + 1] [0] + 0.6 * (drivingline [12 * i + j + 2] [0] - drivingline [12 * i + j + 1] [0])
                    line_1 = drivingline [12 * i + j + 1] [1] + 0.6 * (drivingline [12 * i + j + 2] [1] - drivingline [12 * i + j + 1] [1])
                    line = [line_0, line_1]
                    drivingline_new.append (line)

                    l_id = lane_id [12 * i + j + 2]
                    lane_id_new.append (l_id)

                    pixel_x = pixel_cpos_x [12 * i + j + 1] + 0.6 * (pixel_cpos_x [12 * i + j + 2] - pixel_cpos_x [12 * i + j + 1])
                    pixel_y = pixel_cpos_y [12 * i + j + 1] + 0.6 * (pixel_cpos_y [12 * i + j + 2] - pixel_cpos_y [12 * i + j + 1])
                    geo_x = geo_cpos_x [12 * i + j + 1] + 0.6 * (geo_cpos_x [12 * i + j + 2] - geo_cpos_x [12 * i + j + 1])
                    geo_y = geo_cpos_y [12 * i + j + 1] + 0.6 * (geo_cpos_y [12 * i + j + 2] - geo_cpos_y [12 * i + j + 1])

                    pixel_cpos_x_new.append (pixel_x)
                    pixel_cpos_y_new.append (pixel_y)
                    geo_cpos_x_new.append (geo_x)
                    geo_cpos_y_new.append (geo_y)

                elif j == 8:
                    frame = 10 * i + j+ (frame_index[0]//12 * 10) + frame_index[0] % 12
                    frame_index_new.append(frame)

                    line_0 = drivingline [12 * i + j + 1] [0] + 0.8 * (drivingline [12 * i + j + 2] [0] - drivingline [12 * i + j + 1] [0])
                    line_1 = drivingline [12 * i + j + 1] [1] + 0.8 * (drivingline [12 * i + j + 2] [1] - drivingline [12 * i + j + 1] [1])
                    line = [line_0, line_1]
                    drivingline_new.append (line)

                    l_id = lane_id [12 * i + j + 2]
                    lane_id_new.append (l_id)

                    pixel_x = pixel_cpos_x [12 * i + j + 1] + 0.8 * (pixel_cpos_x [12 * i + j + 2] - pixel_cpos_x [12 * i + j + 1])
                    pixel_y = pixel_cpos_y [12 * i + j + 1] + 0.8 * (pixel_cpos_y [12 * i + j + 2] - pixel_cpos_y [12 * i + j + 1])
                    geo_x = geo_cpos_x [12 * i + j + 1] + 0.8 * (geo_cpos_x [12 * i + j + 2] - geo_cpos_x [12 * i + j + 1])
                    geo_y = geo_cpos_y [12 * i + j + 1] + 0.8 * (geo_cpos_y [12 * i + j + 2] - geo_cpos_y [12 * i + j + 1])

                    pixel_cpos_x_new.append (pixel_x)
                    pixel_cpos_y_new.append (pixel_y)
                    geo_cpos_x_new.append (geo_x)
                    geo_cpos_y_new.append (geo_y)

                elif j == 9:
                    frame = 10 * i + j+ (frame_index[0]//12 * 10) + frame_index[0] % 12
                    frame_index_new.append(frame)

                    line_0 = drivingline [12 * i + j + 2] [0]
                    line_1 = drivingline [12 * i + j + 2] [1]
                    line = [line_0, line_1]
                    drivingline_new.append (line)

                    l_id = lane_id [12 * i + j + 2]
                    lane_id_new.append (l_id)

                    pixel_x = pixel_cpos_x [12 * i + j + 2]
                    pixel_y = pixel_cpos_y [12 * i + j + 2]
                    geo_x = geo_cpos_x [12 * i + j + 2]
                    geo_y = geo_cpos_y [12 * i + j + 2]

                    pixel_cpos_x_new.append (pixel_x)
                    pixel_cpos_y_new.append (pixel_y)
                    geo_cpos_x_new.append (geo_x)
                    geo_cpos_y_new.append (geo_y)
        #print(frame_index_new)
        #print(drivingline_new)
        #print (pixel_cpos_x_new)
        print(type(drivingline_new[0]))
        new_veh_data = {'frame_index': frame_index_new, 'pixel_cpos_x': pixel_cpos_x_new, 'pixel_cpos_y': pixel_cpos_y_new,
                        'drivingline': {'mainroad': drivingline_new}, 'lane_id': lane_id_new,
                        'geo_cpos_x': geo_cpos_x_new, 'geo_cpos_y': geo_cpos_y_new,
                        'start_unix_time': start_unix_time_new, 'detaT': detaT, 'vehicle_length': vehicle_length_new}

        new_vehicle_data[veh_id] = new_veh_data
    #for veh_id, veh_data in new_vehicle_data.items ():

        #print(veh_data['drivingline'])
    return new_vehicle_data

if __name__ == '__main__':
    pkl_file = '/data3/liyitong/HuRong_process/A2/20220617_0850_A2_F5_370_1_Num_4/tp_result_20220617_0850_A2_F5_370_1_old.tppkl'
    new_vehicle_data = run_main(pkl_file)
    file_folder = '/data3/liyitong/HuRong_process/A2/20220617_0850_A2_F5_370_1_Num_4'
    save_file_name = os.path.join (file_folder, 'tp_result_20220617_0850_A2_F5_370_1.tppkl' )
    save_data2pkl (new_vehicle_data, save_file_name)
    # nums = [1,2,3,5,6,7,9,10,11,12,13,14,18]
