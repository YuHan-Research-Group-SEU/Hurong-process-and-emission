# -*- coding: utf-8 -*-
import pickle
import numpy as np
import pandas as pd
import os
from multiprocessing.pool import Pool
import cv2
from PIL import Image
from pyheatmap.heatmap import HeatMap
import matplotlib.pyplot as plt

plt.rcParams ['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams ['axes.unicode_minus'] = False  # 用来正常显示负号
import math
import seaborn as sns


def get_tppkl_data (tppkl_path):
    global start_frame_index
    global end_frame_index
    global all_lane
    global df_veh
    global save_folder
    global video_name
    global vehicles_data
    global detaT
    global start_unix_time
    global temp_driving_name

    with open (tppkl_path, 'rb') as f:
        vehicles_data = pickle.load (f)
    save_folder, tppkl_file_name = os.path.split (tppkl_path)
    video_name, _ = os.path.splitext (tppkl_file_name)
    video_name = video_name [10:]


get_tppkl_data (
    "/data3/liyitong/HuRong_process/B1/20220616_0845_B1_F5_370_1_Num_5/tp_result_20220616_0845_B1_F5_370_1.tppkl")

de_index = {}
for key, value in vehicles_data.items ():
    temp_driving_name = list (value ['drivingline'].keys ()) [0]
    de_index [key] = []
    for i in range (0, len (value ['drivingline'] [temp_driving_name]) - 20):
        if abs (value ['drivingline'] [temp_driving_name] [i + 10] [0] - value ['drivingline'] [temp_driving_name] [i] [
            0]) < 0.5 and value ['lane_id'] [i] < 15:
            de_index [key].append (i)
            print (key, value ['drivingline'] [temp_driving_name] [i] [0])

new_dict = {}
for k, v in de_index.items ():
    if v != []:
        new_dict [k] = {}
        a = v [-1]
        for key in ['frame_index', 'pixel_cpos_x', 'pixel_cpos_y', 'geo_cpos_x', 'geo_cpos_y', 'lane_id', 'lane_dist']:
            new_dict [k] [key] = vehicles_data [k] [key] [a:]
        for key in ['drivingline']:
            new_dict [k] [key] = {'mainroad': []}
            new_dict [k] [key] ['mainroad'] = vehicles_data [k] [key] ['mainroad'] [a:]
# print(new_dict)

for key, value in vehicles_data.items ():
    if key in new_dict.keys ():
        for key_1 in value.keys ():
            if key_1 in ['frame_index', 'pixel_cpos_x', 'pixel_cpos_y', 'geo_cpos_x', 'geo_cpos_y', 'lane_id',
                         'lane_dist', 'drivingline']:
                value [key_1] = new_dict [key] [key_1]

# 保存
with open ('/data3/liyitong/HuRong_process/B1/20220616_0845_B1_F5_370_1_Num_5/tp_result_20220616_0845_B1_F5_370_1.tppkl_new1', 'wb') as f:
    pickle.dump (vehicles_data, f)