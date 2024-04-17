#!/usr/bin/env python 
# -*- coding: utf-8 -*-
'''
@Time : 2022-10-01 16:43
@Author : Xinkai Ji
@contact: jhjxk@hotmail.com
@File : one_video_trajectory_connection.py 
@Software: PyCharm
@desc: 同个视频中轨迹视频的中断，用于跨线桥的遮挡
'''
import copy

import numpy as np
from scipy import interpolate

def connect_drivingline(vehicle_data, region):
    # region = [480, 530, 500, 515, 'linear']
    speed_time = 2  # seconds 用于计算速度的历史时间区间
    max_expand_time = 30  # seconds 最大的延伸时长
    start, end, start_mid, end_mid, method = region
    main_select_veh = {}
    sub_select_veh = {}
    main_vehicle_id_time_order = []
    sub_vehicle_id_time_order = []
    main_start_index_ls = []
    main_end_index_ls = []
    for veh_id, veh_data in vehicle_data.items():
        line_name = list(veh_data['drivingline'].keys())[0]
        frame_index = veh_data['frame_index']
        main_start_index_ls.append(frame_index[0])
        main_end_index_ls.append(frame_index[-1])
        pixel_cpos_x = veh_data['pixel_cpos_x']
        pixel_cpos_y = veh_data['pixel_cpos_y']
        drivingline = [x[0] for x in veh_data['drivingline'][line_name]]
        drivingline_dist = [x[1] for x in veh_data['drivingline'][line_name]]
        lane_id = veh_data['lane_id']
        detaT = veh_data['detaT']
        gap_frame = int(speed_time / detaT)
        if len(frame_index) <= gap_frame+3:
            continue
        max_expand_frame = int(max_expand_time / detaT)
        if drivingline[0] <= start <= drivingline[-1] <= end:  # 递增 下部
            # t增大方向延伸
            expand_frame_index, expand_pixel_cpos_x, expand_pixel_cpos_y, expand_drivingline, expand_drivingline_dist, expand_lane_id = expand_trajectory_linear(
                frame_index,
                pixel_cpos_x, pixel_cpos_y, drivingline, drivingline_dist, lane_id, end_mid, gap_frame,
                max_expand_frame, positive_direction=True)
            if len(expand_frame_index)>0:
                select_position = [[expand_pixel_cpos_x[i],expand_pixel_cpos_y[i],expand_drivingline[i],expand_lane_id[i]] for i in range(len(expand_frame_index))]
                main_select_veh[veh_id] = [expand_frame_index,select_position]
                main_vehicle_id_time_order.append([veh_id, expand_frame_index[0]])
        elif start <= drivingline[0] <= end <= drivingline[-1]:  # 递增 上部
            # t减少方向延伸
            expand_frame_index, expand_pixel_cpos_x, expand_pixel_cpos_y, expand_drivingline, expand_drivingline_dist, expand_lane_id = expand_trajectory_linear(
                frame_index,
                pixel_cpos_x, pixel_cpos_y, drivingline, drivingline_dist, lane_id, start_mid, gap_frame,
                max_expand_frame, positive_direction=False)
            if len(expand_frame_index) > 0:
                select_position = [[expand_pixel_cpos_x[i],expand_pixel_cpos_y[i],expand_drivingline[i],expand_lane_id[i]] for i in range(len(expand_frame_index))]
                sub_select_veh[veh_id] = [expand_frame_index, select_position]
                sub_vehicle_id_time_order.append([veh_id, expand_frame_index[0]])
        elif start <= drivingline[-1] <= end <= drivingline[0]:  # 递减 上部
            # t增大方向延伸
            expand_frame_index, expand_pixel_cpos_x, expand_pixel_cpos_y, expand_drivingline, expand_drivingline_dist, expand_lane_id = expand_trajectory_linear(
                frame_index, pixel_cpos_x, pixel_cpos_y, drivingline, drivingline_dist, lane_id, start_mid, gap_frame,
                max_expand_frame, positive_direction=True)
            if len(expand_frame_index)>0:
                select_position = [[expand_pixel_cpos_x[i],expand_pixel_cpos_y[i],expand_drivingline[i],expand_lane_id[i]] for i in range(len(expand_frame_index))]
                sub_select_veh[veh_id] = [expand_frame_index, select_position]
                sub_vehicle_id_time_order.append([veh_id, expand_frame_index[0]])
        elif drivingline[-1] <= start <= drivingline[0] <= end:
            # t减少方向延伸
            expand_frame_index, expand_pixel_cpos_x, expand_pixel_cpos_y, expand_drivingline, expand_drivingline_dist, expand_lane_id = expand_trajectory_linear(
                frame_index, pixel_cpos_x, pixel_cpos_y, drivingline, drivingline_dist, lane_id, end_mid, gap_frame,
                max_expand_frame, positive_direction=False)
            if len(expand_frame_index)>0:
                select_position = [[expand_pixel_cpos_x[i],expand_pixel_cpos_y[i],expand_drivingline[i],expand_lane_id[i]] for i in range(len(expand_frame_index))]
                main_select_veh[veh_id] = [expand_frame_index, select_position]
                main_vehicle_id_time_order.append([veh_id, expand_frame_index[0]])
    main_start_index = min(main_start_index_ls)
    main_end_index = max(main_end_index_ls)
    matched_main = match_vehicle(main_select_veh,main_vehicle_id_time_order,sub_select_veh, sub_vehicle_id_time_order,
                  main_start_index,main_end_index,vehicle_data,similarity_func=_cal_similarity)
    print(matched_main)
    connect_vehicle_trajectory(matched_main, vehicle_data)
        # veh_data['frame_index'] = frame_index
        # veh_data['pixel_cpos_x'] = pixel_cpos_x
        # veh_data['pixel_cpos_y'] = pixel_cpos_y
        # veh_data['drivingline'][line_name] = [[x, y] for x, y in zip(drivingline, drivingline_dist)]
        # veh_data['lane_id'] = lane_id

def _cal_similarity(main_data,sub_data):
    '''
    计算相似度
    :param main_data: [select_frame_index,select_position]
    :param sub_data:
    :return:
    '''
    sim = np.inf
    main_frame,main_position = main_data
    sub_frame, sub_position = sub_data
    s_frame = max(main_frame[0],sub_frame[0])
    e_frame = min(main_frame[-1], sub_frame[-1])
    if main_frame[0] < 0 or sub_frame[0] < 0:
        return sim
    if e_frame - s_frame < 5:
        return sim
    s_p_index = main_frame.index(s_frame)
    e_p_index = main_frame.index(e_frame)
    main_select_position = main_position[s_p_index:e_p_index+1]
    s_p_index = sub_frame.index(s_frame)
    e_p_index = sub_frame.index(e_frame)
    sub_select_position = sub_position[s_p_index:e_p_index + 1]
    p_m = np.array(main_select_position)
    p_s = np.array(sub_select_position)
    x_m = p_m[:,0]
    x_s = p_s[:,0]
    driving_m = p_m[:,2]
    driving_s = p_s[:, 2]
    lane_m = p_m[:,3]
    lane_s = p_s[:, 3]
    direction = np.dot((driving_m[-1] - driving_m[0]), (driving_s[-1] - driving_s[0]))
    if direction < 0:  # 不同个方向
        return sim
    dist_x = np.mean(np.sqrt(np.sum((x_m - x_s)**2,axis=0)))/10
    dist_driving = np.mean(np.sqrt(np.sum((driving_m - driving_s)**2,axis=0)))
    dist_lane = np.mean(np.sqrt(np.sum((lane_m - lane_s) ** 2, axis=0)))*30
    dist = dist_x+dist_driving+dist_lane
    if dist>2000:
        dist = np.inf
    return dist

def expand_trajectory_linear(frame_index, pixel_cpos_x, pixel_cpos_y, drivingline, drivingline_dist, lane_id,
                             final_dist,
                             gap_frame,
                             max_expand_frame, positive_direction):
    expand_frame_index = []
    expand_pixel_cpos_x = []
    expand_pixel_cpos_y = []
    expand_drivingline = []
    expand_drivingline_dist = []
    expand_lane_id = []
    speed_skip_frame = 3
    if positive_direction:  # type 1 3
        x_speed = (pixel_cpos_x[-1-speed_skip_frame] - pixel_cpos_x[-1 - gap_frame-speed_skip_frame]) / gap_frame
        y_speed = (pixel_cpos_y[-1-speed_skip_frame] - pixel_cpos_y[-1 - gap_frame-speed_skip_frame]) / gap_frame
        drivingline_speed = (drivingline[-1-speed_skip_frame] - drivingline[-1 - gap_frame-speed_skip_frame]) / gap_frame
        drivingline_dist_speed = (drivingline_dist[-1-speed_skip_frame] - drivingline_dist[-1 - gap_frame-speed_skip_frame]) / gap_frame
        if drivingline_speed == 0:
            drivingline_speed += 0.000001
        add_frame_num = int(min(abs((final_dist - drivingline[-1]) / drivingline_speed), max_expand_frame))
        for i in range(add_frame_num):
            # frame_index.append(frame_index[-1] + 1)
            # pixel_cpos_x.append(pixel_cpos_x[-1] + x_speed)
            # pixel_cpos_y.append(pixel_cpos_y[-1] + y_speed)
            # drivingline.append(drivingline[-1] + drivingline_speed)
            # drivingline_dist.append(drivingline_dist[-1] + drivingline_dist_speed)
            # lane_id.append(lane_id[-1])
            if i == 0:
                expand_frame_index.append(frame_index[-1] + 1)
                expand_pixel_cpos_x.append(pixel_cpos_x[-1] + x_speed)
                expand_pixel_cpos_y.append(pixel_cpos_y[-1] + y_speed)
                expand_drivingline.append(drivingline[-1] + drivingline_speed)
                expand_drivingline_dist.append(drivingline_dist[-1] + drivingline_dist_speed)
                expand_lane_id.append(lane_id[-1])
            else:
                expand_frame_index.append(expand_frame_index[-1] + 1)
                expand_pixel_cpos_x.append(expand_pixel_cpos_x[-1] + x_speed)
                expand_pixel_cpos_y.append(expand_pixel_cpos_y[-1] + y_speed)
                expand_drivingline.append(expand_drivingline[-1] + drivingline_speed)
                expand_drivingline_dist.append(expand_drivingline_dist[-1] + drivingline_dist_speed)
                expand_lane_id.append(expand_lane_id[-1])
    else:  # type 2 4
        x_speed = (pixel_cpos_x[speed_skip_frame] - pixel_cpos_x[gap_frame+speed_skip_frame]) / gap_frame
        y_speed = (pixel_cpos_y[speed_skip_frame] - pixel_cpos_y[gap_frame+speed_skip_frame]) / gap_frame
        drivingline_speed = (drivingline[speed_skip_frame] - drivingline[gap_frame+speed_skip_frame]) / gap_frame
        drivingline_dist_speed = (drivingline_dist[speed_skip_frame] - drivingline_dist[gap_frame+speed_skip_frame]) / gap_frame
        if drivingline_speed == 0:
            drivingline_speed += 0.000001
        add_frame_num = int(min(abs((final_dist - drivingline[0]) / drivingline_speed), max_expand_frame))
        for i in range(add_frame_num):
            if i == 0:
                expand_frame_index.insert(0, frame_index[0] - 1)
                expand_pixel_cpos_x.insert(0, pixel_cpos_x[0] + x_speed)
                expand_pixel_cpos_y.insert(0, pixel_cpos_y[0] + y_speed)
                expand_drivingline.insert(0, drivingline[0] + drivingline_speed)
                expand_drivingline_dist.insert(0, drivingline_dist[0] + drivingline_dist_speed)
                expand_lane_id.insert(0, lane_id[0])
            else:
                expand_frame_index.insert(0, expand_frame_index[0] - 1)
                expand_pixel_cpos_x.insert(0, expand_pixel_cpos_x[0] + x_speed)
                expand_pixel_cpos_y.insert(0, expand_pixel_cpos_y[0] + y_speed)
                expand_drivingline.insert(0, expand_drivingline[0] + drivingline_speed)
                expand_drivingline_dist.insert(0, expand_drivingline_dist[0] + drivingline_dist_speed)
                expand_lane_id.insert(0, lane_id[0])
    # if len(expand_frame_index) == 0:
    #     print('ss')
    # return frame_index, pixel_cpos_x, pixel_cpos_y, drivingline, drivingline_dist,lane_id
    return expand_frame_index, expand_pixel_cpos_x, expand_pixel_cpos_y, expand_drivingline, expand_drivingline_dist, expand_lane_id


def connect_vehicle_trajectory(matched_main,vehicle_data):
    threshold = 1000
    matched_veh_num = 0
    for main_veh_id,item in matched_main.items():
        sub_veh_id,dist = item
        if dist > threshold:
            continue
        main_data = vehicle_data[main_veh_id]
        sub_data = vehicle_data[sub_veh_id]
        line_name = list(main_data['drivingline'].keys())[0]
        m_drivingline = [x[0] for x in main_data['drivingline'][line_name]]
        m_drivingline_dist = [x[1] for x in main_data['drivingline'][line_name]]
        s_drivingline = [x[0] for x in sub_data['drivingline'][line_name]]
        s_drivingline_dist = [x[1] for x in sub_data['drivingline'][line_name]]
        if main_data['frame_index'][0] >sub_data['frame_index'][-1]:
            main_data,sub_data=sub_data,main_data
        if max(main_data['frame_index'][0],sub_data['frame_index'][0]) <min(main_data['frame_index'][-1],sub_data['frame_index'][-1]):
            print('con not connect:m:%f,s:%f'%(main_veh_id,sub_veh_id))
            print('main:',main_data['frame_index'][0],',',main_data['frame_index'][-1])
            print('sub:', sub_data['frame_index'][0],',', sub_data['frame_index'][-1])
            continue
        main_data['frame_index'] = [round(x) for x in main_data['frame_index']]
        sub_data['frame_index'] = [round(x) for x in sub_data['frame_index']]
        frame_index = main_data['frame_index'] + sub_data['frame_index']
        new_frame_index = list(range(frame_index[0], frame_index[-1] + 1, 1))
        pos_x = main_data['pixel_cpos_x'] + sub_data['pixel_cpos_x']
        pos_y = main_data['pixel_cpos_y'] + sub_data['pixel_cpos_y']
        drivingline = m_drivingline + s_drivingline
        drivingline_dist = m_drivingline_dist + s_drivingline_dist
        lane_id = main_data['lane_id'] + sub_data['lane_id']
        vehicle_length = main_data['vehicle_length']
        start_unix_time = main_data['start_unix_time']
        detaT = main_data['detaT']
        #lane_dist = []
        #sub_lane_dist = copy.copy(sub_data['lane_dist'][-1])
        #main_lane_dist = copy.copy(main_data['lane_dist'][0])
        # for i in range(sub_data['frame_index'][0] - main_data['frame_index'][-1] - 1):
        #     if i < (sub_data['frame_index'][0] - main_data['frame_index'][-1]) / 2:
        #         lane_dist.append(main_lane_dist)
        #     else:
        #         lane_dist.append(sub_lane_dist)
        # new_lane_dist = main_data['lane_dist'] + lane_dist + sub_data['lane_dist']
        y_ls = [pos_x,pos_y,drivingline,drivingline_dist]
        new_pos_x,new_pos_y,new_drivingline,new_drivingline_dist = get_inter_record(frame_index, y_ls)
        new_lane_id = [round(x) for x in get_inter_record(frame_index,[lane_id],'nearest')[0]]
        new_drivingline_line = [[x,y] for x,y in zip(new_drivingline,new_drivingline_dist)]
        new_veh_data = {'frame_index': new_frame_index, 'pixel_cpos_x': new_pos_x, 'pixel_cpos_y': new_pos_y,
                        'drivingline': {line_name: new_drivingline_line}, 'lane_id': new_lane_id,
                        'start_unix_time': start_unix_time, 'detaT': detaT, 'vehicle_length': vehicle_length}
        vehicle_data[main_veh_id] = new_veh_data
        vehicle_data.pop(sub_veh_id)
        matched_veh_num += 1
    print('bridge matched:%d'%matched_veh_num)


def get_inter_record(frame_index_ls, y_ls, kind='slinear'):
    '''
    对位置和速度进行插值 默认采用线性插值
    :param frame_index_ls:
    :param y_ls:
    :param kind:
    :return:
    '''
    new_y_ls = []
    new_frame_index = list(range(frame_index_ls[0], frame_index_ls[-1] + 1, 1))
    for y in y_ls:
        f = interpolate.interp1d(frame_index_ls, y, kind=kind)
        new_y = f(new_frame_index).tolist()
        new_y_ls.append(new_y)
    return new_y_ls

def match_vehicle(selected_main_vehicle,main_vehicle_id_time_order,selected_sub_vehicle, sub_vehicle_id_time_order,
                  main_start_index,main_end_index,vehicle_data,similarity_func = None ):
    if similarity_func is None:
        similarity_func = _cal_similarity
    main_veh_id_ls = _get_horizon_data(main_vehicle_id_time_order,main_start_index,main_end_index)
    sub_veh_id_ls = _get_horizon_data(sub_vehicle_id_time_order,main_start_index,main_end_index)
    cos_matrix = []
    for main_veh_id in main_veh_id_ls:
        cos =[]
        for sub_veh_id in sub_veh_id_ls:
            main_data = selected_main_vehicle[main_veh_id]
            sub_data = selected_sub_vehicle[sub_veh_id]
            raw_main_data = vehicle_data[main_veh_id]
            raw_sub_data = vehicle_data[sub_veh_id]
            if max(raw_main_data['frame_index'][0],raw_sub_data['frame_index'][0]) < min(raw_main_data['frame_index'][-1],raw_sub_data['frame_index'][-1]):
                dist = np.inf
            else:
                dist = similarity_func(main_data, sub_data)
            cos.append(dist)
        cos_matrix.append(cos)
    cos_matrix = np.array(cos_matrix)
    h,w = cos_matrix.shape
    if h>w:
        temp = np.ones((h,h-w))*np.inf
        cos_matrix = np.hstack((cos_matrix,temp))
    elif h<w:
        temp = np.ones((w-h,w)) * np.inf
        cos_matrix = np.vstack((cos_matrix, temp))
    matched_indices = linear_assignment(cos_matrix)
    matched_main = {}
    for m in matched_indices:
        if m[0]<h and m[1]<w:
            m_id = main_veh_id_ls[m[0]]
            s_id = sub_veh_id_ls[m[1]]
            dist = cos_matrix[m[0],m[1]]
            matched_main[m_id] = [s_id,dist]

    return matched_main

def _get_horizon_data(vehicle_id_time_order,start_index,end_index):
    veh_id_ls = []
    for veh_id, start_frame in vehicle_id_time_order:
        if start_frame >=start_index and start_frame < end_index:
            veh_id_ls.append(veh_id)
        if start_frame > end_index:
            break
    return veh_id_ls

def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))
if __name__ == '__main__':
    import pickle

    tppkl_path = '/data3/liyitong/tp_result_first_frame_M-20220708_Y1_B_F1_1-S-20220708_Y2_B_F1_1.tppkl'
    new_tppkl_path = '/data3/liyitong/tp_result_first_frame_M-20220708_Y1_B_F1_1-S-20220708_Y2_B_F1_1_new1.tppkl'
    #region = [2900,3094,2914,3050,'linear']
    #region = [3064 , 3141 , 3084 , 3129 , 'linear']
    #region = [3164 , 3241 , 3184 , 3229 , 'linear']
    region = [390 , 450 , 400 , 440 , 'linear']
    #region = [1600 , 1900 , 1650 , 1850 , 'linear']
    with open(tppkl_path, 'rb') as f:
        vehicles_data = pickle.load(f)
    print('load success:%s' % tppkl_path)
    # #将对向车道翻转，再进行拼接
    # reverse_veh_id_ls = []
    # for veh_id,veh_data in vehicles_data.items():
    #     print(veh_data ['drivingline'].keys())
    #     frame_index = veh_data ['frame_index']
    #     drivingline = veh_data ['drivingline'] ['mainline']
    #
    #     lane_id = veh_data ['lane_id']
    #
    #     pixel_cpos_x = veh_data ['pixel_cpos_x']
    #     pixel_cpos_y = veh_data ['pixel_cpos_y']
    #
    #     if drivingline[0] > drivingline[-1]:
    #         for i in range(len(drivingline)):
    #             drivingline[i][0] = 650 - drivingline[i][0]
    #         drivingline_mainroad = {}
    #         drivingline_mainroad['mainline'] = drivingline
    #         reverse_veh_id_ls.append(veh_id)

    connect_drivingline(vehicles_data, region)
#翻转回来
    # for veh_id,veh_data in vehicles_data.items():
    #     if veh_id in reverse_veh_id_ls:
    #
    #
    #         frame_index = veh_data ['frame_index']
    #         drivingline = veh_data ['drivingline'] ['mainline']
    #
    #         lane_id = veh_data ['lane_id']
    #
    #         pixel_cpos_x = veh_data ['pixel_cpos_x']
    #         pixel_cpos_y = veh_data ['pixel_cpos_y']
    #
    #         for i in range(len(drivingline)):
    #             drivingline[i][0] = 650 - drivingline[i][0]
    #         drivingline_mainroad = {}
    #         drivingline_mainroad['mainline'] = drivingline
    #         reverse_veh_id_ls.append(veh_id)


    with open(new_tppkl_path, 'wb') as f:
        pickle.dump(vehicles_data, f)
    print('save success:%s' % new_tppkl_path)



