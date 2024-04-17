import numpy as np
import copy
from scipy import interpolate


def main_short_loss(vehicle_data, config={}):
    '''
    短时丢失连接 主函数
    :return:
    '''
    time_extent = config.get('time_extent', 30)  # second  断的时间
    pixel_distance = config.get('pixel_distance', 50)  # second 像素坐标y值 50
    drivingline_distance = config.get('drivingline_distance', 30)  # second断的距离 10
    print("drivingline_distance", drivingline_distance)
    connect_region = config.get('connect_region', [230,400])
    veh_frame_ls = []
    for veh_id, veh_data in vehicle_data.items():
        frame_index = veh_data['frame_index']
        veh_frame_ls.append([veh_id, frame_index[0]])  # 保存每辆车的id和起始时间
    veh_frame_ls.sort(key=lambda x: x[1], reverse=False)  # 按照开始时间升序
    del_veh_ls = []
    for veh_id, veh_data in vehicle_data.items():
        boo = 0
        time_frame = int(time_extent / veh_data['detaT'])  # 1/0.1=10
        frame_index = veh_data['frame_index']
        line_name = list(veh_data['drivingline'].keys())[0]
        drivingline = veh_data['drivingline'][line_name]
        if connect_region[0] <= drivingline[-1][0] <= connect_region[1]:  # 结束的位置在495-510
            for other_veh_id, start_frame_index in veh_frame_ls:
                if frame_index[-1]  <= start_frame_index <= frame_index[
                    -1] +  time_frame:  # 其他车辆开始的时间 在【这辆车结束的时间，+10】
                    temp_driving_name = list(vehicle_data[other_veh_id]['drivingline'].keys())[0]

                    if veh_data['lane_id'][-1] == int(vehicle_data[other_veh_id]['lane_id'][0]) and veh_data['lane_id'][
                        -1] < 10 and connect_region[0]+10 < vehicle_data[other_veh_id]['drivingline'][temp_driving_name][0][0] < connect_region[1]+10:
                        boo = 1
                        other_veh = vehicle_data[other_veh_id]  # 符合条件的其他车辆信息other_veh
                        gap_frame = start_frame_index - frame_index[-1]  # 空缺的时间
                        matched_res = matched(veh_data, other_veh, line_name, gap_frame, pixel_distance,
                                              drivingline_distance)
                        if matched_res == True and other_veh_id not in del_veh_ls:
                            print(1)
                            vehicle_data[veh_id] = connect_vehicle_trajectory(veh_data, other_veh)
                            del_veh_ls.append(other_veh_id)
                    if abs(veh_data['lane_id'][-1] - vehicle_data[other_veh_id]['lane_id'][0]) < 1 and \
                            veh_data['lane_id'][-1] >= 20 and connect_region[0] < \
                            vehicle_data[other_veh_id]['drivingline'][temp_driving_name][0][0] < connect_region[1]:
                        #print(veh_id, other_veh_id, frame_index[-1], start_frame_index)
                        other_veh = vehicle_data[other_veh_id]  # 符合条件的其他车辆信息other_veh
                        gap_frame = start_frame_index - frame_index[-1]  # 空缺的时间
                        # print("gap_frame",gap_frame)
                        matched_res = matched(veh_data, other_veh, line_name, gap_frame, pixel_distance,
                                              drivingline_distance)
                        if matched_res == True and other_veh_id not in del_veh_ls:
                            boo = 1
                            #print(veh_id, other_veh_id)
                            vehicle_data[veh_id] = connect_vehicle_trajectory(veh_data, other_veh)
                            del_veh_ls.append(other_veh_id)

                if boo == 1:
                    break
                elif start_frame_index > frame_index[-1] + time_frame:
                    break
    for veh_id in del_veh_ls:
        # if veh_id in vehicle_data.keys():
        vehicle_data.pop(veh_id)


def matched(main_veh, sub_veh, line_name, gap_frame, pixel_distance, drivingline_distance):
    speed_skip_frame = 3
    speed_gap_frame = int(3 / main_veh['detaT'])
    pixel_cpos_x = main_veh['pixel_cpos_x']
    pixel_cpos_y = main_veh['pixel_cpos_y']
    drivingline = [x[0] for x in main_veh['drivingline'][line_name]]
    drivingline_dist = [x[1] for x in main_veh['drivingline'][line_name]]
    lane_id = main_veh['lane_id']
    if abs(-1 - speed_gap_frame - speed_skip_frame) < len(pixel_cpos_x):
        x_speed = (pixel_cpos_x[-1 - speed_skip_frame] - pixel_cpos_x[
            -1 - speed_gap_frame - speed_skip_frame]) / speed_gap_frame
        y_speed = (pixel_cpos_y[-1 - speed_skip_frame] - pixel_cpos_y[
            -1 - speed_gap_frame - speed_skip_frame]) / speed_gap_frame
        drivingline_speed = (drivingline[-1 - speed_skip_frame] - drivingline[
            -1 - speed_gap_frame - speed_skip_frame]) / speed_gap_frame
        drivingline_dist_speed = (drivingline_dist[-1 - speed_skip_frame] - drivingline_dist[
            -1 - speed_gap_frame - speed_skip_frame]) / speed_gap_frame

        expand_pixel_cpos_x = pixel_cpos_x[-1] + x_speed * gap_frame
        expand_pixel_cpos_y = pixel_cpos_y[-1] + y_speed * gap_frame
        expand_drivingline = drivingline[-1] + drivingline_speed * gap_frame
        expand_drivingline_dist = drivingline_dist[-1] + drivingline_dist_speed * gap_frame
        expand_lane_id = lane_id[-1]
        sub_lane_id = sub_veh['lane_id'][0]
        sub_pixel_cpos_x = sub_veh['pixel_cpos_x'][0]
        sub_pixel_cpos_y = sub_veh['pixel_cpos_y'][0]
        sub_drivingline = sub_veh['drivingline'][line_name][0][0]
        sub_drivingline_dist = sub_veh['drivingline'][line_name][0][1]
        gap_pixel_distance = np.sqrt(
            (expand_pixel_cpos_x - sub_pixel_cpos_x) ** 2 + (expand_pixel_cpos_y - sub_pixel_cpos_y) ** 2)
        gap_drivingline_distance = abs(expand_drivingline - sub_drivingline)
        gap_drivingline_dist_distance = abs(expand_drivingline_dist - sub_drivingline_dist)
    else:
        print("yes1")
        return False
    # if expand_lane_id==sub_lane_id and gap_pixel_distance<pixel_distance and gap_drivingline_distance < drivingline_distance:
    if abs(expand_lane_id - sub_lane_id) < 1:
        return True
    else:
        print("yes2", expand_lane_id, sub_lane_id, gap_drivingline_distance, drivingline_distance)
        return False


def connect_vehicle_trajectory(main_data, sub_data):
    line_name = list(main_data['drivingline'].keys())[0]
    m_drivingline = [x[0] for x in main_data['drivingline'][line_name]]
    m_drivingline_dist = [x[1] for x in main_data['drivingline'][line_name]]
    s_drivingline = [x[0] for x in sub_data['drivingline'][line_name]]
    s_drivingline_dist = [x[1] for x in sub_data['drivingline'][line_name]]
    frame_index = np.append(main_data['frame_index'], sub_data['frame_index'])
    # frame_index = main_data['frame_index'] + sub_data['frame_index']
    # new_frame_index = list(range(int(frame_index[0]), int(frame_index[-1]) + 1, 1))
    new_frame_index = list(range(int(main_data['frame_index'][0]), int(sub_data['frame_index'][-1]) + 1, 1))
    pos_x = main_data['pixel_cpos_x'] + sub_data['pixel_cpos_x']
    pos_y = main_data['pixel_cpos_y'] + sub_data['pixel_cpos_y']
    drivingline = m_drivingline + s_drivingline
    drivingline_dist = m_drivingline_dist + s_drivingline_dist
    lane_id = main_data['lane_id'] + sub_data['lane_id']
    vehicle_length = main_data['vehicle_length']
    start_unix_time = main_data['start_unix_time']
    detaT = main_data['detaT']
    #lane_dist = []
    # sub_lane_dist = copy.copy(sub_data['lane_dist'][-1])
    # main_lane_dist = copy.copy(main_data['lane_dist'][0])
    #for i in range(int(sub_data['frame_index'][0]) - int(main_data['frame_index'][-1]) - 1):
    #     if i < (sub_data['frame_index'][0] - main_data['frame_index'][-1]) / 2:
    #         lane_dist.append(main_lane_dist)
    #     else:
    #         lane_dist.append(sub_lane_dist)
    # new_lane_dist = main_data['lane_dist'] + lane_dist + sub_data['lane_dist']
    y_ls = [pos_x, pos_y, drivingline, drivingline_dist]
    new_pos_x, new_pos_y, new_drivingline, new_drivingline_dist = get_inter_record(frame_index, y_ls)
    new_lane_id = [round(x) for x in get_inter_record(frame_index, [lane_id], 'nearest')[0]]
    new_drivingline_line = [[x, y] for x, y in zip(new_drivingline, new_drivingline_dist)]
    new_veh_data = {'frame_index': new_frame_index, 'pixel_cpos_x': new_pos_x, 'pixel_cpos_y': new_pos_y,
                    'drivingline': {line_name: new_drivingline_line}, 'lane_id': new_lane_id,
                    'start_unix_time': start_unix_time, 'detaT': detaT, 'vehicle_length': vehicle_length}
    return new_veh_data


def get_inter_record(frame_index_ls, y_ls, kind='slinear'):
    '''
    对位置和速度进行插值 默认采用线性插值
    :param frame_index_ls:
    :param y_ls:
    :param kind:
    :return:
    '''
    new_y_ls = []
    new_frame_index = list(range(int(frame_index_ls[0]), int(frame_index_ls[-1]) + 1, 1))
    for y in y_ls:
        f = interpolate.interp1d(frame_index_ls, y, kind=kind)
        new_y = f(new_frame_index).tolist()
        new_y_ls.append(new_y)
    return new_y_ls


if __name__ == '__main__':
    import pickle

    tppkl_path = '/data3/liyitong/HuRong_process/A2/20220616_0725_A2_F2_372_1_Num_4/tp_result_20220616_0725_A2_F2_372_1_new_1.tppkl'
    new_tppkl_path = '/data3/liyitong/HuRong_process/A2/20220616_0725_A2_F2_372_1_Num_4/tp_result_20220616_0725_A2_F2_372_1_new_2.tppkl'
    # region = [250,400,280,380, 'linear']
    with open(tppkl_path, 'rb') as f:
        vehicle_data = pickle.load(f)
    print('load success:%s' % tppkl_path)
    main_short_loss(vehicle_data)

    f_save = open(new_tppkl_path, 'wb')
    pickle.dump(vehicle_data, f_save)
    print('save success:%s' % new_tppkl_path)
    f_save.close()

