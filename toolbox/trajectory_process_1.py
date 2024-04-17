#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time : 2022-07-09 16:10
@Author : Xinkai Ji
@contact: jhjxk@hotmail.com
@File : trajectory_process.py
@Software: PyCharm
@desc:
'''
import os.path
import sys

sys.path.append(os.getcwd())

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


def get_unixtime(timestr):
    '''
    W&2l:unixtime(��)
    :param timestr:  '2022-06-17 06:59:46,837,622'
    :return:
    '''
    if isinstance(timestr, str):
        timestr = timestr[:-4]
        datetime_obj = datetime.strptime(timestr, "%Y-%m-%d %H:%M:%S,%f")
        int_unix_time = int(time.mktime(datetime_obj.timetuple()) * 1e3 + datetime_obj.microsecond / 1e3)
    else:
        int_unix_time = int(timestr)
    return int_unix_time


def unixtime2time(int_unix_time):
    timeStamp = int_unix_time / 1000.0
    timeArray = datetime.fromtimestamp(timeStamp)
    return timeArray


class TrajectoryProcess:

    def __init__(self):
        self.vaild_vehicle = []
        self.vehicles_data = {}
        self.pixel2xy_matrix = None
        # self.vehicles_length = {}
        # self.vehicles_start_time = {}

    def load_data_from_raw_pkl(self, file_path):
        '''
        Ο˓��pkl��-�}pn
        :param file_path:
        :return:
        '''
        with open(file_path, 'rb') as f:
            traj_data = pickle.load(f)
        print('load success:%s' % file_path)
        # print('first item',traj_data[0])
        output_fps = traj_data['output_info']['output_fps']
        self.vehicle = self.traj2vehicle(traj_data['traj_info'], output_fps)

    def traj2vehicle(self, traj_info, fps):
        veh_coll = VehicleCollection(detaT=1 / fps)
        for obj_data in traj_info:
            frame_time = 'NA'
            if len(obj_data) == 3:
                frame_index, output_frame, o_bboxs_res = obj_data
            elif len(obj_data) == 2:
                output_frame, o_bboxs_res = obj_data
                frame_index = output_frame
            elif len(obj_data) == 4:
                frame_index, output_frame, o_bboxs_res, frame_time = obj_data
            if o_bboxs_res.shape[1] == 11:
                o_bboxs_res = self.pixelxy2geoxy(o_bboxs_res)
            if o_bboxs_res.shape[1] == 19:
                b = np.ones(o_bboxs_res.shape[0]) * (-1)
                o_bboxs_res = np.c_[o_bboxs_res, b]

            veh_coll.add_frame_data(output_frame, o_bboxs_res, frame_time)
        return veh_coll

    def pixelxy2geoxy(self, nms_result):
        if self.pixel2xy_matrix is None:
            from utils.MultiVideos import MultiVideos
            multi_config_json = '../config/yingtianstreet/0708/multi_20220708_B_F2.json'
            multi_videos = MultiVideos(multi_config_json)
            road_config = multi_videos.main_video.road_config
            self.pixel2xy_matrix = road_config['pixel2xy_matrix']

        pixel_data = nms_result[:, :8].copy()
        pixel_data = pixel_data.reshape(-1, 2)
        b = np.ones(pixel_data.shape[0])
        pixel_data = np.column_stack((pixel_data, b))
        xy_data = np.matmul(self.pixel2xy_matrix, pixel_data.T).T.reshape(-1, 8)
        return np.hstack((nms_result, xy_data))

    def save_data2pkl(self, file_name):
        print('start save data:%s' % file_name)
        with open(file_name, 'wb') as f:
            pickle.dump(self.vehicles_data, f)

    def load_vehicle_data(self, veh_data_path):
        '''
        �vehpkl��-�}pn
        :param veh_data_path:
        :param data_name:
        :return:
        '''
        print('start load data...')
        with open(veh_data_path, 'rb') as f:
            self.vehicle = pickle.load(f)

    def get_valid_vehicle(self, remove_pixel_region):
        '''
        ��	H�f�id
        :return:
        '''
        print('start get valid data, and interpolate and smooth data...')
        self.vaild_vehicle = []
        for id, v in self.vehicle.vehicles.items():
            records = v.record
            #  d�200'�f
            if len(records) < int(3 / self.vehicle.detaT):
                continue
            xs = records[0].geo_cpos[0]
            ys = records[0].geo_cpos[1]
            xe = records[-1].geo_cpos[0]
            ye = records[-1].geo_cpos[1]
            #  dLvݻ�30s�f
            if math.sqrt((xs - xe) ** 2 + (ys - ye) ** 2) < 30:
                continue
            self.vaild_vehicle.append(id)

        for valid_id in tqdm(self.vaild_vehicle):
            record_ls = self.vehicle.vehicles[valid_id].record
            vehicle_length = self.get_vehicle_length(record_ls)
            # self.vehicles_length[valid_id] = vehicle_length

            self.vehicles_data[valid_id] = self.get_new_record(record_ls, remove_pixel_region)
            self.vehicles_data[valid_id]['start_unix_time'] = (
            get_unixtime(record_ls[0].unixtime), self.vehicle.detaT * 1e3)
            self.vehicles_data[valid_id]['vehicle_length'] = vehicle_length
            self.vehicles_data[valid_id]['detaT'] = self.vehicle.detaT

    def get_new_record(self, record_ls, remove_pixel_region=None):
        '''
        frame index����pn�L�<
        :param record_ls:
        :return:
        '''
        frame_index_ls = []
        # geo_speed_ls = []
        pixel_cpos_x_ls = []
        pixel_cpos_y_ls = []
        geo_cpos_x_ls = []
        geo_cpos_y_ls = []
        for record in record_ls:
            frame_index_ls.append(record.frame_index)
            pixel_cpos_x_ls.append(record.pixel_cpos[0])
            pixel_cpos_y_ls.append(record.pixel_cpos[1])
            geo_cpos_x_ls.append(record.geo_cpos[0])
            geo_cpos_y_ls.append(record.geo_cpos[1])
        if not remove_pixel_region is None:
            if remove_pixel_region[0] == 'x':
                remove_frame_index = self.get_remove_stitch_region_index(pixel_cpos_x_ls, remove_pixel_region[1])
            else:
                remove_frame_index = self.get_remove_stitch_region_index(pixel_cpos_y_ls, remove_pixel_region[1])
            for i in reversed(remove_frame_index):
                frame_index_ls.pop(i)
                pixel_cpos_x_ls.pop(i)
                pixel_cpos_y_ls.pop(i)
                geo_cpos_x_ls.pop(i)
                geo_cpos_y_ls.pop(i)

        new_y_ls = self.get_inter_record(frame_index_ls,
                                         [frame_index_ls, pixel_cpos_x_ls, pixel_cpos_y_ls, geo_cpos_x_ls,
                                          geo_cpos_y_ls])

        new_pixel_cpos_x_ls = smoothWithsEMA(new_y_ls[1], 0.5, self.vehicle.detaT)
        new_pixel_cpos_y_ls = smoothWithsEMA(new_y_ls[2], 0.5, self.vehicle.detaT)
        new_geo_cpos_x_ls = smoothWithsEMA(new_y_ls[3], 0.5, self.vehicle.detaT)
        new_geo_cpos_y_ls = smoothWithsEMA(new_y_ls[4], 0.5, self.vehicle.detaT)
        new_y_dict = {'frame_index': new_y_ls[0], 'pixel_cpos_x': new_pixel_cpos_x_ls,
                      'pixel_cpos_y': new_pixel_cpos_y_ls,
                      'geo_cpos_x': new_geo_cpos_x_ls, 'geo_cpos_y': new_geo_cpos_y_ls}
        return new_y_dict

    def load_drivingline(self, json_file_ls):
        '''
        �}Lf�
        :param json_file_ls:
        :return:
        '''
        from module.DrivingLine import DrivingLineList
        self.drivingline_ls = DrivingLineList(json_file_ls)

    def get_drivingline_dist(self, position_id):
        '''
        ��Lf�Lvݻ�
        :param drivingline_name:
        :param position_id:
        :return:
        '''
        print('start get drivingline distance...')

        for valid_id in tqdm(self.vaild_vehicle):
            self.vehicles_data[valid_id]['drivingline'] = {}
            for drivingline_name in self.drivingline_ls.drivingline_name_list:
                pixel_cpos_x_ls = self.vehicles_data[valid_id]['pixel_cpos_x']
                pixel_cpos_y_ls = self.vehicles_data[valid_id]['pixel_cpos_y']
                xy = [[x, y] for x, y in zip(pixel_cpos_x_ls, pixel_cpos_y_ls)]
                drivingline_dist = self.drivingline_ls.get_global_distance(xy, position_id, drivingline_name, True)
                self.vehicles_data[valid_id]['drivingline'][drivingline_name] = drivingline_dist

    def load_laneline(self, road_config):
        '''
        �}fS�
        :return:
        '''
        from module.Lane import Lane
        line_string_dict = road_config['laneline']
        length_per_pixel = road_config['length_per_pixel']
        lane_region = road_config['lane']
        self.laneline = Lane(line_string_dict, lane_region, length_per_pixel)
        self.length_per_pixel = length_per_pixel

    def get_lane_id_dist(self):
        '''
        ��fSid�ݻfS���ݻ
        :return:
        '''
        print('start get lane id and distance...')
        for valid_id in tqdm(self.vaild_vehicle):
            pixel_cpos_x_ls = self.vehicles_data[valid_id]['pixel_cpos_x']
            pixel_cpos_y_ls = self.vehicles_data[valid_id]['pixel_cpos_y']
            xy = [[x, y] for x, y in zip(pixel_cpos_x_ls, pixel_cpos_y_ls)]
            lane_id_ls, lane_dist_ls = self.laneline.get_lane_id_from_Polygon(xy)
            self.vehicles_data[valid_id]['lane_id'] = lane_id_ls
            self.vehicles_data[valid_id]['lane_dist'] = lane_dist_ls

    def get_vehicle_length(self, record_ls, kind='median'):
        '''
        ��f��� ؤ�(-Mp
        :param record_ls:
        :param kind:
        :return:
        '''
        length_ls = []
        for current_r in record_ls:
            geo_pos = current_r.geo_pos.reshape(4, 2)
            vector1 = geo_pos[0, :] - geo_pos[1, :]
            vector2 = geo_pos[1, :] - geo_pos[2, :]
            length_vector1 = np.linalg.norm(vector1)
            legnth_vector2 = np.linalg.norm(vector2)
            length = max(length_vector1, legnth_vector2)
            length_ls.append(length)
        if kind == 'median':
            return np.median(length_ls)
        else:
            return np.mean(length_ls)

    def load_tppkl(self, pkl_file):
        '''
        �etppklpn
        :param pkl_file:
        :return:
        '''
        with open(pkl_file, 'rb') as f:
            self.vehicles_data = pickle.load(f)
        print('load success:%s' % pkl_file)

    def get_vehicles_from_lane(self, target_lane_id, driving_name=None, x_is_unixtime=False, output_points=False):
        '''
        ���fS
�f�
        :param target_lane_id:
        :param driving_name:
        :param x_is_unixtime:
        :return:
        '''
        lines_ls = []
        speed_ls = []
        if output_points:
            start_points = [[], []]
            finish_points = [[], []]
            lc_points = [[], []]
        for veh_id, veh_data in self.vehicles_data.items():
            # if veh_id not in [508,553]:
            #     continue
            start_unix_time, ns_detaT = veh_data['start_unix_time']
            detaT = ns_detaT / 1000
            lane_id = veh_data['lane_id']
            frame_index = veh_data['frame_index']
            #print(detaT)

            #print(veh_data.keys())
            if driving_name:
                drivingline = veh_data['drivingline']
            else:
                assert len(list(veh_data['drivingline'].keys())) == 1, 'give the drivingline name'
                temp_driving_name = list(veh_data['drivingline'].keys())[0]
                #print(temp_driving_name)
                drivingline = veh_data['drivingline'][temp_driving_name]
            #print(type(drivingline[0]))
            drivingline_dist = smoothWithsEMA([x[0] for x in drivingline], 0.3, detaT)
            for i in range(len(lane_id) - 1):
                if x_is_unixtime:
                    x1 = (frame_index[i] - frame_index[0]) * ns_detaT + start_unix_time
                    x2 = (frame_index[i + 1] - frame_index[0]) * ns_detaT + start_unix_time

                else:
                    x1 = frame_index[i] * detaT
                    x2 = frame_index[i + 1] * detaT
                if lane_id[i] == target_lane_id and lane_id[i + 1] == target_lane_id:
                    y1 = drivingline_dist[i]
                    y2 = drivingline_dist[i + 1]
                    speed = abs((y2 - y1) / detaT * 3.6)
                    speed_ls.append(speed)
                    lines_ls.append([(x1, y1), (x2, y2)])
                if output_points:
                    if lane_id[i] != lane_id[i + 1] and (
                            lane_id[i] == target_lane_id or lane_id[i + 1] == target_lane_id):
                        if lane_id[i] == target_lane_id:
                            lc_points[0].append(x1)
                            lc_points[1].append(drivingline_dist[i])
                        else:
                            lc_points[0].append(x2)
                            lc_points[1].append(drivingline_dist[i + 1])
            if output_points:
                if x_is_unixtime:
                    x_s = start_unix_time
                    x_e = (frame_index[-1] - frame_index[0]) * ns_detaT + start_unix_time
                else:
                    x_s = frame_index[0] * detaT
                    x_e = frame_index[-1] * detaT
                start_points[0].append(x_s)
                start_points[1].append(drivingline_dist[0])
                finish_points[0].append(x_e)
                finish_points[1].append(drivingline_dist[-1])
        if output_points:
            points = {'start': start_points, 'finish': finish_points, 'lc': lc_points}
            return lines_ls, speed_ls, points
        return lines_ls, speed_ls

    def get_xy(self, x_is_unixtime=False):
        xline_ls = []
        yline_ls = []

        speedx_ls = []
        speedy_ls = []
        for veh_id, veh_data in self.vehicles_data.items():

            start_unix_time, ns_detaT = veh_data['start_unix_time']
            detaT = ns_detaT / 1000

            frame_index = veh_data['frame_index']
            xline = veh_data['geo_cpos_x']
            yline = veh_data['geo_cpos_y']
            xline_dist = smoothWithsEMA([xl for xl in xline], 0.3, detaT)
            yline_dist = smoothWithsEMA([yl for yl in yline], 0.3, detaT)
            for i in range(len(frame_index) - 1):
                if x_is_unixtime:
                    t1 = (frame_index[i] - frame_index[0]) * ns_detaT + start_unix_time
                    t2 = (frame_index[i + 1] - frame_index[0]) * ns_detaT + start_unix_time
                else:
                    t1 = frame_index[i] * detaT
                    t2 = frame_index[i + 1] * detaT
                    x1 = xline_dist[i]
                    x2 = xline_dist[i + 1]
                    y1 = yline_dist[i]
                    y2 = yline_dist[i + 1]
                    speedx = (x2 - x1) / detaT * 3.6
                    speedy = (y2 - y1) / detaT * 3.6
                    speedx_ls.append(speedx)
                    speedy_ls.append(speedy)
                    xline_ls.append([(t1, x1), (t2, x2)])
                    yline_ls.append([(t1, y1), (t2, y2)])
        return xline_ls, speedx_ls, yline_ls, speedy_ls

    def plot_xy(self, savepath, x, color_speed, figsize=(20, 5), start_time=None):
        '''
        �6xyh��zh��
        '''
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()
        x_sc = LineCollection(x, array=np.array(color_speed), cmap="jet_r", linewidths=0.1)
        ax.add_collection(x_sc)
        x_sc.set_clim(vmin=0, vmax=120)
        cb = fig.colorbar(x_sc)
        if not start_time is None:
            plt.title('Start time:%s' % start_time)
        ax.autoscale()
        cb.ax.set_title('speed [km/h]', fontsize=8)
        plt.xlabel("Time [s]")
        plt.ylabel("Location [m]")
        plt.savefig(savepath, dpi=1000)
        print('save_img:%s' % savepath)

    def plot_line(self, savepath, lines, color_speed, figsize=(15, 5), start_time=None, points=None):
        '''
        �6h��zh��
        :param savepath:
        :param lines:
        :param color_speed:
        :param figsize:
        :param start_time:
        :return:
        '''
        fig = plt.figure(figsize=figsize)
        # fig, ax = plt.subplots(figsize=(15, 5))
        ax = fig.add_subplot()
        lines_sc = LineCollection(lines, array=np.array(color_speed), cmap="jet_r", linewidths=0.2)
        ax.add_collection(lines_sc)
        lines_sc.set_clim(vmin=0, vmax=120)
        cb = fig.colorbar(lines_sc)
        if not start_time is None:
            plt.title('Start time:%s' % start_time,fontsize=20)
        if not points is None:
            size = 1
            # start_points = points.get('start',None)
            # if not start_points is None:
            #     plt.scatter(start_points[0],start_points[1],c='k',marker='^',label='starting point',s=size)
            # end_points = points.get('finish', None)
            # if not end_points is None:
            #     plt.scatter(end_points[0], end_points[1], c='k', marker='s',label='finishing point',s=size)
            lc_points = points.get('lc', None)
            if not lc_points is None:
                plt.scatter(lc_points[0], lc_points[1], c='k', marker='x', label='lane changing point', s=size)
            plt.legend(loc='upper right',fontsize=16)
        ax.autoscale()
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        # 设置colorbar刻度的字体大小
        cb.ax.tick_params(labelsize=18)
        cb.ax.set_title('speed [km/h]', fontsize=18)
        plt.xlabel("Time [s]",fontsize=18)
        plt.ylabel("Location [m]",fontsize=18)

        # plt.grid(None)
        # plt.show()
        plt.savefig(savepath, dpi=1000,bbox_inches='tight')
        print('save_img:%s' % savepath)

    def get_vehicles_choose (self, lane_id_ls,veh_choose,driving_name=None ,x_is_unixtime=False ,output_points=False) :
        lines_ls = []
        speed_ls = []
        if output_points :
            start_points = [[] , []]
            finish_points = [[] , []]
            lc_points = [[] , []]
        for veh_id , veh_data in self.vehicles_data.items () :
            if veh_id in veh_choose:
                start_unix_time , ns_detaT = veh_data ['start_unix_time']
                detaT = ns_detaT / 1000
                lane_id = veh_data ['lane_id']
                frame_index = veh_data ['frame_index']

                if driving_name :
                    drivingline = veh_data ['drivingline']
                else :
                    assert len (list (veh_data ['drivingline'].keys ())) == 1 , 'give the drivingline name'
                    temp_driving_name = list (veh_data ['drivingline'].keys ()) [0]
                    # print(temp_driving_name)
                    drivingline = veh_data ['drivingline'] [temp_driving_name]
                # print(type(drivingline[0]))
                drivingline_dist = smoothWithsEMA ([x [0] for x in drivingline] , 0.3 , detaT)
                for i in range (len (lane_id) - 1) :
                    if x_is_unixtime :
                        x1 = (frame_index [i] - frame_index [0]) * ns_detaT + start_unix_time
                        x2 = (frame_index [i + 1] - frame_index [0]) * ns_detaT + start_unix_time

                    else :
                        x1 = frame_index [i] * detaT
                        x2 = frame_index [i + 1] * detaT
                    if lane_id[i] in lane_id_ls:
                        y1 = drivingline_dist [i]
                        y2 = drivingline_dist [i + 1]
                        speed = abs ((y2 - y1) / detaT * 3.6)
                        speed_ls.append (speed)
                        lines_ls.append ([(x1 , y1) , (x2 , y2)])
                    if output_points :
                        if lane_id [i] != lane_id [i + 1] :
                            lc_points [0].append (x2)
                            lc_points [1].append (drivingline_dist [i + 1])
                if output_points :
                    if x_is_unixtime :
                        x_s = start_unix_time
                        x_e = (frame_index [-1] - frame_index [0]) * ns_detaT + start_unix_time
                    else :
                        x_s = frame_index [0] * detaT
                        x_e = frame_index [-1] * detaT
                    start_points [0].append (x_s)
                    start_points [1].append (drivingline_dist [0])
                    finish_points [0].append (x_e)
                    finish_points [1].append (drivingline_dist [-1])
        if output_points :
            points = { 'start' : start_points , 'finish' : finish_points , 'lc' : lc_points }
            return lines_ls , speed_ls , points
        return lines_ls , speed_ls


    @staticmethod

    def get_inter_record(frame_index_ls, y_ls, kind='slinear'):
        '''
        �Mn���L�< ؤ�(�'�<
        :param frame_index_ls:
        :param y_ls:
        :param kind:
        :return:
        '''
        new_y_ls = []
        new_frame_index = list(range(frame_index_ls[0], frame_index_ls[-1] + 1, 1))
        for y in y_ls:
            f = interpolate.interp1d(frame_index_ls, y, kind=kind)
            new_y = f(new_frame_index)
            new_y_ls.append(new_y)
        return new_y_ls

    @staticmethod
    def get_remove_stitch_region_index(pixel_value_ls, pixel_region):
        remove_index = []
        for index, pixel_v in enumerate(pixel_value_ls):
            if pixel_region[0] <= pixel_v <= pixel_region[1]:
                remove_index.append(index)
        return remove_index

    def process(self, road_config_json, flight_position_id, raw_data_path, driving_line_json_ls=None,
                remove_pixel_region=None):
        '''
        pn
        :param road_config_json: S�config��
        :param flight_position_id: w޹id
        :param raw_data_path: ��pklpn
        :param driving_line_json_ls: Lf�json
        :param remove_pixel_region: �d�:߄h� ['x',[4805-50,4805+50]]
        :return:
        '''
        from utils import RoadConfig
        save_folder, _ = os.path.split(raw_data_path)
        _, file_name = os.path.split(road_config_json)
        video_name, _ = os.path.splitext(file_name)
        if driving_line_json_ls is None:
            driving_line_json_ls = [road_config_json]
        road_config = RoadConfig.fromfile(road_config_json)

        self.load_data_from_raw_pkl(raw_data_path)
        self.get_valid_vehicle(remove_pixel_region)
        self.load_drivingline(driving_line_json_ls)
        self.get_drivingline_dist(flight_position_id)

        self.load_laneline(road_config)
        self.get_lane_id_dist()
        file_name = os.path.join(save_folder, 'tp_result_%s.tppkl' % video_name)
        self.save_data2pkl(file_name)

    def get_leader_vehicle(self, tppkl_path):
        self.load_tppkl(tppkl_path)
        save_folder, tppkl_file_name = os.path.split(tppkl_path)
        video_name, _ = os.path.splitext(tppkl_file_name)
        video_name = video_name[10:]
        df_data = []
        for veh_id, veh_data in self.vehicles_data.items():
            temp_driving_name = list(veh_data['drivingline'].keys())[0]
            drivingline = veh_data['drivingline'][temp_driving_name]
            for i in range(len(veh_data['frame_index'])):
                frame_index = int(veh_data['frame_index'][i])
                drivingline_dist_x = drivingline[i][0]
                drivingline_dist_y = drivingline[i][1]
                lane_id = veh_data['lane_id'][i]
                temp_d = [frame_index, veh_id, lane_id, drivingline_dist_x, drivingline_dist_y]
                df_data.append(temp_d)
        df_veh = pd.DataFrame(df_data, columns=['frame_index', 'vehicle_id', 'lane_id', 'drivingline_dist_x',
                                                'drivingline_dist_y'])
        # file_name = os.path.join(save_folder, 'drivingline_result_%s.csv' % video_name)
        # df_veh.to_csv(file_name)
        # print('output:%s'%file_name)
        start_frame_index = int(df_veh['frame_index'].min())
        end_frame_index = int(df_veh['frame_index'].max())
        all_lane = df_veh['lane_id'].unique()
        # leader_info = {}
        new_df_ls = []
        for frame_index_c in tqdm(range(start_frame_index, end_frame_index + 1)):
            for lane_id in all_lane:
                if lane_id == -1:
                    continue
                if lane_id > 19:
                    ascending = False
                else:
                    ascending = True
                lane_veh = df_veh[(df_veh['frame_index'] == frame_index_c) & (df_veh['lane_id'] == lane_id)]. \
                    sort_values(by='drivingline_dist_x', ascending=ascending)  # ascending=TrueG�
                for i in range(len(lane_veh)):
                    current_veh_id = lane_veh.iloc[i]['vehicle_id']
                    dist_x = lane_veh.iloc[i]['drivingline_dist_x']
                    dist_y = lane_veh.iloc[i]['drivingline_dist_y']

                    if i == 0:
                        following_veh_id = None
                        following_dist_x = None
                        following_dist_y = None
                    if i == len(lane_veh) - 1:
                        leader_veh_id = None
                        leader_dist_x = None
                        leader_dist_y = None
                    if i > 0 and i < len(lane_veh):
                        following_veh_id = lane_veh.iloc[i - 1]['vehicle_id']
                        following_dist_x = lane_veh.iloc[i - 1]['drivingline_dist_x']
                        following_dist_y = lane_veh.iloc[i - 1]['drivingline_dist_y']
                    if i >= 0 and i < len(lane_veh) - 1:
                        leader_veh_id = lane_veh.iloc[i + 1]['vehicle_id']
                        leader_dist_x = lane_veh.iloc[i + 1]['drivingline_dist_x']
                        leader_dist_y = lane_veh.iloc[i + 1]['drivingline_dist_y']

                    new_line = [frame_index_c, current_veh_id, lane_id, dist_x, dist_y, leader_veh_id, leader_dist_x,
                                leader_dist_y, following_veh_id, following_dist_x, following_dist_y]
                    new_df_ls.append(new_line)
                    # leader_key = '%d_%d'%(frame_index_c,current_veh_id)
                    # leader_info[leader_key] = {'leader_id':leader_veh_id,'leader_dist_x':leader_dist_x,'leader_dist_y':leader_dist_y,
                    #                            'following_id':following_veh_id,'following_dist_x':following_dist_x,'following_dist_y':following_dist_y}

        new_df = pd.DataFrame(new_df_ls, columns=['frame_index', 'vehicle_id', 'lane_id', 'drivingline_dist_x',
                                                  'drivingline_dist_y',
                                                  'leader_vehicle_id', 'leader_drivingline_dist_x',
                                                  'leader_drivingline_dist_y',
                                                  'following_vehicle_id', 'following_drivingline_dist_x',
                                                  'following_drivingline_dist_y'])

        old_df_veh = df_veh[df_veh['lane_id'] != -1]
        if len(new_df) == len(old_df_veh):
            print('total %d lines' % len(new_df))
        else:
            print('old df:%d,new df:%d' % (len(old_df_veh), len(new_df)))
            print('loss data !!!!!')

        file_name_dlpkl = os.path.join(save_folder, 'drivingline_result_%s.dlpkl' % video_name)
        file_name_csv = os.path.join(save_folder, 'drivingline_result_%s.csv' % video_name)
        new_df.to_pickle(file_name_dlpkl)
        new_df.to_csv(file_name_csv)


if __name__ == '__main__':

    pkl_file = '/data2/xinkaiji/mixedroad_process/20220617/output/C1-C2/M-20220617_0725_C2_F2_371_1-S-20220617_0725_C1_F2_370_1/tp_result_first_frame_M-20220617_0725_C2_F2_371_1-S-20220617_0725_C1_F2_370_1.tppkl'
    tp = TrajectoryProcess()
    tp.load_tppkl(pkl_file)
    lane_id_ls = [1, 2, 3, 4]
    for lane_id in lane_id_ls:
        lines_ls, speed_ls = tp.get_vehicles_from_lane(lane_id, x_is_unixtime=True)
        fig_path = '/data2/xinkaiji/mixedroad_process/20220617/output/C1-C2/M-20220617_0725_C2_F2_371_1-S-20220617_0725_C1_F2_370_1/fig/lane%d.jpg' % lane_id
        tp.plot_line(fig_path, lines_ls, speed_ls)

    # yingtianstreet
    # import os
    # raw_data_path = '/data3/xinkaiji/YingtianStreet_Process/20220708/output/Y1-Y2/M-20220708_Y1_A_F1_1-S-20220708_Y2_A_F1_1/stitch_bbox_result_M-20220708_Y1_A_F1_1-S-20220708_Y2_A_F1_1.pkl'
    # config_file = '/data3/xinkaiji/YingtianStreet_Process/20220708/road_config/merge/first_frame_M-20220708_Y1_A_F1_1-S-20220708_Y2_A_F1_1.json'
    # fp_position = 'Y1'
    # tp = TrajectoryProcess()
    # tp.process(config_file, fp_position, raw_data_path)

    # ;�
    # pkl_file = '/data3/xinkaiji/YingtianStreet_Process/20220708/output/Y1-Y2/M-20220708_Y1_A_F1_1-S-20220708_Y2_A_F1_1/tp_result_first_frame_M-20220708_Y1_A_F1_1-S-20220708_Y2_A_F1_1.tppkl'
    # tp = TrajectoryProcess()
    # tp.load_pkl(pkl_file)
    # lane_id_ls = [1,2,3,4]
    # for lane_id in lane_id_ls:
    #     lines_ls,speed_ls = tp.get_vehicles_from_lane(lane_id,x_is_unixtime=False)
    #     fig_path = '/data3/xinkaiji/YingtianStreet_Process/20220708/output/Y1-Y2/M-20220708_Y1_A_F1_1-S-20220708_Y2_A_F1_1/fig/lane%d.jpg'%lane_id
    #     tp.plot_line(fig_path,lines_ls,speed_ls)

    # ��csv
    # tp_pkl = '/data3/xinkaiji/YingtianStreet_Process/20220707/output/Y1-Y2/M-20220707_Y1_B_F2_1-S-20220707_Y2_B_F2_1/tp_result_first_frame_M-20220707_Y1_B_F2_1-S-20220707_Y2_B_F2_1.tppkl'

    # remove_pixel_region = ['x',[4805-50,4805+50]]
    # tp = TrajectoryProcess()
    # tp.get_leader_vehicle(tp_pkl)







