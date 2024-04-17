#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time : 2022-03-25 20:02
@Author : Xinkai Ji
@contact: jhjxk@hotmail.com
@File : Vehicle.py
@Software: PyCharm
@desc:
'''
import numpy as np

class Record:
    def __init__(self,frame_index,speed,score,cls,pixel_pos,geo_pos,lane_id,unixtime,pixel_cpos,geo_cpos):
        self.frame_index = frame_index
        self.geo_speed = speed
        self.pixel_pos = pixel_pos
        self.geo_pos = geo_pos
        self.lane_id = lane_id
        self.unixtime = unixtime
        self.score = score
        self.veh_class = cls
        self.pixel_cpos = pixel_cpos
        self.geo_cpos = geo_cpos


class Vehicle:
    def __init__(self,id):
        self.ID = id
        self.record = []

    def add_record(self,frame_index,speed,score,cls,pixel_pos,geo_pos,lane_id,unixtime,pixel_cpos,geo_cpos):
        record_data = Record(frame_index,speed,score,cls,pixel_pos,geo_pos,lane_id,unixtime,pixel_cpos,geo_cpos)
        self.record.append(record_data)

    def get_last_record(self):
        return self.record[-1]



class VehicleCollection:
    def __init__(self,detaT):
        self.vehicles = {}
        self.detaT = detaT


    def add_frame_data(self,output_frame, o_bboxs_res, frame_time):
        # 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'score', 'class', 'id', 'X1','Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4', 'lane_id'
        # '2022-03-03 17:24:50,201,595'
        for item in o_bboxs_res:
            pixel_pos = item[:8]
            score = item[8]
            cls = item[9]
            id = item[10]
            geo_pos = item[11:19]
            lane_id = item[19]
            self.add_vehicle(id,output_frame,score,cls,pixel_pos,geo_pos,lane_id,frame_time)




    def add_vehicle(self,id,frame_index, score, cls, pixel_pos, geo_pos, lane_id, unixtime):
        geo_cpos = np.mean(geo_pos.reshape(-1, 2), axis=0)
        pixel_cpos = np.mean(pixel_pos.reshape(-1, 2), axis=0)
        if id in self.vehicles:
            veh = self.vehicles[id]
            rec = veh.get_last_record()
            speed = (geo_cpos-rec.geo_cpos)/((frame_index-rec.frame_index)*self.detaT)
            veh.add_record(frame_index,speed,score,cls,pixel_pos,geo_pos,lane_id,unixtime,pixel_cpos,geo_cpos)
        else:
            veh = Vehicle(id)
            veh.add_record(frame_index,None,score,cls,pixel_pos,geo_pos,lane_id,unixtime,pixel_cpos,geo_cpos)
            self.vehicles[id] = veh

