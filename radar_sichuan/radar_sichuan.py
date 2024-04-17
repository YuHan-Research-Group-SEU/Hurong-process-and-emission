import json
import pickle



# 将雷达数据格式进行转换
def run_main(file_path,save_path):

    # 打开文件并逐行加载JSON数据
    with open (file_path , 'r') as file:
        # 逐行读取文件内容
        lines = file.readlines ()
    # 初始化一个空列表，用于存储每个JSON对象
    json_objects = []
    # 遍历每一行，加载JSON对象并添加到列表中
    for line in lines:
        try:
            json_object = json.loads (line)
            json_objects.append (json_object)
        except json.decoder.JSONDecodeError as e:
            print (f"Error decoding JSON: {e}")
    # 打印加载的JSON对象列表

    veh_id_list = []
    for item in json_objects:
        veh_id_frame = item['identities']
        for each_veh in veh_id_frame:
            each_veh_id = each_veh['id']
            if each_veh_id not in veh_id_list:
                veh_id_list.append(each_veh_id)
    #print(veh_id_list)
    vehicles_data = {}

    for veh_id_find in veh_id_list:
        frame_index_ls = []
        drivingline_ls = []
        lane_id_ls = []
        lng_ls = []
        lat_ls = []
        speed_x_ls = []
        speed_y_ls = []
        detaT = 0.05
        #start_unix = 0
        for i in range(len(json_objects)):
            item_frame = json_objects[i]
            time = item_frame['time']
            frame_no = item_frame['frame_no']
            identities = item_frame['identities']

            for item in identities:
                id = item ['id']
                if id == veh_id_find:
                    if len(frame_index_ls) == 0:
                        start_unix = (time,50)
                    x = item['x']
                    y = item['y']
                    line = item['line']
                    vx = item['vx']
                    vy = item['vy']
                    lng = item['lng']
                    lat = item['lat']
                    size_x = item['size_x']
                    size_y = item['size_y']
                    drivingline = [y,x]
                    drivingline_ls.append(drivingline)
                    frame_index_ls.append(frame_no)
                    lane_id_ls.append(line)
                    speed_x_ls.append(vx)
                    speed_y_ls.append(vy)
                    lng_ls.append(lng)
                    lat_ls.append(lat)
        veh_data = {}
        drivingline_dict = { }
        drivingline_dict['mainroad'] = drivingline_ls
        veh_data['drivingline'] = drivingline_dict
        veh_data['speed_x'] = speed_x_ls
        veh_data['speed_y'] = speed_y_ls
        veh_data['lng'] = lng_ls
        veh_data['lat'] = lat_ls
        veh_data['lane_id'] = lane_id_ls
        veh_data['frame_index'] = frame_index_ls
        veh_data['size_x'] = size_x
        veh_data['size_y'] = size_y
        veh_data['start_unix_time'] = start_unix
        veh_data['detaT'] = detaT
        vehicles_data[veh_id_find] = veh_data
    print(vehicles_data)
    # 使用pickle.load()加载字典
    with open (save_path , 'wb') as file:
        pickle.dump (vehicles_data , file)
    # for i in range(len(json_objects)):
    #     if i==0:
    #         time_now = json_objects[i]['time']
    #     else:
    #         time_now = json_objects [i]['time']
    #         time_last = json_objects [i-1]['time']
    #         time_dur = time_now-time_last
    #         print
    # vehicles_data = {}
    #veh_id_now =


if __name__ == '__main__':
    txt_path = '/data3/liyitong/radar_sichuan/TRACK_K1_828_HEAD_2023121811.txt'
    save_path = '/data3/liyitong/radar_sichuan/TRACK_K1_828_HEAD_2023121811.pkl'
    run_main(txt_path,save_path)
