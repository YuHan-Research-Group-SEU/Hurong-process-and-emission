# 读取数据


import pandas as pd
import pickle

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing.pool import Pool

data1 = pd.read_csv("/home/liyitong/Jupyter/test_01.csv")
data2 = pd.read_csv("/home/liyitong/Jupyter/test_02.csv")
data3 = pd.read_csv("/home/liyitong/Jupyter/test_03.csv")
data4 = pd.read_csv("/home/liyitong/Jupyter/test_04.csv")
data5 = pd.read_csv("/home/liyitong/Jupyter/test_05.csv")
data6 = pd.read_csv("/home/liyitong/Jupyter/test_06.csv")
data7 = pd.read_csv("/home/liyitong/Jupyter/test_07.csv")
data8 = pd.read_csv("/home/liyitong/Jupyter/test_08.csv")
data9 = pd.read_csv("/home/liyitong/Jupyter/test_09.csv")
# print(data)
# print(data.columns)
# print(data1)
list1 = list[data1, data2, data3, data4, data5, data6, data7, data8, data9]
for i in range(len(data1['Unnamed: 0'])):
    # print(type(i))
    j = data1['Unnamed: 0'][i].split('-')
    down = eval(j[0])
    up = eval(j[1])
    down += 3940
    up += 3940
    down = str(down)
    up = str(up)
    data1['Unnamed: 0'][i] = down + '-' + up

for i in range(len(data2['Unnamed: 0'])):
    # print(type(i))
    j = data2['Unnamed: 0'][i].split('-')
    down = eval(j[0])
    up = eval(j[1])
    down += 3610
    up += 3610
    down = str(down)
    up = str(up)
    data2['Unnamed: 0'][i] = down + '-' + up

for i in range(len(data2['Unnamed: 0'])):
    # print(type(i))
    j = data3['Unnamed: 0'][i].split('-')
    down = eval(j[0])
    up = eval(j[1])
    down += 3085
    up += 3085
    down = str(down)
    up = str(up)
    data3['Unnamed: 0'][i] = down + '-' + up

for i in range(len(data2['Unnamed: 0'])):
    # print(type(i))
    j = data4['Unnamed: 0'][i].split('-')
    down = eval(j[0])
    up = eval(j[1])
    down += 2555
    up += 2555
    down = str(down)
    up = str(up)
    data4['Unnamed: 0'][i] = down + '-' + up

for i in range(len(data2['Unnamed: 0'])):
    # print(type(i))
    j = data5['Unnamed: 0'][i].split('-')
    down = eval(j[0])
    up = eval(j[1])
    down += 2070
    up += 2070
    down = str(down)
    up = str(up)
    data5['Unnamed: 0'][i] = down + '-' + up

for i in range(len(data2['Unnamed: 0'])):
    # print(type(i))
    j = data6['Unnamed: 0'][i].split('-')
    down = eval(j[0])
    up = eval(j[1])
    down += 1530
    up += 1530
    down = str(down)
    up = str(up)
    data6['Unnamed: 0'][i] = down + '-' + up

for i in range(len(data2['Unnamed: 0'])):
    # print(type(i))
    j = data7['Unnamed: 0'][i].split('-')
    down = eval(j[0])
    up = eval(j[1])
    down += 1020
    up += 1020
    down = str(down)
    up = str(up)
    data7['Unnamed: 0'][i] = down + '-' + up

for i in range(len(data2['Unnamed: 0'])):
    # print(type(i))
    j = data8['Unnamed: 0'][i].split('-')
    down = eval(j[0])
    up = eval(j[1])
    down += 500
    up += 500
    down = str(down)
    up = str(up)
    data8['Unnamed: 0'][i] = down + '-' + up

merged_df = pd.concat([data9, data8], ignore_index=True, sort=False)
merged_df = pd.concat([merged_df, data7], ignore_index=True, sort=False)
merged_df = pd.concat([merged_df, data6], ignore_index=True, sort=False)
merged_df = pd.concat([merged_df, data5], ignore_index=True, sort=False)
merged_df = pd.concat([merged_df, data4], ignore_index=True, sort=False)
merged_df = pd.concat([merged_df, data3], ignore_index=True, sort=False)
merged_df = pd.concat([merged_df, data2], ignore_index=True, sort=False)
merged_df = pd.concat([merged_df, data1], ignore_index=True, sort=False)

merged_df.to_csv('快速路速度热力图F10.csv')
print(merged_df)
# data9 = merged_df[(merged_df != 0).any(axis=1)]
# print(merged_df)


merged_df.set_index(['Unnamed: 0'], inplace=True)
# df_vehicle_speed = merged_df[(merged_df != 0).any(axis=1)]
zero_counts = (merged_df == 0).sum(axis=1)
df_vehicle_speed = merged_df[zero_counts < 50]
# df_vehicle_speed = merged_df
ax = sns.heatmap(df_vehicle_speed.astype('float'), cmap='rainbow_r')
ax.set_ylim(ax.get_ylim()[::-1])
cbar = ax.collections[0].colorbar
cbar.set_label('$km/h$')
plt.title("Speed")
plt.xlabel('Time(s)')
plt.ylabel('Location (m)')

plt.show()

# 创建一个与数据集相同形状的mask，速度为0的位置为True，其余为False
# mask = np.zeros_like(data1.iloc[:, 1:], dtype=np.bool)
# mask[data1.iloc[:, 1:] == 0] = True
# mask = np.zeros_like(data2.iloc[:, 1:], dtype=np.bool)
# mask[data2.iloc[:, 1:] == 0] = True

# 绘制速度图
# plt.figure(figsize=(10,5),dpi=1000)
# print(data1)
# ax = sns.heatmap(df_vehicle_speed.iloc[:, 1:].astype('float'), cmap='rainbow_r', yticklabels=False)
# sns.heatmap(data1.iloc[:, 1:].astype('float'), cmap='hot',xticklabels = "auto", yticklabels="auto", mask=mask, cbar_kws={'label': '$km/h$'})
# sns.heatmap(data2.iloc[:, 1:].astype('float'), cmap='hot',xticklabels = "auto", yticklabels="auto", mask=mask, cbar_kws={'label': '$km/h$'})
# ax.set_ylim(ax.get_ylim()[::-1])
# cbar = ax.collections[0].colorbar
# cbar.set_label('$km/h$')

# plt.xlabel('Time(s)')
# plt.ylabel('Location (m)')
# plt.show()