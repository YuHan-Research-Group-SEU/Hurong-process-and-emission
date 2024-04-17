# from datetime import datetime
#
# # Unix毫秒时间戳
# unix_milliseconds_timestamp = 1702868400037
#
# # 将Unix毫秒时间戳转换为正常日期时间
# normal_time = datetime.utcfromtimestamp(unix_milliseconds_timestamp / 1000.0)
#
# # 打印转换后的日期时间
# print(normal_time)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 创建一个空白图像
fig, ax = plt.subplots()

# 初始化一个空的线对象
line, = ax.plot([], [], 'o-')

# 设置坐标轴范围
ax.set_xlim(0, 2*np.pi)
ax.set_ylim(-1, 1)

# 初始化函数，用于绘制每一帧的数据
def init():
    line.set_data([], [])
    return line,

# 更新函数，用于更新每一帧的数据
def update(frame):
    x = np.linspace(0, 2*np.pi, 1000)
    y = np.sin(x + frame*0.1)  # 这里的0.1控制动画的速度
    line.set_data(x, y)
    return line,

# 创建动画对象
ani = FuncAnimation(fig, update, frames=range(100), init_func=init, blit=True)

# 显示动画
plt.show()
