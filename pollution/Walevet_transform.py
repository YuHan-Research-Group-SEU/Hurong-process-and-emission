
import numpy as np
import matplotlib.pyplot as plt
import pywt
plt.rc("font",family='MicroSoft YaHei',weight="bold")
amplitude = 2.0  # 振幅
frequency = 1.0  # 频率（每秒的周期数）
phase = np.pi/2   # 相位（弧度）

time_steady = np.linspace(0, 10, 1000)
steady_velocity = 2.0 * np.sin(2.0 * np.pi * 0.2 * time_steady)  # 例子中使用了正弦函数

# 添加速度波动
start_wave = 5.0
end_wave = 7.0
amplitude_wave = 5.0

# 在特定时间段内添加速度波动
wave_indices = np.where((time_steady >= start_wave) & (time_steady <= end_wave))
steady_velocity[wave_indices] += amplitude_wave * np.sin(2.0 * np.pi * 1.0 * time_steady[wave_indices])

speed = steady_velocity
time = np.arange(0, 1000)
print(speed)
# 进行连续小波变换
wavelet = 'mexh'  # 墨西哥帽小波
widths = np.arange(1, 31)  # 尺度范围，可以根据需要调整
coefficients, frequencies = pywt.cwt(speed, widths, wavelet)

# 可视化结果
plt.figure(figsize=(12, 6))

# 原始速度时间序列
plt.subplot(2, 1, 1)
plt.plot(time, speed, label='original speed')
plt.title('raw velocity time series')
plt.xlabel('time')
plt.ylabel('speed')
plt.legend()

# 墨西哥帽小波拟合图
plt.subplot(2, 1, 2)
plt.imshow(np.abs(coefficients), aspect='auto', extent=[0, 10, 1, 31], cmap='jet', interpolation='bilinear')
plt.colorbar(label='amplitude')
plt.title('Continuous Wavelet Transform(Mexican hat wavelet)')
plt.xlabel('time')
plt.ylabel('arrange')

plt.tight_layout()
plt.show()

# 找到每个尺度上的系数峰值的时间点
peak_times = []
for i in range(len(widths)):
    scale_coefficients = np.abs(coefficients[i, :])
    peak_index = np.argmax(scale_coefficients)
    peak_time = time[peak_index]
    peak_times.append(peak_time)

# 输出波动的时间点
print("波动的时间点：", peak_times)


