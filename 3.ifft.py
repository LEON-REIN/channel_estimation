# @.@ coding  : utf-8 ^-^
# @Author     : Leon Rein
# @Time       : 2021-03-09  ~  20:05 
# @File       : 3.ifft.py
# @Software   : PyCharm
# @Notice     : It's a WINDOWS version!
#               将并行数据进行调制. 先进行 IFFT 运算, 并加入循环前缀, 生成待发送的复包络的采样值.
#               注: 发送的 OFDM 信号为 s(t) = Re{a(t)*e^(j*2pi*f_c*t)}, a(t) 为模拟复包络
#                   忽略上变频, 即, f_c = 0Hz 时, s(t) = Re{a(t)}

"""
输入: labels64.csv

输出: after_cp80.npy
    numpy 专用二进制储存格式.
    1000 行 80 列, 每个元素为 complex128 格式, 每一行的前 16 列为最后 16 列的复制;
    后 64 列为输入转换为复数后的 IFFT.

"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as interp

'''1. Data pre-processing'''

# Read the integer data. Serial data.
original_64 = np.loadtxt("./data_sets/labels64.csv", delimiter=",").astype(np.int).reshape(-1)
# Mapping int to complex. Real 4QAM modulation.
mQAM_list = [1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j]
after_mapping = np.array([mQAM_list[orig] for orig in original_64])
# Serial to parallel
to_ifft = after_mapping.reshape([1000, 64])

'''2. OFDM-modulation (IFFT)'''

# IFFT by row
after_ifft = np.fft.ifft(to_ifft, axis=-1)  # Axis over which to compute the inverse DFT. The last axis(,64) is used.
# Parseval's theorem
assert abs(sum(abs(to_ifft[0])**2)/64 - sum(abs(after_ifft[0])**2)) < 0.0001

'''3. CP(Cyclic Prefix)'''

# Concatenating numpy arrays horizontally
after_cp = np.concatenate((after_ifft[:, -16:], after_ifft), axis=1)
# Save the result
# np.save("./data_sets/after_cp80.npy", after_cp)
# after_cp = np.load("./data_sets/after_cp80.npy")  # load the data set

'''4. Visualization'''

# Choose one of the OFDM symbols (after_cp, using the I/Q modulation)
sample = np.real(after_cp[0])  # real discrete wave to send, 150ms, f_c = 0Hz
x_axis = np.linspace(0, 150, 80)  # 150ms for 80 points
time_axis = np.linspace(0, 150, 80*5)  # 5*2 = 10 times the sampling rate (Negtive Frequency)

# Interpolate
f = interp.interp1d(x_axis, sample, kind='cubic')
after_DAC = f(time_axis)

# Show the wave
plt.xlim((0, 150))
plt.ylim((-0.2, 0.25))
plt.xlabel(r"$Time\ in\ One\ OFDM\ Symbol\ (ms)$")
plt.ylabel(r"$Voltage (V)$")
plt.xticks([0, 30, 60, 90, 120, 150])
plt.title(r"$One\ OFDM\ Symbol$")

plt.vlines(30, -0.2, 0.25, colors="k", linestyles="dashed")
plt.plot(x_axis, sample)
plt.plot(time_axis, after_DAC)

plt.show()

