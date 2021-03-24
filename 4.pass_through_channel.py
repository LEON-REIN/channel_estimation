# @.@ coding  : utf-8 ^-^
# @Author     : Leon Rein
# @Time       : 2021-03-10  ~  10:37 
# @File       : 4.pass_through_channel.py
# @Software   : PyCharm
# @Notice     : It's a WINDOWS version!
#               经过多径信道并加入高斯白噪声

"""
输入: after_cp80.npy
    进入信道的数据
    norm_amp.mat, norm_delay_ms.mat
    归一化复数信道冲激响应的幅度和时延

输出: after_channel80.npy
    1000 行 64 列, complex 矩阵

"""


import numpy as np
import h5py


'''1. Data pre-processing'''

# Read as serial symbols
after_cp = np.load("./data_sets/after_cp80.npy").reshape(-1)
norm_amp = h5py.File("./data_sets/norm_amp.mat", 'r')['normAmp'][:]
norm_delay_ms = h5py.File("./data_sets/norm_delay_ms.mat", 'r')['normDelay_ms'][:]
# Change their dtypes
norm_delay_ms = norm_delay_ms.reshape(-1).astype(np.float32)
norm_amp.dtype = np.complex
norm_amp = norm_amp.reshape(-1)


'''2. Extraction of discrete channel'''

# norm_delay_ms: [ 0.,  1.64204,  1.6757 ,  5.8748 ,  5.97965, 12.4431 , 12.45593, 21.1346 , 21.19814] ms
x_axis = np.linspace(0, 150, 80+1)  # 150ms for 80 intervals
idx = []
for ms in norm_delay_ms:
    idx.append(np.abs(x_axis-ms).argmin())  # To find where is the nearset num compared to the 'ms'.

# Discrete channel, with time-axis normalized
hn = np.zeros((idx[-1]+1)).astype(np.complex)
for index, h_idx in enumerate(idx):
    hn[h_idx] += norm_amp[index]


'''3. Through Bellhop channel with awgn'''

# Discrete linear convolution
after_bellhop_0 = np.convolve(after_cp, hn, mode='full')
after_bellhop = after_bellhop_0[:80000].reshape(1000, 80)

# AWGN
np.random.seed(1)
SNR = 10  # dB
_snr = 10 ** (SNR/10.0)  # how many times
signal_power = np.sum(np.abs(after_bellhop) ** 2, axis=1) / 80
n_power = signal_power / _snr
noise = np.random.randn(1000, 80, 2).view(np.complex).reshape(1000, 80)
noise_n = noise / np.sqrt(np.sum(np.abs(noise) ** 2, axis=1) / 80).reshape(1000, 1)  # Normalization
# print(np.sum(np.abs(noise_n) ** 2, axis=1) / 80)  # To show the power of each row
after_channel = after_bellhop + noise_n * np.sqrt(n_power.reshape((1000, 1)))

'''4. Save to file'''

# np.save("./data_sets/after_channel80.npy", after_channel)
# after_channel = np.load("./data_sets/after_channel80.npy")  # load the data set

'''5. Visualization'''

# TODO: 可视化












