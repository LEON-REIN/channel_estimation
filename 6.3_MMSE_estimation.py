# @.@ coding  : utf-8 ^-^
# @Author     : Leon Rein
# @Time       : 2021-03-14  ~  16:56 
# @File       : 6.3_MMSE_estimation.py
# @Software   : PyCharm
# @Notice     : It's a WINDOWS version!
#               对 FFT 解调后的数据进行信道估计并均衡, 使用 MMSE 方法

"""
输入: after_fft64_test.npy
    1000 行 64 列, complex 矩阵

输出: demodu_LS.npy
    1000 行 64 列, int 矩阵
"""

import numpy as np
from MyUtils import qamdemod
import matplotlib.pyplot as plt


'''1. Data pre-processing'''

hn = np.array([-1. + 1.22464680e-16j, 0.83357867 - 9.46647260e-01j, 0. + 0.00000000e+00j, 1.02569932 + 5.22276692e-01j,
               0. + 0.00000000e+00j, 0. + 0.00000000e+00j, 0. + 0.00000000e+00j, 0.69663835 + 9.66204296e-01j,
               0. + 0.00000000e+00j, 0. + 0.00000000e+00j, 0. + 0.00000000e+00j, 0.66443826 + 5.86925110e-01j])
hn = np.pad(hn, (0, 64 - len(hn)), 'constant', constant_values=(0, 0))
after_fft_64 = np.load("data_sets/after_fft64_test.npy")  # Received symbols. after_fft64_train.npy
mQAM_list = np.array([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j])
pilot_idx = np.array([11, 25, 39, 53])  # index-32 is symbol '0' -> 1+0j
xn_pilot = np.full(pilot_idx.shape, mQAM_list[3])  # pilot series

SNR = 10  # dB

'''2. LS Channel Estimating'''

hn_pilot_LS = after_fft_64[:, pilot_idx] / xn_pilot  # shape = (1000, 4)

'''3. MMSE Channel Estimating'''

# Frequency domain correlation function
hn_f = np.fft.fft(hn)  # len = 64
rf_list = np.correlate(hn_f, hn_f, mode='full')  # len = 64*2 - 1 = 127
idx_offset = int((len(rf_list) - 1) / 2)
rf_list = rf_list / rf_list[idx_offset]

idx_Rhp = np.arange(64).reshape(64, 1) - pilot_idx.reshape(1, -1)  # shape==(64, 4)
idx_Rpp = pilot_idx.reshape(-1, 1) - pilot_idx.reshape(1, -1)  # shape==(4, 4)


def get_rf(index):
    index = index + idx_offset
    return rf_list[index]


get_rf = np.vectorize(get_rf)

Rhp = get_rf(idx_Rhp)  # shape==(64, 4)
Rpp = get_rf(idx_Rpp) + np.eye(len(pilot_idx)) / (10 ** (SNR * 0.1))  # shape==(4, 4)

hn_mmse = np.ones_like(after_fft_64)
for i in np.arange(after_fft_64.shape[0]):
    hn_mmse[i] = Rhp @ np.linalg.inv(Rpp) @ hn_pilot_LS[i]

'''3. Visualization hn_LS of the first OFDM symbol'''

# TODO: subplot
# n = np.arange(64)
# plt.figure(1)
# plt.scatter(pilot_idx, abs(hn_pilot_LS[0]))
# plt.plot(n, abs(hn_mmse[0]))
#
# plt.figure(2)
# plt.scatter(pilot_idx, hn_pilot_LS[0].real)
# plt.plot(n, hn_mmse[0].real)
#
# plt.figure(3)
# plt.scatter(pilot_idx, hn_pilot_LS[0].imag)
# plt.plot(n, hn_mmse[0].imag)
# plt.show()


'''4. Restore the signal which has been sent'''

xn = after_fft_64 / hn_mmse

'''5. 4QAM Demodulating'''

demodu_MMSE = qamdemod(xn)
# np.save("./data_sets/demodu_MMSE.npy", demodu_MMSE)  # save the data
# demodu_MMSE = np.load("./data_sets/demodu_MMSE.npy")  # read the data

