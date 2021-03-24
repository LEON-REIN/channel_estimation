# @.@ coding  : utf-8 ^-^
# @Author     : Leon Rein
# @Time       : 2021-03-14  ~  16:56 
# @File       : 6.2_LS_estimation.py
# @Software   : PyCharm
# @Notice     : It's a WINDOWS version!
#               对 FFT 解调后的数据进行信道估计并均衡, 使用 LS 方法并线性插值

"""
输入: after_fft64.npy
    1000 行 64 列, complex 矩阵

输出: demodu_LS.npy
    1000 行 64 列, int 矩阵
"""

from MyUtils import qamdemod
import numpy as np
from scipy import interpolate as interp
import matplotlib.pyplot as plt

'''1. Data pre-processing'''

after_fft64 = np.load("./data_sets/after_fft64.npy")
mQAM_list = np.array([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j])
pilot_idx = np.array([11, 25, 39, 53])  # index-32 is symbol '0' -> 1+0j
xn_pilot = np.full(pilot_idx.shape, mQAM_list[3])  # pilot series

'''2. Channel Estimating'''

n = np.arange(64)
hn_LS = np.ones_like(after_fft64)
hn_pilot = after_fft64[:, pilot_idx] / xn_pilot  # shape = (1000, 4)
for i in np.arange(after_fft64.shape[0]):
    f = interp.interp1d(pilot_idx, hn_pilot[i], kind='linear', fill_value="extrapolate")
    hn_LS[i] = f(n)

'''3. Visualization hn_LS of the first OFDM symbol'''

# TODO: subplot
plt.figure(1)
plt.scatter(pilot_idx, abs(hn_pilot[0]))
plt.plot(n, abs(hn_LS[0]))

plt.figure(2)
plt.scatter(pilot_idx, hn_pilot[0].real)
plt.plot(n, hn_LS[0].real)

plt.figure(3)
plt.scatter(pilot_idx, hn_pilot[0].imag)
plt.plot(n, hn_LS[0].imag)
plt.show()


'''4. Restore the signal which has been sent'''

xn = after_fft64 / hn_LS

'''5. 4QAM Demodulating'''

demodu_LS = qamdemod(xn)
# np.save("./data_sets/demodu_LS.npy", demodu_LS)  # save the data
# demodu_LS = np.load("./data_sets/demodu_LS.npy")  # read the data


# TODO: 直接除以原 h 的 fft
