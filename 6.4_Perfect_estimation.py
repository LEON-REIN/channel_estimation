# @.@ coding  : utf-8 ^-^
# @Author     : Leon Rein
# @Time       : 2021-03-24  ~  20:58 
# @File       : 6.4_Perfect_estimation.py
# @Software   : PyCharm
# @Notice     : It's a WINDOWS version!
#


"""
输入: after_fft64.npy
    1000 行 64 列, complex 矩阵

输出: demodu_Perfect.npy
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
after_fft64 = np.load("./data_sets/after_fft64.npy")  # Received symbols


'''2. Perfect Channel Estimating'''

hn_f = np.fft.fft(hn)  # len = 64

'''3. Visualization hn_LS of the first OFDM symbol'''

# n = np.arange(64)
# plt.figure(1)
# plt.plot(n, abs(hn_f))

'''4. Restore the signal which has been sent'''

xn = after_fft64/hn_f

'''5. 4QAM Demodulating'''

demodu_Perfect = qamdemod(xn)
# np.save("./data_sets/demodu_Perfect.npy", demodu_Perfect)

