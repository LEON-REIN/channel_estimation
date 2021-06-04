# @.@ coding  : utf-8 ^-^
# @Author     : Leon Rein
# @Time       : 2021-03-14  ~  16:00 
# @File       : 5.fft.py
# @Software   : PyCharm
# @Notice     : It's a WINDOWS version!
#               去掉信道传来的每一个 OFDM 符号的前缀, 再进行 OFDM 解调, 用于均衡.

"""
输入: after_channel80_test.npy
    1000 行 80 列, complex 矩阵
    after_channel80_train.npy
    10000 行

输出: after_fft64_test.npy
    1000 行 64 列, complex 矩阵
    after_fft64_train.npy
    10000 行
"""


import numpy as np


'''1. Data pre-processing'''

after_channel = np.load("data_sets/after_channel80_train.npy")  # after_channel80_train.npy
drop_cp = after_channel[:, 16:]  # Drop the leading cp (16 complex numbers)

'''2. OFDM-demodulation (FFT)'''

# FFT by row
after_fft64 = np.fft.fft(drop_cp, axis=-1)  # Axis over which to compute the inverse DFT. The last axis(,64) is used.
# Parseval's theorem
assert abs(sum(abs(drop_cp[0])**2) - sum(abs(after_fft64[0])**2)/64) < 0.0001
# Save the result
# np.save("./data_sets/after_fft64_test.npy", after_fft64)
# after_fft64 = np.load("./data_sets/after_fft64_test.npy")  # load the data set

'''3. Visualization'''

from MyUtils import scatterplot as scp

scp(after_fft64[0:3])

