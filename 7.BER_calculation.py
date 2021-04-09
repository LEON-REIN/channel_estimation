# @.@ coding  : utf-8 ^-^
# @Author     : Leon Rein
# @Time       : 2021-03-14  ~  16:58 
# @File       : 7.BER_calculation.py
# @Software   : PyCharm
# @Notice     : It's a WINDOWS version!
#               将信道估计并均衡后的码元和原始码元进行匹配, 计算误码率等指标

"""
输入:
正确的四进制码元(0~3) labels48_test.csv, 1000 行 48 列;
信道估计后并解调的码元(0~3) demodu_*.npy, 1000 行 64 列

输出: 无
"""


import numpy as np
from MyUtils import acc

'''1. Data pre-processing'''

original_48 = np.loadtxt("./data_sets/labels48_test.csv", delimiter=",").astype(np.int)
# original_48 = np.load("./data_sets/labels48_train.npy")

# demodu64 = np.load("./data_sets/test.npy")
demodu64 = np.load("./data_sets/demodu_CENet.npy")  # CENet
# demodu64 = np.load("./data_sets/demodu_LS.npy")  # LS
# demodu64 = np.load("./data_sets/demodu_MMSE.npy")  # MMSE
# demodu64 = np.load("./data_sets/demodu_Perfect.npy")  # Perfect

demodu48 = np.concatenate((demodu64[:, 6:11], demodu64[:, 12:25],
                           demodu64[:, 26:32], demodu64[:, 33:39],
                           demodu64[:, 40:53], demodu64[:, 54:59]), axis=1)

'''2. BER when without channel equalization'''

Pe_no_eq = 0.9310  # from qamdemod.py
BER_no_eq = 0.586875  # from qamdemod.py


'''3. BER & Pe calculation'''

assert demodu48.dtype in [np.int64, np.int32], "dtype is not int!"
# Error rate of symbols
Pe = acc.get_Pe(demodu48, original_48)
# Error rate of bits,and inputs should be np.array which is formed by 0~3
BER = acc.get_BER(demodu48, original_48)

print(f"Pe = {Pe}, BER = {BER}")

