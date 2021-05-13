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

# demodu_64 = np.load("./data_sets/test.npy")
demodu_64 = np.load("./data_sets/demodu_CENet.npy")  # CENet
# demodu_64 = np.load("./data_sets/demodu_LS.npy")  # LS
# demodu_64 = np.load("./data_sets/demodu_MMSE.npy")  # MMSE
# demodu_64 = np.load("./data_sets/demodu_Perfect.npy")  # Perfect

demodu48 = np.concatenate((demodu_64[:, 6:11], demodu_64[:, 12:25],
                           demodu_64[:, 26:32], demodu_64[:, 33:39],
                           demodu_64[:, 40:53], demodu_64[:, 54:59]), axis=1)

'''2. BER & Pe calculation'''

assert demodu48.dtype in [np.int64, np.int32], "dtype is not int!"
# Error rate of symbols
Pe = acc.get_Pe(demodu48, original_48)
# Error rate of bits,and inputs should be np.array which is formed by 0~3
BER = acc.get_BER(demodu48, original_48)

print(f"Pe = {Pe}, BER = {BER}")

'''3. BER&Pe logs (1000symbols, SNR=10dB!!!)

- BER when without channel equalization (from qamdemod.py)
Pe_no_eq = 0.9310;  BER_no_eq = 0.586875 

- Perfect equalization
Pe = 0.08562499999999995, BER = 0.06120833333333331

- LS
Pe = 0.39054;  BER = 0.27606

- MMSE
--- demo
    Pe = 0.37087;  BER = 0.26128

- CENet 
--- V1.2
    Pe = 0.5866458333333333, BER = 0.3942916666666667
--- V2.6
    Pe = 0.43756249999999997, BER = 0.293625
--- V3.2 
    Pe = 0.17218750000000005, BER = 0.12060416666666662
--- V3.6
    Pe = 0.0950833333333333, BER = 0.06799999999999995    

'''