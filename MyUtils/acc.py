# @.@ coding  : utf-8 ^-^
# @Author     : Leon Rein
# @Time       : 2021-03-16  ~  20:19 
# @File       : acc.py
# @Software   : PyCharm
# @Notice     : It's a WINDOWS version!
#               Calculates the Pe & BER of series

import numpy as np
from .get_dataset import get_valid_data


# Error rate of symbols
def get_Pe(demodu: np.ndarray, ans: np.ndarray) -> np.float64:
    error_symbol_num = sum(demodu.reshape(-1) == ans.reshape(-1))
    return 1 - error_symbol_num / len(demodu.reshape(-1))


# Error rate of bits,and inputs should be np.array which is formed by 0~3
def get_BER(demodu: np.ndarray, ans: np.ndarray) -> np.float64:
    demod_list = [[0, 0], [0, 1], [1, 0], [1, 1]]
    demodu_bits = np.array([demod_list[idx] for idx in demodu.reshape(-1)]).reshape(-1)
    ans_bits = np.array([demod_list[idx] for idx in ans.reshape(-1)]).reshape(-1)
    return get_Pe(demodu_bits, ans_bits)


# Get the Pe & BER of two matrices which both have 64 columns.
def get_evaluation(demodu_64, labels64):
    demodu_48 = get_valid_data(demodu_64)
    labels_int_48 = get_valid_data(labels64)
    assert demodu_48.dtype in [np.int64, np.int32], "dtype is not int!"
    # Error rate of symbols
    Pe = get_Pe(demodu_48, labels_int_48)
    # Error rate of bits,and inputs should be np.array which is formed by 0~3
    BER = get_BER(demodu_48, labels_int_48)
    return Pe, BER
