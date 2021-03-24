# @.@ coding  : utf-8 ^-^
# @Author     : Leon Rein
# @Time       : 2021-03-24  ~  19:18 
# @File       : rms.py
# @Software   : PyCharm
# @Notice     : It's a WINDOWS version!
#

import numpy as np


# Calculating the RMS Time Lag
def rms(real_h: np.array, time_interval=150/80) -> np.float:
    """Usage: rms(hn_LS)"""
    powers = np.abs(real_h * real_h.conj())
    time_idx = np.arange(len(real_h)) * time_interval

    tau_2 = np.sum(powers * time_idx.__pow__(2)) / np.sum(powers)
    tau = np.sum(powers * time_idx) / np.sum(powers)

    return np.sqrt(tau_2 - tau.__pow__(2))
