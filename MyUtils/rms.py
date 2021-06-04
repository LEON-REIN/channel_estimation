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


if __name__ == '__main__':
    hn = np.array(
        [-1. + 1.22464680e-16j, 0.83357867 - 9.46647260e-01j, 0. + 0.00000000e+00j, 1.02569932 + 5.22276692e-01j,
         0. + 0.00000000e+00j, 0. + 0.00000000e+00j, 0. + 0.00000000e+00j, 0.69663835 + 9.66204296e-01j,
         0. + 0.00000000e+00j, 0. + 0.00000000e+00j, 0. + 0.00000000e+00j, 0.66443826 + 5.86925110e-01j])
    # hn = np.pad(hn, (0, 64 - len(hn)), 'constant', constant_values=(0, 0))
    print(rms(hn))
