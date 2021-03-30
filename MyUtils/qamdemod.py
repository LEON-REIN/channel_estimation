# @.@ coding  : utf-8 ^-^
# @Author     : Leon Rein
# @Time       : 2021-03-16  ~  18:29 
# @File       : qamdemod.py
# @Software   : PyCharm
# @Notice     : It's a WINDOWS version!
#

"""
This module is intended for demoulating complex arrays into decimal symbols.
"""

import numpy as np


def qamdemod(array: np.ndarray, maplist=None, init_phase=0) -> np.ndarray:
    """
    Demodulate a array of complex numbers into decimal numbers.
    Base on minimum distance criterion

    :param array: np.ndarray, dtype == np.complex
        The input array which is to be demodulated
    :param maplist: list or np.ndarray, dtype == np.complex
        The mapping list
    :param init_phase: a float number
        To correct the initial phase of the array
    :return: np.ndarray, dtype == np.int
        Demodulated result
    """
    if maplist is None:
        maplist = [1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j]
    maplist = np.array(maplist) * np.exp(+2 * np.pi * 1j * init_phase)  # To correct the phase
    # array = np.array(array) * np.exp(-2 * np.pi * 1j * init_phase)  # Equivalent to the above line

    # vectorized map function
    vfunc = np.vectorize(lambda x:
                         np.argmin(np.abs(x - maplist))  # x will be broadcast automatically
                         )

    demod = vfunc(array)
    demod = demod.astype(np.int)

    return demod


if __name__ == '__main__':
    after_fft64 = np.load("../data_sets/after_fft64_test.npy")  # Without channel equalization
    # after_fft64 = np.load("../data_sets/after_fft64_train.npy")  # Without channel equalization
    original_48 = np.loadtxt("../data_sets/labels48_test.csv", delimiter=",").astype(np.int)
    # original_48 = np.load("../data_sets/labels48_train.npy")

    demodu64 = qamdemod(after_fft64)
    demodu48 = np.concatenate((demodu64[:, 6:11], demodu64[:, 12:25],
                               demodu64[:, 26:32], demodu64[:, 33:39],
                               demodu64[:, 40:53], demodu64[:, 54:59]), axis=1)

    import acc

    Pe = acc.get_Pe(demodu48, original_48)
    BER = acc.get_BER(demodu48, original_48)

    print(f"With no equalizations, \nPe = {Pe}, BER = {BER}")
