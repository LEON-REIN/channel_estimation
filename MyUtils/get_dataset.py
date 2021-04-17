# @.@ coding  : utf-8 ^-^
# @Author     : Leon Rein
# @Time       : 2021-04-12  ~  22:10 
# @File       : get_dataset.py
# @Software   : PyCharm
# @Notice     : It's a WINDOWS version!
#               Genrating OFDM symbols & their correct labels by numpy when the SNR specified.

import numpy as np


def get_dataset(num=1000, SNR=10):
    """
    A Dataset Generator with numpy.
    :param num: The number of symbols to generate.
    :param SNR: dB
    :return: Two datasets. labels64 means original symbols of [0~3], #num by 64 np.int matrix.
                           after_fft64 means a #num by 64 complex matrix.
    """

    '''1. Baseband signal'''
    original_48 = np.random.randint(low=0, high=4, size=(num, 48), dtype=np.int)  # 0, 1, 2, 3

    '''2. Inserting the frequency guide and some zeros'''
    original_64 = np.zeros(shape=(num, 64), dtype=np.int)
    original_64[:, 6:11] = original_48[:, 0:5]
    original_64[:, 11] = 3
    original_64[:, 12:25] = original_48[:, 5:18]
    original_64[:, 25] = 3
    original_64[:, 26:32] = original_48[:, 18:24]
    original_64[:, 33:39] = original_48[:, 24:30]
    original_64[:, 39] = 3
    original_64[:, 40:53] = original_48[:, 30:43]
    original_64[:, 53] = 3
    original_64[:, 54:59] = original_48[:, 43:48]

    '''3. Int to complex (mQAM modulation)'''
    mQAM_list = [1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j]
    after_mapping = np.array([mQAM_list[orig] for orig in original_64.reshape(-1)])
    # Serial to parallel
    to_ifft = after_mapping.reshape([-1, 64])

    '''4. OFDM-modulation (IFFT)'''
    # IFFT by row
    after_ifft = np.fft.ifft(to_ifft, axis=-1)  # #num by 64, complex
    # Parseval's theorem, been checked by the 0th symbol.
    # assert abs(sum(abs(to_ifft[0])**2)/64 - sum(abs(after_ifft[0])**2)) < 0.0001

    '''5. CP(Cyclic Prefix)'''
    after_cp = np.concatenate((after_ifft[:, -16:], after_ifft), axis=1)  # #num by 80

    '''6. Pass through channel'''
    after_cp = after_cp.reshape(-1)
    hn = np.array(
        [-1. + 1.22464680e-16j, 0.83357867 - 9.46647260e-01j, 0. + 0.00000000e+00j, 1.02569932 + 5.22276692e-01j,
                   0. + 0.00000000e+00j, 0. + 0.00000000e+00j, 0. + 0.00000000e+00j, 0.69663835 + 9.66204296e-01j,
                   0. + 0.00000000e+00j, 0. + 0.00000000e+00j, 0. + 0.00000000e+00j, 0.66443826 + 5.86925110e-01j])

    # Discrete linear convolution
    after_bellhop_0 = np.convolve(after_cp, hn, mode='full')
    after_bellhop = after_bellhop_0[:after_cp.shape[0]].reshape(-1, 80)

    # AWGN
    # np.random.seed(1)  # Test
    _snr = 10 ** (SNR/10.0)  # dB to how many times
    signal_power = np.sum(np.abs(after_bellhop) ** 2, axis=1) / 80
    n_power = signal_power / _snr
    noise = np.random.randn(after_bellhop.shape[0], after_bellhop.shape[1], 2) \
        .view(np.complex).reshape(after_bellhop.shape)
    noise_n = noise / np.sqrt(np.sum(np.abs(noise) ** 2, axis=1) / 80) \
        .reshape(after_bellhop.shape[0], 1)  # Normalization
    # print(np.sum(np.abs(noise_n) ** 2, axis=1) / 80)  # To show the power of each row
    after_channel = after_bellhop + noise_n * np.sqrt(n_power.reshape((after_bellhop.shape[0], 1)))

    '''7. OFDM-demodulation (FFT)'''
    drop_cp = after_channel[:, 16:]  # Drop the leading cp (16 complex numbers)
    # FFT by row
    after_fft64 = np.fft.fft(drop_cp, axis=-1)
    # Parseval's theorem
    # assert abs(sum(abs(drop_cp[0])**2) - sum(abs(after_fft64[0])**2)/64) < 0.0001

    return after_fft64, original_64


def get_onehot(labels_int, num_classes=4):
    """
    Map a int matrix to a onehot matrix. The shape[0] won't change.
    :param labels_int: N by M
    :param num_classes: int number of classes
    :return: N by M*num_classes
    """
    def one_hot_mapping(x):
        mapping_list = np.eye(num_classes)
        return mapping_list[x]

    onehot = np.array([one_hot_mapping(i) for i in labels_int])

    return onehot.reshape(labels_int.shape[0], -1)


def get_valid_data(labels_64):
    """
    Extraction of valid data, from 64 columns to 48 columns.
    :param labels_64: an integer matrix with 64 columns.
    :return: an integer matrix with 48 columns.
    """
    return np.concatenate((labels_64[:, 6:11], labels_64[:, 12:25],
                           labels_64[:, 26:32], labels_64[:, 33:39],
                           labels_64[:, 40:53], labels_64[:, 54:59]), axis=1)


def get_int_from_onehot(labels_onehot_, num_classes=4):
    """
    Restore the onehot matrix to an integer matrix.
    :param labels_onehot_: a onehot matrix
    :param num_classes: number of classes
    :return: an integer matrix
    """
    temp = labels_onehot_.reshape(-1, num_classes)
    return np.argmax(temp, axis=1).reshape(-1, 64).astype(np.int)  # onehot to 0~3


if __name__ == '__main__':

    labels64 = np.load("../data_sets/labels64_train.npy")
    labels48 = np.load("../data_sets/labels48_train.npy")
    ans = np.load("../data_sets/labels64_onehot_train.npy")
    assert np.sum(ans - get_onehot(labels64)) < 0.00001
    assert np.sum(labels48 - get_valid_data(labels64)) < 0.00001
    print("Pass the test!")

    """You can generate dataset for DNN like this!"""
    data_64, labels_int_64 = get_dataset(num=15000, SNR=-8)
    labels_onehot = get_onehot(labels_int_64, 4)
    dB_list = [-7, -6, -5, -4, -3, -1, 0, 1, 2, 3, 4, 5, 7,  9, 10, 11, 14, 20, 30, np.Inf]
    for SNR_ in dB_list:
        new_data_64, new_labels_int_64 = get_dataset(num=10000, SNR=SNR_)
        new_labels_onehot = get_onehot(new_labels_int_64, 4)
        data_64 = np.concatenate((data_64, new_data_64), axis=0)
        labels_onehot = np.concatenate((labels_onehot, new_labels_onehot), axis=0)

    # np.save(f'D:\move\Desktop\data_64.npy', data_64)
    # np.save(f'D:\move\Desktop\labels_onehot.npy', labels_onehot)
