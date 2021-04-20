# @.@ coding  : utf-8 ^-^
# @Author     : Leon Rein
# @Time       : 2021-03-17  ~  20:16 
# @File       : All_in_One.py
# @Software   : PyCharm
# @Notice     : It's a WINDOWS version!
#               Includes all the steps.

import os  # Need to appear at the top

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # Warnings or Errors ONLY
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses, metrics, models
import tensorflow_addons as tfa  # IMPORTANT for models with tfa.activations.mish!!!
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as interp
from MyUtils import acc, get_dataset, qamdemod, get_onehot, get_int_from_onehot

###################################################################################################
# Basic simulation conditions
# ...
#
#
#
#
#
#
#
#
#
###################################################################################################


"""
*********************************************************************************
                 Preparing commonly used variables & functions
*********************************************************************************
"""

dB_list = [-7, -5, -3, -1, 1, 3, 5, 7, 10, 15, 20, 21, 30, 40, 50]
mQAM_list = np.array([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j])


'''For CENet'''
BATCH_SIZE = 1000


def flat_batch(dataset):
    return dataset.batch(64, drop_remainder=False)


# Custom loss function
class MultiCrossEntropy(losses.Loss):
    def call(self, y_true, y_pred):
        # print(y_true.shape, y_pred.shape)  # (None, 256) (None, 256)
        y_true = tf.split(y_true, 64, axis=-1)  # a list of 64 * vectors
        y_pred = tf.split(y_pred, 64, axis=-1)
        myloss = 0
        for (true, pred) in zip(y_true, y_pred):
            myloss += losses.categorical_crossentropy(true, pred)

        return myloss

    def get_config(self):
        config = super().get_config()
        return config


# Custom metric function for total 64 symbols
def acc_of_all(y_true, y_pred):
    y_true = tf.split(y_true, 64, axis=-1)  # a list of 64 * vectors
    y_pred = tf.split(y_pred, 64, axis=-1)
    num = 0
    for (true, pred) in zip(y_true, y_pred):
        num += tf.reduce_mean(metrics.categorical_accuracy(true, pred))

    return num / 64


# Custom metric function acting on only 48 valid symbols
def acc_of_valid(y_true, y_pred):
    y_true = tf.split(y_true, 64, axis=-1)  # a list of 64 * vectors
    y_pred = tf.split(y_pred, 64, axis=-1)
    num = 0
    for idx, (true, pred) in enumerate(zip(y_true, y_pred)):
        if idx not in [0, 1, 2, 3, 4, 5, 11, 25, 32, 39, 53, 59, 60, 61, 62, 63]:
            num += tf.reduce_mean(metrics.categorical_accuracy(true, pred))

    return num / 48


# 'CENet/V3.6/20210412-131800/CENet-V3.6.h5'
model = models.load_model('CENet/V3.6/20210420-085719/CENet-V3.6.h5',
                          custom_objects={
                              'MultiCrossEntropy': MultiCrossEntropy,
                              'acc_of_all': acc_of_all,
                              'acc_of_valid': acc_of_valid,
                          })

"""
*********************************************************************************
                        Main Loop Here!
*********************************************************************************
"""

Pe_array, BER_array = [], []

# dB_list = [10]  # Just for test
for SNR in dB_list:
    # Get Simulation Data
    data_64, labels_int_64 = get_dataset(num=1000, SNR=SNR)
    Pe_list, BER_list = [], []

    '''1. CENet'''
    data_to_test = tf.data.Dataset.from_tensor_slices(
        tf.constant(data_64.view(np.float).reshape(-1, 2), dtype=tf.float32),
    )

    data_to_test = data_to_test.window(64).flat_map(flat_batch)

    labels_onehot = tf.data.Dataset.from_tensor_slices(
        tf.constant(get_onehot(labels_int_64), dtype=tf.float32)
    )
    to_test = tf.data.Dataset.zip((data_to_test, labels_onehot)) \
        .batch(BATCH_SIZE) \
        .prefetch(tf.data.experimental.AUTOTUNE) \
        .cache()
    demodu = get_int_from_onehot(model.predict(to_test))

    Pe, BER = acc.get_evaluation(demodu, labels_int_64)
    Pe_list.append(Pe)
    BER_list.append(BER)

    '''2. LS'''
    pilot_idx = np.array([11, 25, 39, 53])
    xn_pilot = np.full(pilot_idx.shape, mQAM_list[3])  # pilot series
    n = np.arange(64)
    hn_LS = np.ones_like(data_64)
    hn_pilot_LS = data_64[:, pilot_idx] / xn_pilot  # shape = (1000, 4)

    for i in np.arange(data_64.shape[0]):
        f = interp.interp1d(pilot_idx, hn_pilot_LS[i], kind='linear', fill_value="extrapolate")
        hn_LS[i] = f(n)

    xn = data_64 / hn_LS
    demodu = qamdemod(xn)

    Pe, BER = acc.get_evaluation(demodu, labels_int_64)
    Pe_list.append(Pe)
    BER_list.append(BER)

    '''3. MMSE'''
    # hn is from 4.pass_through_channel.py
    hn = np.array(
        [-1. + 1.22464680e-16j, 0.83357867 - 9.46647260e-01j, 0. + 0.00000000e+00j, 1.02569932 + 5.22276692e-01j,
                   0. + 0.00000000e+00j, 0. + 0.00000000e+00j, 0. + 0.00000000e+00j, 0.69663835 + 9.66204296e-01j,
                   0. + 0.00000000e+00j, 0. + 0.00000000e+00j, 0. + 0.00000000e+00j, 0.66443826 + 5.86925110e-01j])
    hn = np.pad(hn, (0, 64 - len(hn)), 'constant', constant_values=(0, 0))

    # Frequency domain correlation function
    hn_f = np.fft.fft(hn)  # len = 64
    rf_list = np.correlate(hn_f, hn_f, mode='full')  # len = 64*2 - 1 = 127
    idx_offset = int((len(rf_list) - 1) / 2)
    rf_list = rf_list / rf_list[idx_offset]

    idx_Rhp = np.arange(64).reshape(64, 1) - pilot_idx.reshape(1, -1)  # shape==(64, 4)
    idx_Rpp = pilot_idx.reshape(-1, 1) - pilot_idx.reshape(1, -1)  # shape==(4, 4)


    def get_rf(index):
        index = index + idx_offset
        return rf_list[index]


    get_rf = np.vectorize(get_rf)

    Rhp = get_rf(idx_Rhp)  # shape==(64, 4)
    Rpp = get_rf(idx_Rpp) + np.eye(len(pilot_idx)) / (10 ** (SNR * 0.1))  # shape==(4, 4)

    hn_mmse = np.ones_like(data_64)
    for i in np.arange(data_64.shape[0]):
        hn_mmse[i] = Rhp @ np.linalg.inv(Rpp) @ hn_pilot_LS[i]

    xn = data_64 / hn_mmse
    demodu = qamdemod(xn)

    Pe, BER = acc.get_evaluation(demodu, labels_int_64)
    Pe_list.append(Pe)
    BER_list.append(BER)

    '''4. Perfect Equalization'''
    xn = data_64/hn_f
    demodu = qamdemod(xn)

    Pe, BER = acc.get_evaluation(demodu, labels_int_64)
    Pe_list.append(Pe)
    BER_list.append(BER)

    '''5. No Equalization'''
    demodu = qamdemod(data_64)

    Pe, BER = acc.get_evaluation(demodu, labels_int_64)
    Pe_list.append(Pe)
    BER_list.append(BER)

    '''Aggregating data'''
    # print(f"Pe_list = {Pe_list}, \nBER_list = {BER_list}")
    Pe_array.append(Pe_list)
    BER_array.append(BER_list)

# Each row represents the performance (Pe or BER) of an equalization method as the SNR varies.
Pe_array = np.array(Pe_array).transpose()
BER_array = np.array(BER_array).transpose()

"""
*********************************************************************************
                               Visualization!
*********************************************************************************
"""


with plt.style.context(['ieee', 'grid']):
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.axes(yscale="log")
    plt.plot(dB_list, Pe_array[0], 'k*-', label="CENet")
    plt.plot(dB_list, Pe_array[1], label="LS")
    plt.plot(dB_list, Pe_array[2], label="MMSE")
    plt.plot(dB_list, Pe_array[3], label="Perfect")
    plt.plot(dB_list, Pe_array[4], label="No EQ")
    plt.autoscale(tight=True)
    plt.legend(edgecolor='k')
    plt.title("Pe  Performance")
    plt.ylabel("Pe")
    plt.ylim([10**(-2.6), 1.2])
    plt.xlabel("SNR [dB]")
    plt.gcf().subplots_adjust(left=0.15, bottom=0.15)  # Expanded display area
    plt.show()


with plt.style.context(['ieee', 'grid']):
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.axes(yscale="log")
    plt.plot(dB_list, BER_array[0], 'k*-', label="CENet")
    plt.plot(dB_list, BER_array[1], label="LS")
    plt.plot(dB_list, BER_array[2], label="MMSE")
    plt.plot(dB_list, BER_array[3], label="Perfect")
    plt.plot(dB_list, BER_array[4], label="No EQ")
    plt.autoscale(tight=True)
    plt.legend(edgecolor='k')
    plt.title("BER Performance")
    plt.ylabel("BER")
    plt.ylim([10**(-2.6), 1.2])
    plt.xlabel("SNR [dB]")
    plt.gcf().subplots_adjust(left=0.15, bottom=0.15)  # Expanded display area
    plt.show()
