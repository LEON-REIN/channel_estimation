# @.@ coding  : utf-8 ^-^
# @Author     : Leon Rein
# @Time       : 2021-03-14  ~  16:55 
# @File       : 6.1_CENet_estimation.py
# @Software   : PyCharm
# @Notice     : It's a WINDOWS version!
#


import os  # Need to appear at the top
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # Warnings or Errors ONLY
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses, metrics, models
import tensorflow_addons as tfa  # IMPORTANT for models with tfa.activations.mish!!!
import numpy as np


BATCH_SIZE = 1000


# 1000 by 64
after_fft64_test = np.load("data_sets/after_fft64_test.npy")
# 1000 by 256
labels_to_test = np.load("data_sets/labels64_onehot_test.npy")


def flat_batch(dataset):
    return dataset.batch(64, drop_remainder=False)


'''2.2 Preparing Test Data'''
# (64000, 2)
data_to_test = tf.data.Dataset.from_tensor_slices(
    tf.constant(after_fft64_test.view(np.float).reshape(-1, 2), dtype=tf.float32),
)

# (64000, 2) -> (1000, 64, 2)
data_to_test = data_to_test.window(64).flat_map(flat_batch)
# (1000, 256)
labels_to_test = tf.data.Dataset.from_tensor_slices(
    tf.constant(labels_to_test, dtype=tf.float32)
)
to_test = tf.data.Dataset.zip((data_to_test, labels_to_test)) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.experimental.AUTOTUNE) \
    .cache()  # cache the dataset into RAM


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

    return num/64


# Custom metric function acting on only 48 valid symbols
def acc_of_valid(y_true, y_pred):
    y_true = tf.split(y_true, 64, axis=-1)  # a list of 64 * vectors
    y_pred = tf.split(y_pred, 64, axis=-1)
    num = 0
    for idx, (true, pred) in enumerate(zip(y_true, y_pred)):
        if idx not in [0, 1, 2, 3, 4, 5, 11, 25, 32, 39, 53, 59, 60, 61, 62, 63]:
            num += tf.reduce_mean(metrics.categorical_accuracy(true, pred))

    return num/48


# 'CENet/V1.2/20210405-201842/CENet-V1.2.h5'
# Pe = 0.5866458333333333, BER = 0.3942916666666667
# 'CENet/V2.6/20210405-135003/CENet-V2.6.h5'
# Pe = 0.43756249999999997, BER = 0.293625
# 'CENet/V3.2/20210412-210848/CENet-V3.2.h5'
# Pe = 0.17218750000000005, BER = 0.12060416666666662
# 'CENet/V3.6/20210412-131800/CENet-V3.6.h5'
# Pe = 0.09387500000000004, BER = 0.06678125000000001
model = models.load_model('CENet/V3.6/20210412-131800/CENet-V3.6.h5',
                          custom_objects={
                              'MultiCrossEntropy': MultiCrossEntropy,
                              'acc_of_all': acc_of_all,
                              'acc_of_valid': acc_of_valid,
                           })

aa = model.predict(to_test)
bb = aa.reshape(-1, 4)
cc = np.argmax(bb, axis=1).reshape(-1, 64).astype(np.int)  # onehot to 0~3
np.save("./data_sets/demodu_CENet.npy", cc)

