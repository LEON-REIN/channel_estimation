# @.@ coding  : utf-8 ^-^
# @Author     : Leon Rein
# @Time       : 2021-03-14  ~  16:59 
# @File       : CENet.py
# @Software   : PyCharm
# @Notice     : It's a WINDOWS version!
#               信道估计神经网络的线下训练, 可供线上使用时 fine-tune


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # Warnings or Errors ONLY
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses, metrics
import numpy as np
import pandas as pd

'''
1. Load datasets
'''

# 10000 rows by 64 columns. Each element is a complex number.
after_fft64_train = np.load("data_sets/after_fft64_train.npy")  # Received symbols
# 1000 by 64
after_fft64_test = np.load("data_sets/after_fft64_test.npy")

# 10000 rows by 64*4=256 columns. Onehot coded labels.
labels_to_train = np.load("data_sets/labels64_onehot_train.npy")
# 1000 by 256
labels_to_test = np.load("data_sets/labels64_onehot_test.npy")

'''
2. Data Processing
'''

#
data_to_train = tf.data.Dataset.from_tensor_slices((
    tf.constant(after_fft64_train.real.reshape(-1), dtype=tf.float32),
    tf.constant(after_fft64_train.imag.reshape(-1), dtype=tf.float32)
))
