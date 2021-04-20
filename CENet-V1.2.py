# @.@ coding  : utf-8 ^-^
# @Author     : Leon Rein
# @Time       : 2021-03-14  ~  16:59 
# @File       : CENet-V1.2.py
# @Software   : PyCharm
# @Notice     : It's a WINDOWS version!
#               信道估计神经网络的线下训练, 可供线上使用时 fine-tune
#               初版架构, 准确率上限为 40% 左右


import os  # Need to appear at the top
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # Warnings or Errors ONLY
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import datetime
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses, metrics
import numpy as np


# CUDA config
tf.config.experimental.set_memory_growth(
    tf.config.experimental.list_physical_devices(device_type='GPU')[0], True
)

# logdir
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Time stamp
logdir = os.path.join('CENet', __file__[-3-4:-3], stamp)  # path to log: ./CENet/version/stamp

'''
0. Hyperparameters
'''

BATCH_SIZE = 128

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
2. Data Processing & Pipeline
'''


def flat_batch(dataset):
    return dataset.batch(64, drop_remainder=False)


'''2.1 Preparing Training Data'''
# shape = (640000, 2)
data_to_train = tf.data.Dataset.from_tensor_slices(
    tf.constant(after_fft64_train.view(np.float).reshape(-1, 2), dtype=tf.float32)
)

# (640000, 2) -> (10000, 64, 2)
data_to_train = data_to_train.window(64).flat_map(flat_batch)
# (10000, 256)
labels_to_train = tf.data.Dataset.from_tensor_slices(
    tf.constant(labels_to_train, dtype=tf.float32)
)
to_train = tf.data.Dataset.zip((data_to_train, labels_to_train)) \
    .batch(BATCH_SIZE) \
    .prefetch(tf.data.experimental.AUTOTUNE) \
    .cache()  # cache the dataset into RAM

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

'''
3. Keras Modeling (Functional API)
'''

inputs = layers.Input(shape=(64, 2))  # time_steps = 64, channel_num/feature_num = 2
x = layers.LayerNormalization(axis=-2)(inputs)  # out: (, 64, 2); acts on 64
x = layers.LSTM(8, return_sequences=True)(x)
x = layers.LSTM(8, return_sequences=True)(x)
x = layers.LayerNormalization(axis=-1)(x)
x = layers.Conv1D(8, kernel_size=3, activation='relu', padding='same')(x)
x = layers.AveragePooling1D(pool_size=2, strides=None, padding='same')(x)
x = layers.Conv1D(1, kernel_size=3, activation='relu', padding='same')(x)
x = layers.Flatten()(x)  # Or, tf.squeeze
x = layers.Dense(16, activation='relu')(x)
x = layers.Dense(16, activation='relu')(x)
outputs = layers.concatenate([layers.Dense(4, activation='softmax', name='out_' + str(i))(x)
                              for i in range(64)])  # 64 softmax layers
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# model.summary()


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


model.compile(
    optimizer=optimizers.Nadam(),
    loss=MultiCrossEntropy(),  # same loss func for 64 softmax units
    metrics=[acc_of_all, acc_of_valid]
)

'''
4. Train the Model (Custom cycle-training model)
'''

# model.load_weights("CENet/V1.2/20210402-105120/ckpt/cp-0100.ckpt")

# Callback 1 -- tensorboard
# Execute "!tensorboard --logdir CENet\V1.2" in Ipython, if in Windows, to display TensorBoard.
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=4, profile_batch=0)
# Callback 2 -- logging learning_rate
file_writer = tf.summary.create_file_writer(os.path.join(logdir, 'metrics'))
file_writer.set_as_default()
# Callback 3 -- saving weights
checkpoint_path = os.path.join(logdir, 'ckpts', 'cp-{epoch:04d}.ckpt')
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1,
                                                 save_weights_only=True, period=40)


# Callback 4 -- learning_rate decay
def lr_schedule(epoch):
    learning_rate = 0.001
    if epoch > 150:
        learning_rate = 0.0005
    if epoch > 180:
        learning_rate = 0.00001
    if epoch > 200:
        learning_rate = 0.000001

    tf.summary.scalar('learning_rate', data=learning_rate, step=epoch)  # Callback 3

    return learning_rate


lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

print('\nSTART TRAINING!\n')
model.fit(to_train,
          # batch_size=BATCH_SIZE,
          validation_data=to_test,
          epochs=250,
          callbacks=[tensorboard_callback, lr_callback, cp_callback],
          workers=4,
          )
# Epoch 250:
# acc_of_all: 0.5743 - acc_of_valid: 0.4324
# val_acc_of_all: 0.5663 - val_acc_of_valid: 0.4217


'''
5. Save the Model
'''

tf.keras.utils.plot_model(model, os.path.join(logdir, 'CENet-'+__file__[-3-4:-3]+'.png'),
                          show_shapes=True, dpi=300)
# 1.1
# Saved as model.h5
model.save(os.path.join(logdir, 'CENet-'+__file__[-3-4:-3]+'.h5'))  # the old Keras H5 format


'''
6. Use the Model
'''

aa = model.predict(to_test)
bb = aa.reshape(-1, 4)
cc = np.argmax(bb, axis=1).reshape(-1, 64).astype(np.int)  # onehot to 0~3
# np.save("./data_sets/demodu_CENet.npy", cc)
# Pe = 0.5783125, BER = 0.3911875. From 7.BER_calculation.py
# 0.5783 + 0.4217 = 1

